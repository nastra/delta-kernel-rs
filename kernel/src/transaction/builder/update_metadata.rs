//! Protocol and metadata preparation for transactions against an existing table.
//!
//! [`prepare_update_table_metadata`] merges the requested schema and configuration changes with the
//! read snapshot, derives the table features and properties that the final schema and configuration
//! require, and validates the result. It mirrors Delta Kernel Java's
//! `TransactionMetadataFactory.buildUpdateTableMetadata`.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::metadata_derivation::{
    ValidatedTableProperties, maybe_auto_enable_property_driven_features, maybe_enable_collations,
    maybe_enable_file_type_preview, maybe_enable_ict_for_catalog_managed,
    maybe_enable_timestamp_ntz, maybe_enable_v2_checkpoint_for_policy, maybe_enable_variant_type,
    validate_and_normalize_delta_properties, validate_extract_features_and_properties_for,
};
use crate::actions::{Metadata, Protocol};
use crate::schema::{ColumnMetadataKey, MetadataValue, SchemaRef, StructField, StructType};
use crate::snapshot::Snapshot;
use crate::table_configuration::TableConfiguration;
use crate::table_features::{
    ColumnMappingMode, IcebergCompatValidationContext, LEGACY_WRITER_FEATURES,
    TABLE_FEATURES_MIN_READER_VERSION, TABLE_FEATURES_MIN_WRITER_VERSION, TableFeature,
    V2_VALIDATOR, V3_VALIDATOR, assign_column_mapping_metadata, assign_nested_cm_ids,
    drop_column_mapping_metadata, ensure_domain_metadata_dependency,
    extract_enabled_reader_features, find_max_column_id_in_schema,
    get_column_mapping_mode_from_properties, validate_iceberg_compat_if_needed,
};
use crate::table_properties::{
    CHECKPOINT_INTERVAL, COLUMN_MAPPING_MAX_COLUMN_ID, DATA_SKIPPING_NUM_INDEXED_COLS,
    DELTA_PROPERTY_PREFIX, ENABLE_ICEBERG_COMPAT_V2, ENABLE_ICEBERG_COMPAT_V3,
    ENABLE_ICEBERG_NATIVE_V4, ENABLE_IN_COMMIT_TIMESTAMPS,
    IN_COMMIT_TIMESTAMP_ENABLEMENT_TIMESTAMP, IN_COMMIT_TIMESTAMP_ENABLEMENT_VERSION,
    MATERIALIZED_ROW_COMMIT_VERSION_COLUMN_NAME, MATERIALIZED_ROW_ID_COLUMN_NAME,
    PARQUET_FORMAT_VERSION, SET_TRANSACTION_RETENTION_DURATION, TableProperties,
};
use crate::transaction::schema_evolution::{SchemaOperation, evolve_metadata};
use crate::transforms::{SchemaTransform, transform_output_type};
use crate::{Engine, KernelError, KernelResult};

/// Universal formats (UniForm) the table is converted to, as a comma-separated list.
const UNIVERSAL_FORMAT_ENABLED_FORMATS: &str = "delta.universalFormat.enabledFormats";

/// `delta.*` properties CREATE TABLE accepts but an update rejects, because Java's editable
/// `TableConfig.VALID_PROPERTIES` does not contain them. `delta.enableIcebergNativeV4` exists
/// only under the `adaptive-metadata-in-dev` feature.
const NOT_SETTABLE_ON_UPDATE: &[&str] = &[
    SET_TRANSACTION_RETENTION_DURATION,
    PARQUET_FORMAT_VERSION,
    ENABLE_ICEBERG_NATIVE_V4,
];

/// Operation name used in property validation errors.
const UPDATE_TABLE: &str = "UPDATE TABLE";

/// Schema and configuration changes requested for an existing table.
#[derive(Debug, Default)]
pub(super) struct UpdateTableMetadataRequest {
    /// Schema changes applied in order to the snapshot schema.
    pub(super) schema_changes: Option<Vec<SchemaOperation>>,
    /// Table properties to set. May contain `delta.feature.<name>=supported` signals.
    pub(super) properties_added: HashMap<String, String>,
    /// Table property keys to unset.
    pub(super) property_keys_removed: HashSet<String>,
}

impl UpdateTableMetadataRequest {
    /// Returns `true` if the request changes neither the schema nor the configuration.
    pub(super) fn is_empty(&self) -> bool {
        self.schema_changes.is_none()
            && self.properties_added.is_empty()
            && self.property_keys_removed.is_empty()
    }
}

/// Table configuration a transaction against an existing table commits.
#[derive(Debug)]
pub(super) struct PreparedUpdateTableMetadata {
    pub(super) table_config: TableConfiguration,
    pub(super) should_emit_metadata: bool,
    pub(super) should_emit_protocol: bool,
}

/// Merges `request` with the read snapshot and derives the protocol and metadata to commit.
///
/// The result retains every protocol feature of the read snapshot, preserves column-mapping
/// identities, materialized row-tracking column names, and existing in-commit timestamp
/// enablement metadata. Features required by the final schema or configuration are added to the
/// protocol. When the transaction enables in-commit timestamps, the enablement version and
/// timestamp are recorded for the commit at `read_snapshot.version() + 1` and `commit_timestamp`.
///
/// Enabling column mapping mode `name` on a table without column mapping assigns ids to the
/// existing fields and uses their current names as physical names, since existing data files were
/// written with those names.
///
/// Rejected transitions on existing tables: column mapping mode changes other than `none` ->
/// `name`, a column mapping mode change combined with schema changes, enabling or disabling row
/// tracking, enabling or disabling `icebergCompatV3`, and disabling in-commit timestamps on a
/// catalog-managed table.
pub(super) fn prepare_update_table_metadata(
    read_snapshot: &Snapshot,
    request: UpdateTableMetadataRequest,
    commit_timestamp: i64,
    engine: &dyn Engine,
) -> KernelResult<PreparedUpdateTableMetadata> {
    let read_config = read_snapshot.table_configuration();
    if request.is_empty() {
        return Ok(PreparedUpdateTableMetadata {
            table_config: read_config.clone(),
            should_emit_metadata: false,
            should_emit_protocol: false,
        });
    }
    let UpdateTableMetadataRequest {
        schema_changes,
        properties_added,
        property_keys_removed,
    } = request;
    let properties_added = validate_and_normalize_delta_properties(properties_added, UPDATE_TABLE)?;
    if let Some(key) = properties_added
        .keys()
        .find(|key| NOT_SETTABLE_ON_UPDATE.contains(&key.as_str()))
    {
        return Err(KernelError::generic(format!(
            "Setting delta property '{key}' is not supported during {UPDATE_TABLE}"
        )));
    }
    validate_java_int_ranges(&properties_added)?;
    validate_property_removals(&properties_added, &property_keys_removed)?;

    // Schema changes preserve the column-mapping identities of surviving fields and assign new
    // identities (and `maxColumnId`) to added fields. The evolved schema is validated against the
    // final protocol below, after deriving the features it requires.
    let has_schema_changes = schema_changes.is_some();
    let (mut evolved_metadata, mut schema) = match schema_changes {
        Some(changes) => apply_schema_changes(read_config, changes)?,
        None => (read_config.metadata().clone(), read_config.logical_schema()),
    };

    // Kernel-managed keys are rejected by the property allow-list.
    let mut requested =
        validate_extract_features_and_properties_for(properties_added.clone(), UPDATE_TABLE)?;
    let mut configuration = evolved_metadata.configuration().clone();
    configuration.extend(requested.properties.drain());
    configuration.retain(|key, _| !property_keys_removed.contains(key));
    validate_transitions(read_config, &configuration, has_schema_changes)?;
    let new_mode = get_column_mapping_mode_from_properties(&configuration)?;
    let enables_column_mapping =
        get_column_mapping_mode_from_properties(read_config.metadata().configuration())?
            != new_mode;
    if enables_column_mapping {
        schema = assign_column_mapping_to_existing_fields(&schema, &mut configuration)?;
    }
    // Like Java's `rewriteFieldIdsForIceberg`, IcebergCompat requires ids for the array element and
    // map key/value slots of a column-mapped schema.
    let iceberg_compat_enabled = |configuration: &HashMap<String, String>| {
        [ENABLE_ICEBERG_COMPAT_V2, ENABLE_ICEBERG_COMPAT_V3]
            .iter()
            .any(|key| configuration.get(*key).map(String::as_str) == Some("true"))
    };
    let enables_iceberg_compat = iceberg_compat_enabled(&configuration)
        && !iceberg_compat_enabled(read_config.metadata().configuration());
    if new_mode != ColumnMappingMode::None
        && iceberg_compat_enabled(&configuration)
        && (enables_column_mapping || enables_iceberg_compat)
    {
        schema = assign_iceberg_compat_nested_ids(&schema, &mut configuration)?;
    }
    if enables_column_mapping || enables_iceberg_compat {
        evolved_metadata = evolved_metadata.with_schema(schema.clone())?;
    }
    let explicit_ict = properties_added
        .get(ENABLE_IN_COMMIT_TIMESTAMPS)
        .map(String::as_str);

    // Already supported features need no protocol change, and must not re-trigger the
    // enablement rules for newly added features (such as catalogManaged -> inCommitTimestamp).
    requested
        .reader_features
        .retain(|feature| !is_already_supported(read_config, feature));
    requested
        .writer_features
        .retain(|feature| !is_already_supported(read_config, feature));
    let enables_catalog_managed = requested
        .writer_features
        .contains(&TableFeature::CatalogManaged);
    match explicit_ict {
        Some(value) if enables_catalog_managed && value.eq_ignore_ascii_case("false") => {
            return Err(KernelError::generic(
                "Cannot disable inCommitTimestamp when enabling catalogManaged",
            ));
        }
        None if enables_catalog_managed => {
            configuration.insert(ENABLE_IN_COMMIT_TIMESTAMPS.to_string(), "true".to_string());
        }
        _ => {}
    }
    requested.properties = configuration;

    maybe_enable_variant_type(&schema, &mut requested);
    maybe_enable_timestamp_ntz(&schema, &mut requested);
    maybe_enable_collations(&schema, &mut requested);
    maybe_enable_file_type_preview(&schema, &mut requested);
    maybe_auto_enable_property_driven_features(&mut requested);
    maybe_enable_ict_for_catalog_managed(&mut requested)?;
    maybe_enable_v2_checkpoint_for_policy(&mut requested);
    ensure_domain_metadata_dependency(
        &mut requested.reader_features,
        &mut requested.writer_features,
    );
    let ValidatedTableProperties {
        properties: mut configuration,
        mut reader_features,
        mut writer_features,
    } = requested;
    reader_features.retain(|feature| !is_already_supported(read_config, feature));
    writer_features.retain(|feature| !is_already_supported(read_config, feature));
    let should_emit_protocol = !writer_features.is_empty();
    let protocol = if should_emit_protocol {
        upgrade_protocol(read_config, reader_features, writer_features)?
    } else {
        read_config.protocol().clone()
    };

    if explicit_ict.is_some_and(|value| value.eq_ignore_ascii_case("false")) {
        if protocol.has_table_feature(&TableFeature::CatalogManaged) {
            return Err(KernelError::generic(
                "Cannot disable inCommitTimestamp on a catalogManaged table",
            ));
        }
        configuration.remove(IN_COMMIT_TIMESTAMP_ENABLEMENT_VERSION);
        configuration.remove(IN_COMMIT_TIMESTAMP_ENABLEMENT_TIMESTAMP);
    }

    let mut table_config = TableConfiguration::try_new(
        evolved_metadata.with_configuration(configuration),
        protocol,
        read_config.table_root().clone(),
        read_config.version(),
    )?;

    if !read_config.is_feature_enabled(&TableFeature::InCommitTimestamp)
        && table_config.is_feature_enabled(&TableFeature::InCommitTimestamp)
    {
        // Follows the in-commit timestamp rule for existing tables: no earlier than one
        // millisecond after the previous commit.
        let previous_timestamp = read_snapshot.reader().get_timestamp(engine)?;
        let in_commit_timestamp = commit_timestamp.max(previous_timestamp + 1);
        let metadata = table_config
            .metadata()
            .clone()
            .with_configuration_entry(
                IN_COMMIT_TIMESTAMP_ENABLEMENT_VERSION,
                (read_snapshot.version() + 1).to_string(),
            )
            .with_configuration_entry(
                IN_COMMIT_TIMESTAMP_ENABLEMENT_TIMESTAMP,
                in_commit_timestamp.to_string(),
            );
        table_config = TableConfiguration::try_new_with_schema(
            &table_config,
            metadata,
            table_config.logical_schema(),
        )?;
    }

    validate_row_tracking_column_names(&table_config)?;
    let should_emit_metadata =
        has_schema_changes || table_config.metadata() != read_config.metadata();
    // A metadata-only change can break a feature requirement without changing the protocol, so
    // the final configuration is validated whenever either action is emitted.
    if should_emit_metadata || should_emit_protocol {
        table_config.ensure_write_supported()?;
        for validator in [&V2_VALIDATOR, &V3_VALIDATOR] {
            validate_iceberg_compat_if_needed(
                &table_config,
                validator,
                IcebergCompatValidationContext::Write,
            )?;
        }
        validate_iceberg_compat_v2_without_deletion_vectors(read_config, &table_config)?;
        validate_universal_format(table_config.metadata().configuration())?;
    }
    Ok(PreparedUpdateTableMetadata {
        table_config,
        should_emit_metadata,
        should_emit_protocol,
    })
}

/// Returns `true` if `table_config` supports `feature` or, for a stable feature with a preview
/// predecessor, the preview feature. Adding the stable feature next to a supported preview feature
/// could break clients that only understand the preview feature.
fn is_already_supported(table_config: &TableConfiguration, feature: &TableFeature) -> bool {
    let preview = match feature {
        TableFeature::VariantType => Some(TableFeature::VariantTypePreview),
        TableFeature::TypeWidening => Some(TableFeature::TypeWideningPreview),
        TableFeature::Collations => Some(TableFeature::CollationsPreview),
        _ => None,
    };
    table_config.is_feature_supported(feature)
        || preview.is_some_and(|preview| table_config.is_feature_supported(&preview))
}

/// Applies `changes` to the table schema, rejecting tables whose schema Kernel cannot yet evolve.
fn apply_schema_changes(
    table_config: &TableConfiguration,
    changes: Vec<SchemaOperation>,
) -> KernelResult<(Metadata, SchemaRef)> {
    let unsupported_iceberg_compat = [TableFeature::IcebergCompatV2, TableFeature::IcebergCompatV3]
        .into_iter()
        .find(|feature| table_config.is_feature_enabled(feature));
    if let Some(feature) = unsupported_iceberg_compat {
        return Err(KernelError::unsupported(format!(
            "Schema changes are not yet supported on tables with {feature} enabled"
        )));
    }
    if table_config.is_feature_enabled(&TableFeature::AllowColumnDefaults) {
        return Err(KernelError::unsupported(
            "Schema changes are not yet supported on tables with allowColumnDefaults enabled",
        ));
    }
    if table_config.is_feature_supported(&TableFeature::GeneratedColumns)
        && table_config.has_generation_expressions()
    {
        return Err(KernelError::unsupported(
            "Schema changes are not yet supported on tables with generated columns",
        ));
    }
    evolve_metadata(table_config, changes)
}

/// Rejects integer properties outside the range Java accepts. Java parses them with
/// `Integer.valueOf` both when it validates them and when it reads the table configuration, so a
/// larger value would break Java and Spark writers.
fn validate_java_int_ranges(properties_added: &HashMap<String, String>) -> KernelResult<()> {
    for (key, min) in [
        (CHECKPOINT_INTERVAL, 1),
        (DATA_SKIPPING_NUM_INDEXED_COLS, -1),
    ] {
        let Some(value) = properties_added.get(key) else {
            continue;
        };
        let in_range = value
            .parse::<i64>()
            .is_ok_and(|number| (min..=i64::from(i32::MAX)).contains(&number));
        if !in_range {
            return Err(KernelError::generic(format!(
                "Invalid value '{value}' for table property '{key}': must be between {min} and {}",
                i32::MAX
            )));
        }
    }
    Ok(())
}

fn validate_property_removals(
    properties_added: &HashMap<String, String>,
    property_keys_removed: &HashSet<String>,
) -> KernelResult<()> {
    let mut overlapping: Vec<_> = property_keys_removed
        .iter()
        .filter(|key| properties_added.contains_key(*key))
        .map(String::as_str)
        .collect();
    if !overlapping.is_empty() {
        overlapping.sort_unstable();
        return Err(KernelError::generic(format!(
            "Table properties cannot be both set and unset in one transaction: {}",
            overlapping.join(", ")
        )));
    }
    // Like Java's `withTablePropertiesRemoved`, no `delta.*` key may be unset, whatever its case.
    // Unsetting one would bypass the transition rules and the Kernel-managed keys.
    if let Some(key) = property_keys_removed
        .iter()
        .find(|key| key.to_ascii_lowercase().starts_with(DELTA_PROPERTY_PREFIX))
    {
        return Err(KernelError::generic(format!(
            "Unsetting table property '{key}' is not supported during {UPDATE_TABLE}"
        )));
    }
    Ok(())
}

/// Rejects configuration transitions that Kernel does not support on existing tables.
fn validate_transitions(
    old: &TableConfiguration,
    new_configuration: &HashMap<String, String>,
    has_schema_changes: bool,
) -> KernelResult<()> {
    let old_properties = old.table_properties();
    let new_properties = TableProperties::from(new_configuration.iter());
    // Only `none` -> `name` keeps existing data files readable without rewriting them: the
    // current field names become the physical names.
    let old_mode = get_column_mapping_mode_from_properties(old.metadata().configuration())?;
    let new_mode = get_column_mapping_mode_from_properties(new_configuration)?;
    if old_mode != new_mode {
        if (old_mode, new_mode) != (ColumnMappingMode::None, ColumnMappingMode::Name) {
            return Err(KernelError::unsupported(format!(
                "Changing the column mapping mode from {old_mode:?} to {new_mode:?} is not \
                 supported on an existing table"
            )));
        }
        if has_schema_changes {
            return Err(KernelError::unsupported(
                "Cannot update the column mapping mode and perform schema changes in one \
                 transaction",
            ));
        }
    }
    // Enabling requires backfilling row IDs and row commit versions; disabling is irreversible.
    if old_properties.enable_row_tracking.unwrap_or(false)
        != new_properties.enable_row_tracking.unwrap_or(false)
    {
        return Err(KernelError::unsupported(
            "Enabling or disabling row tracking is not supported on an existing table",
        ));
    }
    let iceberg_compat_v3_enabled = |configuration: &HashMap<String, String>| {
        configuration
            .get(ENABLE_ICEBERG_COMPAT_V3)
            .is_some_and(|value| value.eq_ignore_ascii_case("true"))
    };
    match (
        iceberg_compat_v3_enabled(old.metadata().configuration()),
        iceberg_compat_v3_enabled(new_configuration),
    ) {
        (false, true) => Err(KernelError::unsupported(format!(
            "Enabling '{ENABLE_ICEBERG_COMPAT_V3}' is not supported on an existing table"
        ))),
        (true, false) => Err(KernelError::unsupported(format!(
            "Disabling '{ENABLE_ICEBERG_COMPAT_V3}' is not supported on an existing table"
        ))),
        _ => Ok(()),
    }
}

/// Rejects `icebergCompatV2` when the final or the read protocol supports `deletionVectors`, as
/// Java's `CHECK_HAS_NO_DELETION_VECTORS` does. Files written while deletion vectors were enabled
/// may still carry them, so supported is enough to reject, and the read protocol is checked too.
fn validate_iceberg_compat_v2_without_deletion_vectors(
    read_config: &TableConfiguration,
    table_config: &TableConfiguration,
) -> KernelResult<()> {
    let supports_deletion_vectors = [read_config, table_config]
        .iter()
        .any(|config| config.is_feature_supported(&TableFeature::DeletionVectors));
    if table_config.is_feature_enabled(&TableFeature::IcebergCompatV2) && supports_deletion_vectors
    {
        return Err(KernelError::unsupported(format!(
            "Table feature '{}' is incompatible with '{}'",
            TableFeature::DeletionVectors,
            TableFeature::IcebergCompatV2
        )));
    }
    Ok(())
}

/// Requires exactly one of `icebergCompatV2` and `icebergCompatV3` to be enabled when
/// `delta.universalFormat.enabledFormats` contains `iceberg`, and at most one whenever the property
/// is set. Mirrors Java's `IcebergUniversalFormatMetadataValidatorAndUpdater.validate`.
fn validate_universal_format(configuration: &HashMap<String, String>) -> KernelResult<()> {
    let Some(formats) = configuration.get(UNIVERSAL_FORMAT_ENABLED_FORMATS) else {
        return Ok(());
    };
    let iceberg_enabled = formats.split(',').any(|format| format.trim() == "iceberg");
    let compat_keys = [ENABLE_ICEBERG_COMPAT_V2, ENABLE_ICEBERG_COMPAT_V3];
    let enabled_compat_count = compat_keys
        .iter()
        .filter(|key| {
            configuration
                .get(**key)
                .is_some_and(|value| value.eq_ignore_ascii_case("true"))
        })
        .count();
    if iceberg_enabled && enabled_compat_count == 0 {
        return Err(KernelError::generic(format!(
            "One of '{}' must be set to \"true\" to enable the iceberg universal format in \
             '{UNIVERSAL_FORMAT_ENABLED_FORMATS}'",
            compat_keys.join("' or '")
        )));
    }
    if enabled_compat_count > 1 {
        return Err(KernelError::generic(format!(
            "'{}' cannot be enabled at the same time",
            compat_keys.join("' and '")
        )));
    }
    Ok(())
}

/// Requires materialized row-tracking column names when row tracking is enabled, and rejects
/// configured names that collide with a top-level logical or physical column name, whether or not
/// row tracking is enabled.
fn validate_row_tracking_column_names(table_config: &TableConfiguration) -> KernelResult<()> {
    let row_tracking_enabled = table_config.table_properties().enable_row_tracking == Some(true);
    let schema = table_config.logical_schema();
    let column_mapping_mode = table_config.column_mapping_mode();
    for key in [
        MATERIALIZED_ROW_ID_COLUMN_NAME,
        MATERIALIZED_ROW_COMMIT_VERSION_COLUMN_NAME,
    ] {
        let Some(name) = table_config.metadata().configuration().get(key) else {
            if row_tracking_enabled {
                return Err(KernelError::generic(format!(
                    "Row tracking is enabled but the materialized column name '{key}' is missing"
                )));
            }
            continue;
        };
        if schema
            .fields()
            .any(|field| field.name() == name || field.physical_name(column_mapping_mode) == name)
        {
            return Err(KernelError::generic(format!(
                "Materialized row-tracking column name '{name}' conflicts with a table column"
            )));
        }
    }
    Ok(())
}

/// Assigns column-mapping ids and physical names to every field of `schema` and records the new
/// `delta.columnMapping.maxColumnId` in `configuration`. Each physical name is the field's current
/// name, as in Java's `ColumnMapping.assignColumnIdAndPhysicalName(metadata, isNewTable = false)`.
///
/// Column-mapping annotations left over from earlier history are dropped first: with mapping
/// disabled they do not describe the data files, which were written with the field names.
fn assign_column_mapping_to_existing_fields(
    schema: &StructType,
    configuration: &mut HashMap<String, String>,
) -> KernelResult<SchemaRef> {
    struct FieldNameAsPhysicalName;
    impl<'a> SchemaTransform<'a> for FieldNameAsPhysicalName {
        transform_output_type!(|'a, T| Cow<'a, T>);
        fn transform_struct_field(&mut self, field: &'a StructField) -> Cow<'a, StructField> {
            let mut field = field.clone();
            field.data_type = self.transform(&field.data_type).into_owned();
            field.metadata.insert(
                ColumnMetadataKey::ColumnMappingPhysicalName.as_ref().into(),
                MetadataValue::String(field.name.clone()),
            );
            Cow::Owned(field)
        }
        // Like CREATE TABLE, column mapping does not reach the fields inside variant and file
        // values.
        fn transform_variant(&mut self, stype: &'a StructType) -> Cow<'a, StructType> {
            Cow::Borrowed(stype)
        }
        fn transform_file(&mut self, stype: &'a StructType) -> Cow<'a, StructType> {
            Cow::Borrowed(stype)
        }
    }
    let schema = drop_column_mapping_metadata(schema);
    let schema = FieldNameAsPhysicalName
        .transform_struct(&schema)
        .into_owned();
    let mut max_id = TableProperties::from(configuration.iter())
        .column_mapping_max_column_id
        .unwrap_or(0);
    let schema = assign_column_mapping_metadata(&schema, &mut max_id, false)?;
    configuration.insert(COLUMN_MAPPING_MAX_COLUMN_ID.to_string(), max_id.to_string());
    Ok(Arc::new(schema))
}

/// Assigns `delta.columnMapping.nested.ids` to the array and map slots of a column-mapped `schema`
/// that have none, and records the new `delta.columnMapping.maxColumnId` in `configuration`.
fn assign_iceberg_compat_nested_ids(
    schema: &StructType,
    configuration: &mut HashMap<String, String>,
) -> KernelResult<SchemaRef> {
    let mut max_id = TableProperties::from(configuration.iter())
        .column_mapping_max_column_id
        .unwrap_or(0)
        .max(find_max_column_id_in_schema(schema).unwrap_or(0));
    let schema = assign_nested_cm_ids(schema, &mut max_id)?;
    configuration.insert(COLUMN_MAPPING_MAX_COLUMN_ID.to_string(), max_id.to_string());
    Ok(Arc::new(schema))
}

/// Adds `reader_features` and `writer_features` to the protocol of `table_config`, retaining every
/// feature the protocol already supports, including features implied by legacy versions.
fn upgrade_protocol(
    table_config: &TableConfiguration,
    new_reader_features: Vec<TableFeature>,
    new_writer_features: Vec<TableFeature>,
) -> KernelResult<Protocol> {
    let protocol = table_config.protocol();
    let mut writer_features = match protocol.writer_features() {
        Some(features) => features.to_vec(),
        None => LEGACY_WRITER_FEATURES
            .iter()
            .filter(|feature| table_config.is_feature_supported(feature))
            .cloned()
            .collect(),
    };
    writer_features.extend(new_writer_features);
    let reader_features = (protocol.min_reader_version() == TABLE_FEATURES_MIN_READER_VERSION
        || !new_reader_features.is_empty())
    .then(|| {
        let mut features = extract_enabled_reader_features(protocol);
        features.extend(new_reader_features);
        features
    });
    let min_reader_version = if reader_features.is_some() {
        TABLE_FEATURES_MIN_READER_VERSION
    } else {
        protocol.min_reader_version()
    };
    Protocol::try_new(
        min_reader_version,
        TABLE_FEATURES_MIN_WRITER_VERSION,
        reader_features,
        Some(writer_features),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;

    use super::*;
    use crate::committer::FileSystemCommitter;
    use crate::engine::sync::SyncEngine;
    use crate::object_store::memory::InMemory;
    use crate::schema::{ArrayType, DataType, MapType};
    use crate::snapshot::SnapshotRef;
    use crate::table_configuration::InCommitTimestampEnablement;
    use crate::table_features::SET_TABLE_FEATURE_SUPPORTED_PREFIX;
    use crate::table_properties::{CHECKPOINT_POLICY, COLUMN_MAPPING_MODE, ENABLE_ROW_TRACKING};
    use crate::transaction::create_table::create_table;
    use crate::unit_test_utils::{load_test_table, test_schema_flat};
    use crate::utils::current_time_ms;

    fn create_test_table(properties: &[(&str, &str)]) -> (Arc<SyncEngine>, SnapshotRef) {
        let engine = Arc::new(SyncEngine::new_with_store(Arc::new(InMemory::new())));
        let snapshot = create_table("memory:///", test_schema_flat(), "test", true)
            .with_table_properties(properties.iter().copied())
            .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))
            .unwrap()
            .commit(engine.as_ref())
            .unwrap()
            .unwrap_post_commit_snapshot();
        (engine, snapshot)
    }

    fn properties_request(properties: &[(&str, &str)]) -> UpdateTableMetadataRequest {
        UpdateTableMetadataRequest {
            properties_added: properties
                .iter()
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .collect(),
            ..Default::default()
        }
    }

    fn add_column_request(field: StructField) -> UpdateTableMetadataRequest {
        UpdateTableMetadataRequest {
            schema_changes: Some(vec![SchemaOperation::add_column(None, field)]),
            ..Default::default()
        }
    }

    fn prepare(
        engine: &dyn Engine,
        snapshot: &Snapshot,
        request: UpdateTableMetadataRequest,
    ) -> KernelResult<PreparedUpdateTableMetadata> {
        prepare_update_table_metadata(snapshot, request, current_time_ms()?, engine)
    }

    fn with_table_config(snapshot: &Snapshot, table_config: TableConfiguration) -> Snapshot {
        Snapshot::new_with_crc(
            snapshot.log_segment().clone(),
            table_config,
            None,
            true,
            false,
        )
        .unwrap()
    }

    /// Returns `snapshot` with `key` set to `value`, or removed when `value` is `None`, in its
    /// metadata. Used for table states that CREATE TABLE cannot produce.
    fn with_configuration_entry(snapshot: &Snapshot, key: &str, value: Option<&str>) -> Snapshot {
        let read_config = snapshot.table_configuration();
        let mut configuration = read_config.metadata().configuration().clone();
        match value {
            Some(value) => configuration.insert(key.to_string(), value.to_string()),
            None => configuration.remove(key),
        };
        let metadata = read_config
            .metadata()
            .clone()
            .with_configuration(configuration);
        let table_config = TableConfiguration::try_new_with_schema(
            read_config,
            metadata,
            read_config.logical_schema(),
        )
        .unwrap();
        with_table_config(snapshot, table_config)
    }

    fn configuration_value<'a>(config: &'a TableConfiguration, key: &str) -> Option<&'a str> {
        config
            .metadata()
            .configuration()
            .get(key)
            .map(String::as_str)
    }

    #[test]
    fn empty_request_keeps_read_configuration() {
        let (engine, snapshot) = create_test_table(&[]);
        let prepared = prepare(engine.as_ref(), &snapshot, Default::default()).unwrap();
        assert!(!prepared.should_emit_metadata);
        assert!(!prepared.should_emit_protocol);
        assert_eq!(
            prepared.table_config.metadata(),
            snapshot.table_configuration().metadata()
        );
    }

    #[test]
    fn unchanged_property_does_not_emit_metadata() {
        let (engine, snapshot) = create_test_table(&[("app.key", "value")]);
        let prepared = prepare(
            engine.as_ref(),
            &snapshot,
            properties_request(&[("app.key", "value")]),
        )
        .unwrap();
        assert!(!prepared.should_emit_metadata);
        assert!(!prepared.should_emit_protocol);
    }

    #[rstest]
    #[case::legacy_protocol(true)]
    #[case::table_features_protocol(false)]
    fn derived_feature_retains_existing_protocol_features(#[case] legacy: bool) {
        let (engine, snapshot, _tempdir) = if legacy {
            load_test_table("table-without-dv-small").unwrap()
        } else {
            let (engine, snapshot) = create_test_table(&[
                ("delta.enableDeletionVectors", "true"),
                ("delta.appendOnly", "true"),
            ]);
            (engine as Arc<dyn Engine>, snapshot, None)
        };
        let read_config = snapshot.table_configuration();
        let retained: Vec<_> = [
            TableFeature::AppendOnly,
            TableFeature::Invariants,
            TableFeature::DeletionVectors,
        ]
        .into_iter()
        .filter(|feature| read_config.is_feature_supported(feature))
        .collect();
        assert!(!retained.is_empty());

        let prepared = prepare(
            engine.as_ref(),
            &snapshot,
            add_column_request(StructField::nullable("ts", DataType::TIMESTAMP_NTZ)),
        )
        .unwrap();

        assert!(prepared.should_emit_protocol);
        assert!(prepared.should_emit_metadata);
        let config = &prepared.table_config;
        assert!(config.is_feature_supported(&TableFeature::TimestampWithoutTimezone));
        for feature in &retained {
            assert!(
                config.is_feature_supported(feature),
                "{feature} was dropped"
            );
        }
    }

    #[rstest]
    #[case::timestamp_ntz(DataType::TIMESTAMP_NTZ, TableFeature::TimestampWithoutTimezone)]
    #[case::variant(DataType::unshredded_variant(), TableFeature::VariantType)]
    #[case::file(DataType::file_type(), TableFeature::FileTypePreview)]
    fn schema_change_derives_type_feature(
        #[case] data_type: DataType,
        #[case] feature: TableFeature,
    ) {
        let (engine, snapshot) = create_test_table(&[]);
        let prepared = prepare(
            engine.as_ref(),
            &snapshot,
            add_column_request(StructField::nullable("added", data_type)),
        )
        .unwrap();
        assert!(prepared.should_emit_protocol);
        assert!(prepared.table_config.is_feature_supported(&feature));
    }

    #[rstest]
    #[case::v2_checkpoint(&[(CHECKPOINT_POLICY, "v2")], TableFeature::V2Checkpoint, true)]
    #[case::property_driven(
        &[("delta.enableChangeDataFeed", "true")],
        TableFeature::ChangeDataFeed,
        true
    )]
    #[case::protocol_only_feature_signal(
        &[("delta.feature.domainMetadata", "supported")],
        TableFeature::DomainMetadata,
        false
    )]
    fn property_request_derives_required_feature(
        #[case] properties: &[(&str, &str)],
        #[case] feature: TableFeature,
        #[case] should_emit_metadata: bool,
    ) {
        let (engine, snapshot) = create_test_table(&[]);
        assert!(
            !snapshot
                .table_configuration()
                .is_feature_supported(&feature)
        );
        let prepared = prepare(engine.as_ref(), &snapshot, properties_request(properties)).unwrap();
        assert!(prepared.should_emit_protocol);
        assert_eq!(prepared.should_emit_metadata, should_emit_metadata);
        assert!(prepared.table_config.is_feature_supported(&feature));
        assert!(
            !prepared
                .table_config
                .metadata()
                .configuration()
                .keys()
                .any(|key| key.starts_with(SET_TABLE_FEATURE_SUPPORTED_PREFIX))
        );
    }

    #[test]
    fn already_supported_feature_does_not_emit_protocol() {
        let (engine, snapshot) = create_test_table(&[(CHECKPOINT_POLICY, "v2")]);
        let prepared = prepare(
            engine.as_ref(),
            &snapshot,
            properties_request(&[("app.key", "value")]),
        )
        .unwrap();
        assert!(!prepared.should_emit_protocol);
        assert!(prepared.should_emit_metadata);
        assert_eq!(
            prepared.table_config.protocol(),
            snapshot.table_configuration().protocol()
        );
    }

    #[test]
    fn catalog_managed_enablement_derives_in_commit_timestamps() {
        let (engine, snapshot) = create_test_table(&[]);
        let prepared = prepare(
            engine.as_ref(),
            &snapshot,
            properties_request(&[("delta.feature.catalogManaged", "supported")]),
        )
        .unwrap();
        let config = &prepared.table_config;
        assert!(config.is_feature_supported(&TableFeature::CatalogManaged));
        assert!(config.is_feature_enabled(&TableFeature::InCommitTimestamp));
        assert_eq!(
            configuration_value(config, ENABLE_IN_COMMIT_TIMESTAMPS),
            Some("true")
        );
        let InCommitTimestampEnablement::Enabled {
            enablement: Some((version, _)),
        } = config.in_commit_timestamp_enablement().unwrap()
        else {
            panic!("expected recorded enablement metadata");
        };
        assert_eq!(version, snapshot.version() + 1);
    }

    #[test]
    fn disabling_in_commit_timestamps_on_catalog_managed_table_is_rejected() {
        let (engine, snapshot) = create_test_table(&[]);
        let catalog_managed = prepare(
            engine.as_ref(),
            &snapshot,
            properties_request(&[("delta.feature.catalogManaged", "supported")]),
        )
        .unwrap();
        let snapshot = with_table_config(&snapshot, catalog_managed.table_config);
        let error = prepare(
            engine.as_ref(),
            &snapshot,
            properties_request(&[(ENABLE_IN_COMMIT_TIMESTAMPS, "false")]),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Cannot disable inCommitTimestamp on a catalogManaged table"),
            "{error}"
        );
    }

    #[test]
    fn catalog_managed_enablement_rejects_disabled_in_commit_timestamps() {
        let (engine, snapshot) = create_test_table(&[]);
        let error = prepare(
            engine.as_ref(),
            &snapshot,
            properties_request(&[
                ("delta.feature.catalogManaged", "supported"),
                (ENABLE_IN_COMMIT_TIMESTAMPS, "false"),
            ]),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Cannot disable inCommitTimestamp when enabling catalogManaged"),
            "{error}"
        );
    }

    #[test]
    fn schema_change_preserves_column_mapping_identities() {
        let (engine, snapshot) = create_test_table(&[(COLUMN_MAPPING_MODE, "name")]);
        let read_config = snapshot.table_configuration();
        let read_max_id = read_config.table_properties().column_mapping_max_column_id;
        let mut request = add_column_request(StructField::nullable("added", DataType::STRING));
        request
            .properties_added
            .insert("app.key".to_string(), "value".to_string());

        let prepared = prepare(engine.as_ref(), &snapshot, request).unwrap();

        let schema = prepared.table_config.logical_schema();
        for field in read_config.logical_schema().fields() {
            assert_eq!(schema.field(field.name()), Some(field));
        }
        let added = schema.field("added").unwrap();
        assert_ne!(
            added.physical_name(ColumnMappingMode::Name),
            added.name().as_str()
        );
        assert!(
            prepared
                .table_config
                .table_properties()
                .column_mapping_max_column_id
                > read_max_id
        );
        assert_eq!(
            configuration_value(&prepared.table_config, "app.key"),
            Some("value")
        );
    }

    #[test]
    fn row_tracking_column_names_are_preserved() {
        let (engine, snapshot) = create_test_table(&[(ENABLE_ROW_TRACKING, "true")]);
        let read_config = snapshot.table_configuration();
        let prepared = prepare(
            engine.as_ref(),
            &snapshot,
            add_column_request(StructField::nullable("added", DataType::STRING)),
        )
        .unwrap();
        for key in [
            MATERIALIZED_ROW_ID_COLUMN_NAME,
            MATERIALIZED_ROW_COMMIT_VERSION_COLUMN_NAME,
        ] {
            assert!(configuration_value(read_config, key).is_some());
            assert_eq!(
                configuration_value(&prepared.table_config, key),
                configuration_value(read_config, key)
            );
        }
    }

    #[rstest]
    #[case::set_max_column_id(&[(COLUMN_MAPPING_MAX_COLUMN_ID, "100")], &[], "Setting delta property")]
    #[case::set_row_id_name(&[(MATERIALIZED_ROW_ID_COLUMN_NAME, "other")], &[], "Setting delta property")]
    #[case::set_ict_enablement_version(
        &[(IN_COMMIT_TIMESTAMP_ENABLEMENT_VERSION, "1")],
        &[],
        "Setting delta property"
    )]
    #[case::unset_row_id_name(&[], &[MATERIALIZED_ROW_ID_COLUMN_NAME], "Unsetting table property")]
    #[case::unset_row_commit_version_name(
        &[],
        &[MATERIALIZED_ROW_COMMIT_VERSION_COLUMN_NAME],
        "Unsetting table property"
    )]
    #[case::unset_ict_enablement_timestamp(
        &[],
        &[IN_COMMIT_TIMESTAMP_ENABLEMENT_TIMESTAMP],
        "Unsetting table property"
    )]
    #[case::unset_feature_signal(&[], &["delta.feature.rowTracking"], "Unsetting table property")]
    #[case::unset_in_commit_timestamps(
        &[],
        &[ENABLE_IN_COMMIT_TIMESTAMPS],
        "Unsetting table property"
    )]
    #[case::unset_delta_key_in_other_case(&[], &["DELTA.appendOnly"], "Unsetting table property")]
    #[case::set_and_unset(&[("app.key", "value")], &["app.key"], "both set and unset")]
    #[case::set_transaction_retention(
        &[("delta.setTransactionRetentionDuration", "interval 2 weeks")],
        &[],
        "Setting delta property 'delta.setTransactionRetentionDuration'"
    )]
    #[case::iceberg_native_v4(
        &[("delta.enableIcebergNativeV4", "true")],
        &[],
        "Setting delta property 'delta.enableIcebergNativeV4'"
    )]
    #[case::parquet_format_version(
        &[("DELTA.PARQUET.FORMAT.VERSION", "2.12.0")],
        &[],
        "Setting delta property 'delta.parquet.format.version'"
    )]
    #[case::set_and_unset_after_canonicalization(
        &[("DELTA.APPENDONLY", "true")],
        &["delta.appendOnly"],
        "both set and unset"
    )]
    #[case::uppercase_row_tracking(
        &[(ENABLE_ROW_TRACKING, "TRUE")],
        &[],
        "Invalid value 'TRUE' for table property 'delta.enableRowTracking'"
    )]
    #[case::mixed_case_iceberg_compat_v2(
        &[(ENABLE_ICEBERG_COMPAT_V2, "True")],
        &[],
        "Invalid value 'True' for table property 'delta.enableIcebergCompatV2'"
    )]
    fn kernel_managed_and_conflicting_property_requests_are_rejected(
        #[case] added: &[(&str, &str)],
        #[case] removed: &[&str],
        #[case] expected: &str,
    ) {
        let (engine, snapshot) = create_test_table(&[(ENABLE_ROW_TRACKING, "true")]);
        let mut request = properties_request(added);
        request.property_keys_removed = removed.iter().map(|key| key.to_string()).collect();
        let error = prepare(engine.as_ref(), &snapshot, request).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }

    #[rstest]
    #[case::enable_row_tracking(&[], &[(ENABLE_ROW_TRACKING, "true")], "row tracking")]
    #[case::disable_row_tracking(
        &[(ENABLE_ROW_TRACKING, "true")],
        &[(ENABLE_ROW_TRACKING, "false")],
        "row tracking"
    )]
    #[case::enable_iceberg_compat_v3(
        &[(COLUMN_MAPPING_MODE, "name"), (ENABLE_ROW_TRACKING, "true")],
        &[(ENABLE_ICEBERG_COMPAT_V3, "true")],
        "Enabling 'delta.enableIcebergCompatV3'"
    )]
    #[case::disable_iceberg_compat_v3(
        &[(ENABLE_ICEBERG_COMPAT_V3, "true")],
        &[(ENABLE_ICEBERG_COMPAT_V3, "false")],
        "Disabling 'delta.enableIcebergCompatV3'"
    )]
    #[case::enable_column_mapping_id(&[], &[(COLUMN_MAPPING_MODE, "id")], "column mapping mode")]
    #[case::disable_column_mapping(
        &[(COLUMN_MAPPING_MODE, "name")],
        &[(COLUMN_MAPPING_MODE, "none")],
        "column mapping mode"
    )]
    #[case::change_column_mapping(
        &[(COLUMN_MAPPING_MODE, "name")],
        &[(COLUMN_MAPPING_MODE, "id")],
        "column mapping mode"
    )]
    fn unsupported_transitions_are_rejected(
        #[case] table_properties: &[(&str, &str)],
        #[case] requested: &[(&str, &str)],
        #[case] expected: &str,
    ) {
        let (engine, snapshot) = create_test_table(table_properties);
        let error = prepare(engine.as_ref(), &snapshot, properties_request(requested)).unwrap_err();
        assert!(matches!(error, KernelError::Unsupported(_)), "{error}");
        assert!(error.to_string().contains(expected), "{error}");
    }

    #[test]
    fn enabling_column_mapping_name_mode_assigns_existing_field_names_as_physical_names() {
        let (engine, snapshot) = create_test_table(&[]);
        let read_schema = snapshot.table_configuration().logical_schema();
        let prepared = prepare(
            engine.as_ref(),
            &snapshot,
            properties_request(&[(COLUMN_MAPPING_MODE, "name")]),
        )
        .unwrap();

        assert!(prepared.should_emit_protocol);
        assert!(prepared.should_emit_metadata);
        let config = &prepared.table_config;
        assert!(config.is_feature_supported(&TableFeature::ColumnMapping));
        assert_eq!(config.column_mapping_mode(), ColumnMappingMode::Name);
        let schema = config.logical_schema();
        let mut ids: Vec<_> = schema
            .fields()
            .map(|field| {
                assert_eq!(field.physical_name(ColumnMappingMode::Name), field.name());
                field.column_mapping_id().unwrap()
            })
            .collect();
        ids.sort_unstable();
        let field_count = read_schema.fields().count() as i64;
        assert_eq!(ids, (1..=field_count).collect::<Vec<_>>());
        assert_eq!(
            config.table_properties().column_mapping_max_column_id,
            Some(field_count)
        );
    }

    #[test]
    fn enabling_column_mapping_name_mode_assigns_nested_fields_and_drops_leftover_annotations() {
        fn annotated(
            mut field: StructField,
            key: ColumnMetadataKey,
            value: MetadataValue,
        ) -> StructField {
            field.metadata.insert(key.as_ref().into(), value);
            field
        }
        fn collect_struct_fields<'a>(data_type: &'a DataType, out: &mut Vec<&'a StructField>) {
            match data_type {
                DataType::Struct(inner) => {
                    for field in inner.fields() {
                        out.push(field);
                        collect_struct_fields(field.data_type(), out);
                    }
                }
                DataType::Array(array) => collect_struct_fields(array.element_type(), out),
                DataType::Map(map) => {
                    collect_struct_fields(map.key_type(), out);
                    collect_struct_fields(map.value_type(), out);
                }
                _ => {}
            }
        }

        let (engine, snapshot) = create_test_table(&[]);
        // Leftover annotations on a mapping-disabled schema, which Kernel must not reuse.
        let leftover_schema = Arc::new(
            StructType::try_new([
                StructField::nullable("id", DataType::INTEGER),
                StructField::nullable(
                    "s",
                    StructType::try_new([
                        annotated(
                            StructField::nullable("a", DataType::INTEGER),
                            ColumnMetadataKey::ColumnMappingPhysicalName,
                            MetadataValue::String("stale-a".to_string()),
                        ),
                        StructField::nullable(
                            "b",
                            StructType::try_new([StructField::nullable("c", DataType::STRING)])
                                .unwrap(),
                        ),
                    ])
                    .unwrap(),
                ),
                annotated(
                    StructField::nullable(
                        "arr",
                        ArrayType::new(
                            StructType::try_new([StructField::nullable("x", DataType::INTEGER)])
                                .unwrap(),
                            true,
                        ),
                    ),
                    ColumnMetadataKey::ColumnMappingNestedIds,
                    MetadataValue::Other(serde_json::json!({ "arr.element": 99 })),
                ),
                annotated(
                    StructField::nullable(
                        "m",
                        MapType::new(
                            DataType::STRING,
                            StructType::try_new([StructField::nullable("y", DataType::INTEGER)])
                                .unwrap(),
                            true,
                        ),
                    ),
                    ColumnMetadataKey::ParquetFieldNestedIds,
                    MetadataValue::Other(serde_json::json!({ "m.key": 98 })),
                ),
            ])
            .unwrap(),
        );
        let read_config = snapshot.table_configuration();
        let metadata = read_config
            .metadata()
            .clone()
            .with_schema(leftover_schema.clone())
            .unwrap();
        let read_config =
            TableConfiguration::try_new_with_schema(read_config, metadata, leftover_schema)
                .unwrap();
        let snapshot = with_table_config(&snapshot, read_config);

        let prepared = prepare(
            engine.as_ref(),
            &snapshot,
            properties_request(&[(COLUMN_MAPPING_MODE, "name")]),
        )
        .unwrap();

        let schema = prepared.table_config.logical_schema();
        let schema_type = DataType::from(schema.as_ref().clone());
        let mut fields = Vec::new();
        collect_struct_fields(&schema_type, &mut fields);
        let names: Vec<_> = fields.iter().map(|field| field.name().as_str()).collect();
        assert_eq!(names, ["id", "s", "a", "b", "c", "arr", "x", "m", "y"]);
        let mut ids = Vec::new();
        for field in &fields {
            assert_eq!(field.physical_name(ColumnMappingMode::Name), field.name());
            ids.push(field.column_mapping_id().unwrap());
            for key in [
                ColumnMetadataKey::ColumnMappingNestedIds,
                ColumnMetadataKey::ParquetFieldNestedIds,
            ] {
                assert!(field.get_config_value(&key).is_none(), "{field:?}");
            }
        }
        ids.sort_unstable();
        assert_eq!(ids, (1..=9).collect::<Vec<_>>());
        assert_eq!(
            prepared
                .table_config
                .table_properties()
                .column_mapping_max_column_id,
            Some(9)
        );
    }

    #[test]
    fn enabling_column_mapping_with_schema_change_is_rejected() {
        let (engine, snapshot) = create_test_table(&[]);
        let mut request = add_column_request(StructField::nullable("added", DataType::STRING));
        request
            .properties_added
            .insert(COLUMN_MAPPING_MODE.to_string(), "name".to_string());
        let error = prepare(engine.as_ref(), &snapshot, request).unwrap_err();
        assert!(matches!(error, KernelError::Unsupported(_)), "{error}");
        assert!(
            error
                .to_string()
                .contains("Cannot update the column mapping mode and perform schema changes"),
            "{error}"
        );
    }

    #[test]
    fn metadata_only_change_violating_feature_requirement_is_rejected() {
        let (engine, snapshot) = create_test_table(&[
            (ENABLE_ICEBERG_COMPAT_V2, "true"),
            ("delta.feature.deletionVectors", "supported"),
        ]);
        assert!(
            snapshot
                .table_configuration()
                .is_feature_supported(&TableFeature::DeletionVectors)
        );
        let error = prepare(
            engine.as_ref(),
            &snapshot,
            properties_request(&[("delta.enableDeletionVectors", "true")]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("deletionVectors"), "{error}");
    }

    #[test]
    fn disabling_iceberg_compat_v2_on_iceberg_universal_format_table_is_rejected() {
        let (engine, snapshot) = create_test_table(&[
            (COLUMN_MAPPING_MODE, "name"),
            (ENABLE_ICEBERG_COMPAT_V2, "true"),
        ]);
        // CREATE TABLE does not accept the UniForm property, so set it on the read metadata.
        let snapshot =
            with_configuration_entry(&snapshot, UNIVERSAL_FORMAT_ENABLED_FORMATS, Some("iceberg"));
        let error = prepare(
            engine.as_ref(),
            &snapshot,
            properties_request(&[(ENABLE_ICEBERG_COMPAT_V2, "false")]),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("must be set to \"true\" to enable the iceberg universal format"),
            "{error}"
        );
    }

    #[rstest]
    #[case::feature_signal(&[("delta.feature.allowColumnDefaults", "supported")], true)]
    #[case::other_property(&[("delta.enableChangeDataFeed", "true")], false)]
    fn allow_column_defaults_is_never_derived(
        #[case] requested: &[(&str, &str)],
        #[case] rejected: bool,
    ) {
        let (engine, snapshot) = create_test_table(&[]);
        let mut request = properties_request(requested);
        request.schema_changes = Some(vec![SchemaOperation::add_column(
            None,
            StructField::nullable("added", DataType::STRING),
        )]);
        let result = prepare(engine.as_ref(), &snapshot, request);
        if rejected {
            let error = result.unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("Enabling feature 'allowColumnDefaults'"),
                "{error}"
            );
        } else {
            let config = result.unwrap().table_config;
            assert!(!config.is_feature_supported(&TableFeature::AllowColumnDefaults));
        }
    }

    // Writer version 5 implies checkConstraints, which Kernel writes only with this feature.
    #[cfg(feature = "check-constraints-in-dev")]
    #[test]
    fn legacy_column_mapping_protocol_upgrade_retains_implied_features() {
        let (engine, snapshot) = create_test_table(&[(COLUMN_MAPPING_MODE, "name")]);
        // A (2, 5) protocol implies columnMapping without listing table features.
        let read_config = snapshot.table_configuration();
        let legacy = TableConfiguration::try_new(
            read_config.metadata().clone(),
            Protocol::try_new(2, 5, None::<Vec<String>>, None::<Vec<String>>).unwrap(),
            read_config.table_root().clone(),
            read_config.version(),
        )
        .unwrap();
        assert!(legacy.is_feature_supported(&TableFeature::ColumnMapping));
        let snapshot = with_table_config(&snapshot, legacy);

        let prepared = prepare(
            engine.as_ref(),
            &snapshot,
            add_column_request(StructField::nullable("ts", DataType::TIMESTAMP_NTZ)),
        )
        .unwrap();

        assert!(prepared.should_emit_protocol);
        let config = &prepared.table_config;
        assert_eq!(config.column_mapping_mode(), ColumnMappingMode::Name);
        for feature in [
            TableFeature::ColumnMapping,
            TableFeature::AppendOnly,
            TableFeature::Invariants,
            TableFeature::CheckConstraints,
            TableFeature::ChangeDataFeed,
            TableFeature::GeneratedColumns,
            TableFeature::TimestampWithoutTimezone,
        ] {
            assert!(
                config.is_feature_supported(&feature),
                "{feature} was dropped"
            );
        }
        let protocol = config.protocol();
        assert!(
            protocol
                .reader_features()
                .is_some_and(|features| features.contains(&TableFeature::ColumnMapping))
        );
    }

    #[test]
    fn enabling_iceberg_compat_v2_without_column_mapping_is_rejected() {
        let (engine, snapshot) = create_test_table(&[]);
        let error = prepare(
            engine.as_ref(),
            &snapshot,
            properties_request(&[(ENABLE_ICEBERG_COMPAT_V2, "true")]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("columnMapping"), "{error}");
    }

    #[rstest]
    #[case::checkpoint_interval_max(CHECKPOINT_INTERVAL, "2147483647")]
    #[case::num_indexed_cols_all(DATA_SKIPPING_NUM_INDEXED_COLS, "-1")]
    #[case::num_indexed_cols_max(DATA_SKIPPING_NUM_INDEXED_COLS, "2147483647")]
    fn java_int_range_bounds_are_accepted(#[case] key: &str, #[case] value: &str) {
        let (engine, snapshot) = create_test_table(&[]);
        let prepared = prepare(
            engine.as_ref(),
            &snapshot,
            properties_request(&[(key, value)]),
        )
        .unwrap();
        assert_eq!(
            configuration_value(&prepared.table_config, key),
            Some(value)
        );
    }

    #[test]
    fn enabling_in_commit_timestamps_records_enablement() {
        let (engine, snapshot) = create_test_table(&[]);
        let commit_timestamp = current_time_ms().unwrap();
        let prepared = prepare_update_table_metadata(
            &snapshot,
            properties_request(&[(ENABLE_IN_COMMIT_TIMESTAMPS, "true")]),
            commit_timestamp,
            engine.as_ref(),
        )
        .unwrap();
        let config = &prepared.table_config;
        assert!(prepared.should_emit_protocol);
        assert!(config.is_feature_enabled(&TableFeature::InCommitTimestamp));
        let InCommitTimestampEnablement::Enabled {
            enablement: Some((version, timestamp)),
        } = config.in_commit_timestamp_enablement().unwrap()
        else {
            panic!("expected recorded enablement metadata");
        };
        assert_eq!(version, snapshot.version() + 1);
        assert!(timestamp >= commit_timestamp);
    }

    #[rstest]
    #[case::enabled_at_creation(&[(ENABLE_IN_COMMIT_TIMESTAMPS, "true")], None)]
    #[case::enabled_later(
        &[
            (ENABLE_IN_COMMIT_TIMESTAMPS, "true"),
            (IN_COMMIT_TIMESTAMP_ENABLEMENT_VERSION, "0"),
            (IN_COMMIT_TIMESTAMP_ENABLEMENT_TIMESTAMP, "1000"),
        ],
        Some((0, 1000))
    )]
    fn existing_in_commit_timestamp_enablement_is_preserved(
        #[case] table_properties: &[(&str, &str)],
        #[case] expected: Option<(u64, i64)>,
    ) {
        let (engine, snapshot) = create_test_table(&[(ENABLE_IN_COMMIT_TIMESTAMPS, "true")]);
        // CREATE TABLE rejects the enablement properties, so set them on the read metadata.
        let read_config = snapshot.table_configuration();
        let metadata = read_config.metadata().clone().with_configuration(
            table_properties
                .iter()
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .collect(),
        );
        let read_config = TableConfiguration::try_new_with_schema(
            read_config,
            metadata,
            read_config.logical_schema(),
        )
        .unwrap();
        let snapshot = with_table_config(&snapshot, read_config);

        let prepared = prepare(
            engine.as_ref(),
            &snapshot,
            add_column_request(StructField::nullable("added", DataType::STRING)),
        )
        .unwrap();
        assert_eq!(
            prepared
                .table_config
                .in_commit_timestamp_enablement()
                .unwrap(),
            InCommitTimestampEnablement::Enabled {
                enablement: expected
            }
        );
    }

    #[test]
    fn disabling_in_commit_timestamps_removes_enablement() {
        let (engine, snapshot) = create_test_table(&[]);
        let enabled = prepare(
            engine.as_ref(),
            &snapshot,
            properties_request(&[(ENABLE_IN_COMMIT_TIMESTAMPS, "true")]),
        )
        .unwrap();
        let snapshot = with_table_config(&snapshot, enabled.table_config);

        let prepared = prepare(
            engine.as_ref(),
            &snapshot,
            properties_request(&[(ENABLE_IN_COMMIT_TIMESTAMPS, "false")]),
        )
        .unwrap();
        let config = &prepared.table_config;
        assert!(!config.is_feature_enabled(&TableFeature::InCommitTimestamp));
        assert!(configuration_value(config, IN_COMMIT_TIMESTAMP_ENABLEMENT_VERSION).is_none());
        assert!(configuration_value(config, IN_COMMIT_TIMESTAMP_ENABLEMENT_TIMESTAMP).is_none());
    }

    #[test]
    fn missing_row_tracking_column_name_is_rejected() {
        let (engine, snapshot) = create_test_table(&[(ENABLE_ROW_TRACKING, "true")]);
        let snapshot = with_configuration_entry(&snapshot, MATERIALIZED_ROW_ID_COLUMN_NAME, None);
        let error = prepare(
            engine.as_ref(),
            &snapshot,
            add_column_request(StructField::nullable("added", DataType::STRING)),
        )
        .unwrap_err();
        assert!(
            error.to_string().contains(&format!(
                "materialized column name '{MATERIALIZED_ROW_ID_COLUMN_NAME}' is missing"
            )),
            "{error}"
        );
    }

    #[rstest]
    fn row_tracking_column_name_conflicting_with_column_is_rejected(
        #[values(true, false)] row_tracking_enabled: bool,
    ) {
        let (engine, snapshot) = create_test_table(&[(ENABLE_ROW_TRACKING, "true")]);
        let snapshot = if row_tracking_enabled {
            snapshot
        } else {
            // Names stay configured after row tracking is disabled.
            Arc::new(with_configuration_entry(
                &snapshot,
                ENABLE_ROW_TRACKING,
                Some("false"),
            ))
        };
        let row_id_name = configuration_value(
            snapshot.table_configuration(),
            MATERIALIZED_ROW_ID_COLUMN_NAME,
        )
        .unwrap()
        .to_string();
        let error = prepare(
            engine.as_ref(),
            &snapshot,
            add_column_request(StructField::nullable(row_id_name, DataType::STRING)),
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("conflicts with a table column"),
            "{error}"
        );
    }
}
