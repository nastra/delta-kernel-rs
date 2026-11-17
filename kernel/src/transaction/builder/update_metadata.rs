//! Protocol and metadata preparation for updates to an existing table.
//!
//! [`prepare_update_table_metadata`] takes the requested metadata of an update (the read
//! snapshot's metadata with the schema and property changes applied), derives the protocol
//! features that the final schema requires, validates the transition from the read snapshot, and
//! validates the final table configuration. It mirrors the update path of Delta Kernel Java's
//! `TransactionMetadataFactory`.
//!
//! [`prepare_clustering_columns`] then resolves requested clustering columns against that final
//! configuration and adds the `clustering` feature when the protocol lacks it.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use super::create_table::{
    ValidatedTableProperties, maybe_enable_collations, maybe_enable_file_type_preview,
    maybe_enable_timestamp_ntz, maybe_enable_variant_type,
};
use crate::actions::{Metadata, Protocol};
use crate::clustering::resolve_physical_clustering_columns;
use crate::expressions::ColumnName;
use crate::schema::{ColumnMetadataKey, MetadataValue, SchemaRef, StructField, StructType};
use crate::table_configuration::TableConfiguration;
use crate::table_features::{
    ColumnMappingMode, IcebergCompatValidationContext, LEGACY_WRITER_FEATURES,
    TABLE_FEATURES_MIN_READER_VERSION, TABLE_FEATURES_MIN_WRITER_VERSION, TableFeature,
    V2_VALIDATOR, V3_VALIDATOR, add_feature_to_lists, assign_column_mapping_metadata,
    assign_nested_cm_ids, drop_column_mapping_metadata, ensure_domain_metadata_dependency,
    extract_enabled_reader_features, find_max_column_id_in_schema,
    get_column_mapping_mode_from_properties, validate_iceberg_compat_if_needed,
};
use crate::table_properties::{
    COLUMN_MAPPING_MAX_COLUMN_ID, ENABLE_ICEBERG_COMPAT_V2, ENABLE_ICEBERG_COMPAT_V3,
    ENABLE_ROW_TRACKING, MATERIALIZED_ROW_COMMIT_VERSION_COLUMN_NAME,
    MATERIALIZED_ROW_ID_COLUMN_NAME, TableProperties,
};
use crate::transforms::{SchemaTransform, transform_output_type};
use crate::{KernelError, KernelResult, Result};

/// Universal formats (UniForm) the table is converted to, as a comma-separated list.
const UNIVERSAL_FORMAT_ENABLED_FORMATS: &str = "delta.universalFormat.enabledFormats";

/// Table configuration that an update to an existing table commits.
#[derive(Debug)]
pub(super) struct PreparedUpdateTableMetadata {
    pub(super) table_config: TableConfiguration,
    /// `true` if the protocol differs from the read snapshot's protocol.
    pub(super) should_emit_protocol: bool,
}

/// Derives and validates the table configuration for `requested`, the metadata an update commits.
///
/// `requested` is the read snapshot's metadata with the schema and property changes of the update
/// applied. The result retains every protocol feature of `read_config` and adds the features the
/// final schema requires (`timestampNtz`, `variantType`, `collations`, `fileType-preview`), using
/// the same rules as CREATE TABLE. The protocol changes only when a feature is added.
///
/// Transitions from `read_config` follow Java Kernel:
/// - The column mapping mode may only change from `none` to `name`, and not together with a schema
///   change. Enabling `name` assigns ids to the existing fields and uses their current names as
///   physical names, since the existing data files were written with those names.
/// - Row tracking and `icebergCompatV3` cannot be enabled or disabled.
/// - When column mapping and IcebergCompat V2 or V3 are enabled, array and map slots without
///   `delta.columnMapping.nested.ids` are assigned ids.
///
/// The final configuration must be writable by Kernel and pass the IcebergCompat write checks, the
/// row-tracking column-name checks, and the UniForm configuration checks.
pub(super) fn prepare_update_table_metadata(
    read_config: &TableConfiguration,
    requested: Metadata,
    has_schema_changes: bool,
) -> Result<PreparedUpdateTableMetadata> {
    let mut configuration = requested.configuration().clone();
    validate_transitions(read_config, &configuration, has_schema_changes)?;

    let original_schema: SchemaRef = Arc::new(requested.parse_schema()?);
    let mut schema = original_schema.clone();
    let new_mode = get_column_mapping_mode_from_properties(&configuration)?;
    let enables_column_mapping =
        get_column_mapping_mode_from_properties(read_config.metadata().configuration())?
            != new_mode;
    if enables_column_mapping {
        schema = assign_column_mapping_to_existing_fields(&schema, &mut configuration)?;
    }
    if new_mode != ColumnMappingMode::None && is_iceberg_compat_enabled(&configuration) {
        schema = assign_iceberg_compat_nested_ids(&schema, &mut configuration)?;
    }
    let requested = if Arc::ptr_eq(&schema, &original_schema) {
        requested
    } else {
        requested.with_schema(schema.clone())?
    };

    let mut derived = ValidatedTableProperties {
        properties: HashMap::new(),
        reader_features: Vec::new(),
        writer_features: Vec::new(),
    };
    if enables_column_mapping {
        add_feature_to_lists(
            TableFeature::ColumnMapping,
            &mut derived.reader_features,
            &mut derived.writer_features,
        );
    }
    maybe_enable_variant_type(&schema, &mut derived);
    maybe_enable_timestamp_ntz(&schema, &mut derived);
    maybe_enable_collations(&schema, &mut derived);
    maybe_enable_file_type_preview(&schema, &mut derived);
    let ValidatedTableProperties {
        mut reader_features,
        mut writer_features,
        ..
    } = derived;
    ensure_domain_metadata_dependency(&mut reader_features, &mut writer_features);
    reader_features.retain(|feature| !is_already_supported(read_config, feature));
    writer_features.retain(|feature| !is_already_supported(read_config, feature));
    let should_emit_protocol = !writer_features.is_empty();
    let metadata = requested.with_configuration(configuration);
    let table_config = if should_emit_protocol {
        TableConfiguration::try_new(
            metadata,
            upgrade_protocol(read_config, reader_features, writer_features)?,
            read_config.table_root().clone(),
            read_config.version(),
        )?
    } else {
        TableConfiguration::try_new_with_schema(read_config, metadata, schema)?
    };
    validate_final_configuration(read_config, &table_config)?;
    Ok(PreparedUpdateTableMetadata {
        table_config,
        should_emit_protocol,
    })
}

/// Table configuration and physical clustering columns that a clustering change commits.
#[derive(Debug)]
pub(super) struct PreparedClusteringColumns {
    pub(super) table_config: TableConfiguration,
    /// `true` if the protocol of `table_config` gained the `clustering` feature.
    pub(super) added_clustering_feature: bool,
    /// Physical paths to record in the `delta.clustering` domain, in request order.
    pub(super) physical_columns: Vec<ColumnName>,
}

/// Resolves `logical_columns` against `table_config`, the final configuration of the update, and
/// adds the `clustering` feature (and its `domainMetadata` dependency) if the protocol lacks it.
///
/// Columns resolve against the final schema and column mapping mode, so they may name fields added
/// by schema changes in the same update and get the physical names that enabling column mapping
/// assigns. An empty list clears the clustering columns and skips resolution. The feature is never
/// removed. Mirrors Java's `TransactionMetadataFactory.performProtocolUpgrades` (with
/// `clusteringRequired`) and `updateColumnMappingMetadataAndResolveClusteringColumns`.
pub(super) fn prepare_clustering_columns(
    table_config: TableConfiguration,
    logical_columns: &[ColumnName],
) -> Result<PreparedClusteringColumns> {
    let physical_columns = if logical_columns.is_empty() {
        Vec::new()
    } else {
        resolve_physical_clustering_columns(
            table_config.logical_schema_ref(),
            logical_columns,
            table_config.column_mapping_mode(),
        )?
    };
    if table_config.is_feature_supported(&TableFeature::ClusteredTable) {
        return Ok(PreparedClusteringColumns {
            table_config,
            added_clustering_feature: false,
            physical_columns,
        });
    }
    let mut reader_features = Vec::new();
    let mut writer_features = Vec::new();
    add_feature_to_lists(
        TableFeature::ClusteredTable,
        &mut reader_features,
        &mut writer_features,
    );
    ensure_domain_metadata_dependency(&mut reader_features, &mut writer_features);
    reader_features.retain(|feature| !table_config.is_feature_supported(feature));
    writer_features.retain(|feature| !table_config.is_feature_supported(feature));
    let protocol = upgrade_protocol(&table_config, reader_features, writer_features)?;
    let upgraded = TableConfiguration::try_new(
        table_config.metadata().clone(),
        protocol,
        table_config.table_root().clone(),
        table_config.version(),
    )?;
    upgraded.ensure_write_supported()?;
    Ok(PreparedClusteringColumns {
        table_config: upgraded,
        added_clustering_feature: true,
        physical_columns,
    })
}

/// Reads a boolean table property the way Java's `TableConfig` does (`Boolean.parseBoolean`), so
/// that a value such as `TRUE` cannot slip past the transition checks.
fn is_property_true(configuration: &HashMap<String, String>, key: &str) -> bool {
    configuration
        .get(key)
        .is_some_and(|value| value.eq_ignore_ascii_case("true"))
}

fn is_iceberg_compat_enabled(configuration: &HashMap<String, String>) -> bool {
    [ENABLE_ICEBERG_COMPAT_V2, ENABLE_ICEBERG_COMPAT_V3]
        .iter()
        .any(|key| is_property_true(configuration, key))
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

/// Rejects configuration transitions that Kernel does not support on existing tables.
fn validate_transitions(
    read_config: &TableConfiguration,
    new_configuration: &HashMap<String, String>,
    has_schema_changes: bool,
) -> KernelResult<()> {
    let old_configuration = read_config.metadata().configuration();
    // Java: `ColumnMapping.verifyColumnMappingChange` and, for schema evolution,
    // `TransactionMetadataFactory.validateMetadataChangeAndApplyTypeWidening`.
    let old_mode = get_column_mapping_mode_from_properties(old_configuration)?;
    let new_mode = get_column_mapping_mode_from_properties(new_configuration)?;
    if old_mode != new_mode {
        if (old_mode, new_mode) != (ColumnMappingMode::None, ColumnMappingMode::Name) {
            return Err(KernelError::unsupported(format!(
                "Changing column mapping mode from '{old_mode}' to '{new_mode}' is not supported"
            )));
        }
        if has_schema_changes {
            return Err(KernelError::unsupported(
                "Cannot update mapping mode and perform schema evolution",
            ));
        }
    }
    // Java: `RowTracking.throwIfRowTrackingToggled`. Enabling requires backfilling row ids and row
    // commit versions, and disabling cannot be undone.
    if is_property_true(old_configuration, ENABLE_ROW_TRACKING)
        != is_property_true(new_configuration, ENABLE_ROW_TRACKING)
    {
        return Err(KernelError::unsupported(
            "Row tracking support cannot be changed once the table is created.",
        ));
    }
    // Java: `IcebergCompatV3MetadataValidatorAndUpdater.validateIcebergCompatV3Change`.
    match (
        is_property_true(old_configuration, ENABLE_ICEBERG_COMPAT_V3),
        is_property_true(new_configuration, ENABLE_ICEBERG_COMPAT_V3),
    ) {
        (false, true) => Err(KernelError::unsupported(format!(
            "Cannot enable {ENABLE_ICEBERG_COMPAT_V3} on an existing table. Enablement is only \
             supported upon table creation."
        ))),
        (true, false) => Err(KernelError::unsupported(format!(
            "Disabling {ENABLE_ICEBERG_COMPAT_V3} on an existing table is not allowed."
        ))),
        _ => Ok(()),
    }
}

/// Validates the configuration an update commits. Java runs the equivalent checks on the effective
/// metadata and protocol of every update.
fn validate_final_configuration(
    read_config: &TableConfiguration,
    table_config: &TableConfiguration,
) -> Result<()> {
    // Java: `TableFeatures.validateKernelCanWriteToTable`.
    table_config.ensure_write_supported()?;
    // Java: `performIcebergCompatUpgradesAndValidation`.
    for validator in [&V2_VALIDATOR, &V3_VALIDATOR] {
        validate_iceberg_compat_if_needed(
            table_config,
            validator,
            IcebergCompatValidationContext::Write,
        )?;
    }
    validate_iceberg_compat_v2_without_deletion_vectors(read_config, table_config)?;
    validate_universal_format(table_config.metadata().configuration())?;
    validate_row_tracking_column_names(table_config)?;
    Ok(())
}

/// Rejects `icebergCompatV2` when the final or the read protocol supports `deletionVectors`, as
/// Java's `IcebergCompatMetadataValidatorAndUpdater.CHECK_HAS_NO_DELETION_VECTORS` does. Files
/// written while deletion vectors were enabled may still carry them, so support is enough to
/// reject.
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
            "Table features [{}] are incompatible with {}.",
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
        .filter(|key| is_property_true(configuration, key))
        .count();
    if iceberg_enabled && enabled_compat_count == 0 {
        return Err(KernelError::generic(format!(
            "Invalid value for table property '{UNIVERSAL_FORMAT_ENABLED_FORMATS}': One of {} \
             must be set to \"true\" to enable iceberg uniform format.",
            compat_keys.join(" or ")
        )));
    }
    if enabled_compat_count > 1 {
        return Err(KernelError::generic(format!(
            "Invalid value for table property '{UNIVERSAL_FORMAT_ENABLED_FORMATS}': '{}' cannot \
             be enabled at the same time.",
            compat_keys.join("' and '")
        )));
    }
    Ok(())
}

/// Requires materialized row-tracking column names when row tracking is enabled, and rejects
/// configured names that equal a top-level logical or physical column name, whether or not row
/// tracking is enabled. Mirrors Java's
/// `MaterializedRowTrackingColumn.validateRowTrackingConfigsNotMissing` and
/// `MaterializedRowTrackingColumn.throwIfColumnNamesConflictWithSchema`.
fn validate_row_tracking_column_names(table_config: &TableConfiguration) -> KernelResult<()> {
    let configuration = table_config.metadata().configuration();
    let row_tracking_enabled = is_property_true(configuration, ENABLE_ROW_TRACKING);
    let schema = table_config.logical_schema();
    let column_mapping_mode = table_config.column_mapping_mode();
    for key in [
        MATERIALIZED_ROW_ID_COLUMN_NAME,
        MATERIALIZED_ROW_COMMIT_VERSION_COLUMN_NAME,
    ] {
        let Some(name) = configuration.get(key) else {
            if row_tracking_enabled {
                return Err(KernelError::generic(format!(
                    "Row tracking is enabled but the materialized column name `{key}` is missing."
                )));
            }
            continue;
        };
        if schema
            .fields()
            .any(|field| field.name() == name || field.physical_name(column_mapping_mode) == name)
        {
            return Err(KernelError::generic(format!(
                "Cannot use column name '{name}' because it is reserved for internal use"
            )));
        }
    }
    Ok(())
}

/// Assigns column-mapping ids and physical names to every field of `schema` and records the new
/// `delta.columnMapping.maxColumnId` in `configuration`. Each physical name is the field's current
/// name, as in Java's `ColumnMapping.assignColumnIdAndPhysicalNameToField` with
/// `isNewTable = false`.
///
/// Column-mapping annotations left over from earlier history are dropped first: with mapping
/// disabled they do not describe the data files, which were written with the field names.
fn assign_column_mapping_to_existing_fields(
    schema: &StructType,
    configuration: &mut HashMap<String, String>,
) -> KernelResult<SchemaRef> {
    /// Sets each field's physical name to its name. Variant and file sub-fields are left as they
    /// are, as CREATE TABLE does.
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
/// that have none, as Java's `ColumnMapping.rewriteFieldIdsForIceberg` does, and records the new
/// `delta.columnMapping.maxColumnId` in `configuration` when ids are assigned. Returns `schema`
/// itself when every slot already has an id.
fn assign_iceberg_compat_nested_ids(
    schema: &SchemaRef,
    configuration: &mut HashMap<String, String>,
) -> KernelResult<SchemaRef> {
    let mut max_id = TableProperties::from(configuration.iter())
        .column_mapping_max_column_id
        .unwrap_or(0)
        .max(find_max_column_id_in_schema(schema).unwrap_or(0));
    let with_nested_ids = assign_nested_cm_ids(schema, &mut max_id)?;
    if &with_nested_ids == schema.as_ref() {
        return Ok(schema.clone());
    }
    configuration.insert(COLUMN_MAPPING_MAX_COLUMN_ID.to_string(), max_id.to_string());
    Ok(Arc::new(with_nested_ids))
}

/// Adds `new_reader_features` and `new_writer_features` to the protocol of `table_config`,
/// retaining every feature the protocol already supports, including features implied by legacy
/// versions.
///
/// The result always uses writer version 7. The reader version stays legacy (1 or 2) unless a
/// reader-writer feature is added, which moves it to version 3 with explicit reader features.
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
    use strum::IntoEnumIterator as _;

    use super::*;
    use crate::committer::FileSystemCommitter;
    use crate::engine::sync::SyncEngine;
    use crate::object_store::memory::InMemory;
    use crate::schema::{ArrayType, DataType, MapType, StructField, schema_ref};
    use crate::table_properties::{
        COLUMN_MAPPING_MODE, ENABLE_DELETION_VECTORS, ENABLE_IN_COMMIT_TIMESTAMPS,
        IN_COMMIT_TIMESTAMP_ENABLEMENT_TIMESTAMP, IN_COMMIT_TIMESTAMP_ENABLEMENT_VERSION,
    };
    use crate::transaction::create_table::create_table;
    use crate::transaction::schema_evolution::{SchemaOperation, evolve_metadata};
    use crate::unit_test_utils::{load_test_table, test_schema_flat};

    fn create_test_table(schema: SchemaRef, properties: &[(&str, &str)]) -> TableConfiguration {
        let engine = SyncEngine::new_with_store(Arc::new(InMemory::new()));
        create_table("memory:///", schema, "test", true)
            .with_table_properties(properties.iter().copied())
            .build(&engine, Box::new(FileSystemCommitter::new()))
            .unwrap()
            .commit(&engine)
            .unwrap()
            .unwrap_post_commit_snapshot()
            .table_configuration()
            .clone()
    }

    fn add_column(
        read_config: &TableConfiguration,
        field: StructField,
    ) -> Result<PreparedUpdateTableMetadata> {
        let (metadata, _) =
            evolve_metadata(read_config, vec![SchemaOperation::add_column(None, field)])?;
        prepare_update_table_metadata(read_config, metadata, true)
    }

    /// Returns the read metadata with `changes` applied: `Some` sets a key, `None` removes it.
    /// Stands in for property updates that the public setters do not accept yet.
    fn with_configuration(
        read_config: &TableConfiguration,
        changes: &[(&str, Option<&str>)],
    ) -> Metadata {
        let mut configuration = read_config.metadata().configuration().clone();
        for (key, value) in changes {
            match value {
                Some(value) => configuration.insert(key.to_string(), value.to_string()),
                None => configuration.remove(*key),
            };
        }
        read_config
            .metadata()
            .clone()
            .with_configuration(configuration)
    }

    fn assert_error_contains<T: std::fmt::Debug>(result: Result<T>, expected: &str) {
        let error = result.unwrap_err().to_string();
        assert!(
            error.contains(expected),
            "error={error}, expected={expected}"
        );
    }

    fn nested_ids(config: &TableConfiguration, field: &str) -> Option<MetadataValue> {
        config
            .logical_schema()
            .field(field)
            .unwrap()
            .get_config_value(&ColumnMetadataKey::ColumnMappingNestedIds)
            .cloned()
    }

    #[rstest]
    fn schema_change_derives_type_feature_and_retains_existing_features(
        #[values(
            (DataType::TIMESTAMP_NTZ, TableFeature::TimestampWithoutTimezone),
            (DataType::unshredded_variant(), TableFeature::VariantType),
            (DataType::file_type(), TableFeature::FileTypePreview)
        )]
        added: (DataType, TableFeature),
        #[values(
            "table-without-dv-small",
            "cdf-column-mapping-name-mode",
            "kernel_created"
        )]
        table: &str,
    ) {
        let (data_type, feature) = added;
        let read_config = if table == "kernel_created" {
            create_test_table(
                test_schema_flat(),
                &[
                    (ENABLE_DELETION_VECTORS, "true"),
                    ("delta.appendOnly", "true"),
                ],
            )
        } else {
            let (_engine, snapshot, _tempdir) = load_test_table(table).unwrap();
            snapshot.table_configuration().clone()
        };
        assert!(!read_config.is_feature_supported(&feature));
        let retained: Vec<_> = TableFeature::iter()
            .filter(|feature| read_config.is_feature_supported(feature))
            .collect();
        assert!(!retained.is_empty());

        let prepared = add_column(&read_config, StructField::nullable("added", data_type)).unwrap();

        assert!(prepared.should_emit_protocol);
        let config = &prepared.table_config;
        assert!(config.is_feature_supported(&feature));
        for retained_feature in &retained {
            assert!(
                config.is_feature_supported(retained_feature),
                "{retained_feature} was dropped"
            );
        }
        assert_eq!(
            config.column_mapping_mode(),
            read_config.column_mapping_mode()
        );
    }

    #[test]
    fn schema_change_without_new_feature_keeps_protocol() {
        let read_config = create_test_table(schema_ref! { nullable "ts": TIMESTAMP_NTZ }, &[]);
        let prepared = add_column(
            &read_config,
            StructField::nullable("ts2", DataType::TIMESTAMP_NTZ),
        )
        .unwrap();
        assert!(!prepared.should_emit_protocol);
        assert_eq!(prepared.table_config.protocol(), read_config.protocol());
    }

    #[test]
    fn schema_change_preserves_in_commit_timestamp_enablement() {
        let read_config =
            create_test_table(test_schema_flat(), &[(ENABLE_IN_COMMIT_TIMESTAMPS, "true")]);
        let prepared = add_column(
            &read_config,
            StructField::nullable("added", DataType::STRING),
        )
        .unwrap();
        for key in [
            ENABLE_IN_COMMIT_TIMESTAMPS,
            IN_COMMIT_TIMESTAMP_ENABLEMENT_VERSION,
            IN_COMMIT_TIMESTAMP_ENABLEMENT_TIMESTAMP,
        ] {
            assert_eq!(
                prepared.table_config.metadata().configuration().get(key),
                read_config.metadata().configuration().get(key),
                "{key}"
            );
        }
    }

    #[test]
    fn enabling_column_mapping_name_mode_uses_field_names_as_physical_names() {
        let schema = Arc::new(StructType::new_unchecked([
            StructField::nullable("id", DataType::LONG),
            StructField::nullable(
                "s",
                StructType::new_unchecked([StructField::nullable("inner", DataType::STRING)]),
            ),
            StructField::nullable(
                "arr",
                ArrayType::new(
                    StructType::new_unchecked([StructField::nullable("e", DataType::INTEGER)]),
                    true,
                ),
            ),
            StructField::nullable("m", MapType::new(DataType::STRING, DataType::LONG, true)),
            StructField::nullable("v", DataType::unshredded_variant()),
        ]));
        let read_config = create_test_table(schema, &[]);
        let requested = with_configuration(&read_config, &[(COLUMN_MAPPING_MODE, Some("name"))]);

        let prepared = prepare_update_table_metadata(&read_config, requested, false).unwrap();

        let config = &prepared.table_config;
        assert!(prepared.should_emit_protocol);
        assert!(config.is_feature_supported(&TableFeature::ColumnMapping));
        assert_eq!(config.column_mapping_mode(), ColumnMappingMode::Name);
        struct Collect(Vec<(String, MetadataValue, i64)>);
        impl<'a> SchemaTransform<'a> for Collect {
            transform_output_type!(|'a, T| Cow<'a, T>);
            fn transform_struct_field(&mut self, field: &'a StructField) -> Cow<'a, StructField> {
                let physical_name = field
                    .get_config_value(&ColumnMetadataKey::ColumnMappingPhysicalName)
                    .cloned();
                let id = field.get_config_value(&ColumnMetadataKey::ColumnMappingId);
                let Some(MetadataValue::Number(id)) = id else {
                    panic!("missing id on {}: {id:?}", field.name());
                };
                self.0
                    .push((field.name().clone(), physical_name.unwrap(), *id));
                self.recurse_into_struct_field(field)
            }
            fn transform_variant(&mut self, stype: &'a StructType) -> Cow<'a, StructType> {
                Cow::Borrowed(stype)
            }
        }
        let mut collect = Collect(Vec::new());
        let _ = collect.transform_struct(&config.logical_schema());
        let mut ids = Vec::new();
        for (name, physical_name, id) in collect.0 {
            assert_eq!(physical_name, MetadataValue::String(name));
            ids.push(id);
        }
        ids.sort_unstable();
        assert_eq!(ids, (1..=ids.len() as i64).collect::<Vec<_>>());
        assert_eq!(
            config.table_properties().column_mapping_max_column_id,
            Some(ids.len() as i64)
        );
    }

    #[rstest]
    #[case::name_to_none(&[(COLUMN_MAPPING_MODE, "name")], &[(COLUMN_MAPPING_MODE, Some("none"))], false, "Changing column mapping mode from 'name' to 'none' is not supported")]
    #[case::none_to_id(&[], &[(COLUMN_MAPPING_MODE, Some("id"))], false, "Changing column mapping mode from 'none' to 'id' is not supported")]
    #[case::name_to_id(&[(COLUMN_MAPPING_MODE, "name")], &[(COLUMN_MAPPING_MODE, Some("id"))], false, "Changing column mapping mode from 'name' to 'id' is not supported")]
    #[case::mode_change_with_schema_change(&[], &[(COLUMN_MAPPING_MODE, Some("name"))], true, "Cannot update mapping mode and perform schema evolution")]
    #[case::enable_row_tracking(&[], &[(ENABLE_ROW_TRACKING, Some("true"))], false, "Row tracking support cannot be changed")]
    #[case::enable_row_tracking_other_case(&[], &[(ENABLE_ROW_TRACKING, Some("TRUE"))], false, "Row tracking support cannot be changed")]
    #[case::disable_row_tracking(&[(ENABLE_ROW_TRACKING, "true")], &[(ENABLE_ROW_TRACKING, Some("false"))], false, "Row tracking support cannot be changed")]
    #[case::remove_row_tracking(&[(ENABLE_ROW_TRACKING, "true")], &[(ENABLE_ROW_TRACKING, None)], false, "Row tracking support cannot be changed")]
    #[case::enable_iceberg_compat_v3(&[], &[(ENABLE_ICEBERG_COMPAT_V3, Some("true"))], false, "Cannot enable delta.enableIcebergCompatV3 on an existing table")]
    #[case::disable_iceberg_compat_v3(&[(ENABLE_ICEBERG_COMPAT_V3, "true")], &[(ENABLE_ICEBERG_COMPAT_V3, Some("false"))], false, "Disabling delta.enableIcebergCompatV3 on an existing table is not allowed")]
    #[case::row_tracking_name_missing(&[(ENABLE_ROW_TRACKING, "true")], &[(MATERIALIZED_ROW_ID_COLUMN_NAME, None)], false, "Row tracking is enabled but the materialized column name")]
    #[case::uniform_without_iceberg_compat(&[], &[(UNIVERSAL_FORMAT_ENABLED_FORMATS, Some("iceberg"))], false, "must be set to \"true\" to enable iceberg uniform format")]
    fn unsupported_update_is_rejected(
        #[case] create_properties: &[(&str, &str)],
        #[case] changes: &[(&str, Option<&str>)],
        #[case] has_schema_changes: bool,
        #[case] expected: &str,
    ) {
        let read_config = create_test_table(test_schema_flat(), create_properties);
        let requested = with_configuration(&read_config, changes);
        assert_error_contains(
            prepare_update_table_metadata(&read_config, requested, has_schema_changes),
            expected,
        );
    }

    #[test]
    fn added_column_conflicting_with_row_tracking_column_name_is_rejected() {
        let read_config = create_test_table(test_schema_flat(), &[(ENABLE_ROW_TRACKING, "true")]);
        let name = read_config
            .metadata()
            .configuration()
            .get(MATERIALIZED_ROW_ID_COLUMN_NAME)
            .unwrap()
            .clone();
        assert_error_contains(
            add_column(&read_config, StructField::nullable(name, DataType::LONG)),
            "is reserved for internal use",
        );
        // The materialized names are retained for unrelated changes.
        let prepared =
            add_column(&read_config, StructField::nullable("added", DataType::LONG)).unwrap();
        for key in [
            MATERIALIZED_ROW_ID_COLUMN_NAME,
            MATERIALIZED_ROW_COMMIT_VERSION_COLUMN_NAME,
        ] {
            assert_eq!(
                prepared.table_config.metadata().configuration().get(key),
                read_config.metadata().configuration().get(key)
            );
        }
    }

    fn iceberg_compat_v2_table() -> TableConfiguration {
        let schema = Arc::new(StructType::new_unchecked([
            StructField::nullable("id", DataType::LONG),
            StructField::nullable("arr", ArrayType::new(DataType::INTEGER, true)),
            StructField::nullable(
                "s",
                StructType::new_unchecked([StructField::nullable(
                    "m",
                    MapType::new(
                        DataType::STRING,
                        ArrayType::new(DataType::INTEGER, true),
                        true,
                    ),
                )]),
            ),
        ]));
        create_test_table(
            schema,
            &[
                (COLUMN_MAPPING_MODE, "name"),
                (ENABLE_ICEBERG_COMPAT_V2, "true"),
            ],
        )
    }

    #[test]
    fn iceberg_compat_update_assigns_missing_nested_ids_and_keeps_existing_ones() {
        let created = iceberg_compat_v2_table();
        let expected_arr = nested_ids(&created, "arr");
        assert!(expected_arr.is_some());
        // Drop the nested ids of `arr` to model a table written without them.
        struct DropNestedIds;
        impl<'a> SchemaTransform<'a> for DropNestedIds {
            transform_output_type!(|'a, T| Cow<'a, T>);
            fn transform_struct_field(&mut self, field: &'a StructField) -> Cow<'a, StructField> {
                let mut field = field.clone();
                if field.name() == "arr" {
                    field
                        .metadata
                        .remove(ColumnMetadataKey::ColumnMappingNestedIds.as_ref());
                }
                Cow::Owned(field)
            }
        }
        let stripped = DropNestedIds
            .transform_struct(&created.logical_schema())
            .into_owned();
        let read_config = TableConfiguration::try_new(
            created
                .metadata()
                .clone()
                .with_schema(Arc::new(stripped))
                .unwrap(),
            created.protocol().clone(),
            created.table_root().clone(),
            created.version(),
        )
        .unwrap();
        let read_max_id = read_config
            .table_properties()
            .column_mapping_max_column_id
            .unwrap();

        let requested = with_configuration(&read_config, &[("app.key", Some("value"))]);
        let prepared = prepare_update_table_metadata(&read_config, requested, false).unwrap();

        let config = &prepared.table_config;
        assert!(!prepared.should_emit_protocol);
        let Some(MetadataValue::Other(serde_json::Value::Object(arr_ids))) =
            nested_ids(config, "arr")
        else {
            panic!("expected nested ids for arr");
        };
        assert_eq!(arr_ids.len(), 1);
        assert_eq!(
            arr_ids.values().next().unwrap().as_i64(),
            Some(read_max_id + 1)
        );
        assert_eq!(nested_ids(config, "s"), nested_ids(&read_config, "s"));
        assert_eq!(
            config.table_properties().column_mapping_max_column_id,
            Some(read_max_id + 1)
        );
    }

    #[test]
    fn iceberg_compat_update_with_complete_nested_ids_keeps_schema() {
        let read_config = iceberg_compat_v2_table();
        let requested = with_configuration(&read_config, &[("app.key", Some("value"))]);
        let prepared = prepare_update_table_metadata(&read_config, requested, false).unwrap();
        assert_eq!(
            prepared.table_config.logical_schema(),
            read_config.logical_schema()
        );
        assert_eq!(
            prepared
                .table_config
                .table_properties()
                .column_mapping_max_column_id,
            read_config.table_properties().column_mapping_max_column_id
        );
    }

    #[test]
    fn iceberg_compat_v2_with_supported_deletion_vectors_is_rejected() {
        let created = iceberg_compat_v2_table();
        let protocol = created.protocol();
        let mut reader_features = protocol.reader_features().unwrap().to_vec();
        reader_features.push(TableFeature::DeletionVectors);
        let mut writer_features = protocol.writer_features().unwrap().to_vec();
        writer_features.push(TableFeature::DeletionVectors);
        let read_config = TableConfiguration::try_new(
            created.metadata().clone(),
            Protocol::try_new_modern(reader_features, writer_features).unwrap(),
            created.table_root().clone(),
            created.version(),
        )
        .unwrap();
        let requested = with_configuration(&read_config, &[("app.key", Some("value"))]);
        assert_error_contains(
            prepare_update_table_metadata(&read_config, requested, false),
            "Table features [deletionVectors] are incompatible with icebergCompatV2",
        );
    }

    #[rstest]
    #[case::v2_enabled(&[(UNIVERSAL_FORMAT_ENABLED_FORMATS, Some("iceberg"))], None)]
    #[case::v2_disabled(
        &[(UNIVERSAL_FORMAT_ENABLED_FORMATS, Some("iceberg")), (ENABLE_ICEBERG_COMPAT_V2, Some("false"))],
        Some("must be set to \"true\" to enable iceberg uniform format")
    )]
    fn universal_format_requires_exactly_one_iceberg_compat_version(
        #[case] changes: &[(&str, Option<&str>)],
        #[case] expected_error: Option<&str>,
    ) {
        let read_config = iceberg_compat_v2_table();
        let requested = with_configuration(&read_config, changes);
        let result = prepare_update_table_metadata(&read_config, requested, false);
        match expected_error {
            Some(expected) => assert_error_contains(result, expected),
            None => {
                result.unwrap();
            }
        }
    }
}
