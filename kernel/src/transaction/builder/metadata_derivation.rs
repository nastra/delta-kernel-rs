//! Protocol and metadata derivation rules shared by table-creating and table-updating builders.
//!
//! Builders collect the requested table properties into [`ValidatedTableProperties`], then apply
//! the `maybe_*` helpers to derive the table features and properties that the final schema and
//! configuration require.

use std::collections::HashMap;

use crate::schema::SchemaRef;
use crate::schema::collation_utils::schema_has_collations;
use crate::schema::file_utils::schema_contains_file_type;
use crate::schema::variant_utils::schema_contains_variant_type;
use crate::table_features::{
    SET_TABLE_FEATURE_SUPPORTED_PREFIX, SET_TABLE_FEATURE_SUPPORTED_VALUE, TableFeature,
    add_feature_to_lists, auto_enable_property_driven_features, schema_contains_timestamp_ntz,
};
#[cfg(feature = "adaptive-metadata-in-dev")]
use crate::table_properties::ENABLE_ICEBERG_NATIVE_V4;
use crate::table_properties::{
    APPEND_ONLY, CHECKPOINT_INTERVAL, CHECKPOINT_POLICY, CHECKPOINT_WRITE_STATS_AS_JSON,
    CHECKPOINT_WRITE_STATS_AS_STRUCT, COLUMN_MAPPING_MODE, CheckpointPolicy,
    DATA_SKIPPING_NUM_INDEXED_COLS, DATA_SKIPPING_STATS_COLUMNS, DELETED_FILE_RETENTION_DURATION,
    DELTA_PROPERTY_PREFIX, ENABLE_CHANGE_DATA_FEED, ENABLE_DELETION_VECTORS,
    ENABLE_EXPIRED_LOG_CLEANUP, ENABLE_ICEBERG_COMPAT_V2, ENABLE_ICEBERG_COMPAT_V3,
    ENABLE_IN_COMMIT_TIMESTAMPS, ENABLE_ROW_TRACKING, ENABLE_TYPE_WIDENING, LOG_RETENTION_DURATION,
    PARQUET_COMPRESSION_CODEC, PARQUET_FORMAT_VERSION, ParquetCompressionCodec,
    SET_TRANSACTION_RETENTION_DURATION, TableProperties, strip_check_constraint_prefix,
};
use crate::{KernelError, KernelResult};

/// Table features that a transaction may add via `delta.feature.*=supported` or enablement
/// properties.
///
/// Feature signals (`delta.feature.X=supported`) are validated against this list.
/// Only features in this list can be enabled via feature signals.
pub(super) const ALLOWED_DELTA_FEATURES: &[TableFeature] = &[
    // DomainMetadata is required for clustering and other system domain operations
    TableFeature::DomainMetadata,
    // ColumnMapping enables column mapping (name/id mode)
    TableFeature::ColumnMapping,
    // InCommitTimestamp enables in-commit timestamps (writer-only)
    TableFeature::InCommitTimestamp,
    // VacuumProtocolCheck ensures consistent protocol checks during VACUUM
    TableFeature::VacuumProtocolCheck,
    // CatalogManaged enables catalog-managed table support
    TableFeature::CatalogManaged,
    // Note: Clustering is NOT included here. Users should not enable clustering via
    // `delta.feature.clustering = supported`. Instead, clustering is enabled by
    // specifying clustering columns via `with_data_layout()`.
    TableFeature::DeletionVectors,
    TableFeature::V2Checkpoint,
    // Simple protocol-only features: enabling these only updates the protocol action.
    // They can also be auto-enabled via their enablement properties (e.g. delta.appendOnly=true)
    // through `maybe_auto_enable_property_driven_features`.
    TableFeature::AppendOnly,
    TableFeature::ChangeDataFeed,
    TableFeature::TypeWidening,
    TableFeature::RowTracking,
    TableFeature::VariantType,
    TableFeature::VariantShredding,
    // Invariants is auto-enabled by `maybe_enable_invariants` when the schema has non-null
    // fields. Allowing explicit `delta.feature.invariants=supported` lets users pre-enable
    // the feature on an all-nullable table so a later ALTER TABLE ADD COLUMN NOT NULL does
    // not need a protocol upgrade.
    TableFeature::Invariants,
    // MaterializePartitionColumns keeps partition columns in the data files instead of
    // dropping them on write. There is no `delta.*` enablement property; the only opt-in at
    // create time is the explicit feature signal
    // `delta.feature.materializePartitionColumns=supported`.
    TableFeature::MaterializePartitionColumns,
    TableFeature::Collations,
    // IcebergCompatV2 is a writer-only feature that gates Iceberg V2 conversion compatibility.
    // Dependent features (ColumnMapping) are auto-added during create table.
    TableFeature::IcebergCompatV2,
    // IcebergCompatV3 is a writer-only feature that gates Iceberg V3 conversion compatibility.
    // Dependent features (ColumnMapping, RowTracking, DomainMetadata) are auto-added during
    // create table.
    TableFeature::IcebergCompatV3,
    #[cfg(feature = "adaptive-metadata-in-dev")]
    TableFeature::IcebergNativeV4Preview,
];

/// Features an update, but not CREATE TABLE, may add through `delta.feature.<name>=supported`.
/// Delta Kernel Java accepts these signals on update. A feature that Kernel cannot write, such as
/// `checkpointProtection` without the `checkpoint-protection` feature, is still rejected when the
/// final configuration is validated, as in Java's `validateKernelCanWriteToTable`.
pub(super) const UPDATE_ONLY_FEATURE_SIGNALS: &[TableFeature] = &[
    TableFeature::TimestampWithoutTimezone,
    TableFeature::CheckpointProtection,
    // Only the explicit signal adds it: an update never derives allowColumnDefaults.
    TableFeature::AllowColumnDefaults,
    TableFeature::TypeWideningPreview,
    TableFeature::VariantTypePreview,
    TableFeature::VariantShreddingPreview,
    TableFeature::CollationsPreview,
];

/// The single allow-list of `delta.*` properties a transaction may set. Any other `delta.*` key
/// is rejected.
pub(super) const ALLOWED_DELTA_PROPERTIES: &[&str] = &[
    // ColumnMapping mode property: triggers column mapping transform
    COLUMN_MAPPING_MODE,
    // InCommitTimestamp enablement property: triggers ICT auto-enablement
    ENABLE_IN_COMMIT_TIMESTAMPS,
    // Checkpoint stats format properties
    CHECKPOINT_WRITE_STATS_AS_JSON,
    CHECKPOINT_WRITE_STATS_AS_STRUCT,
    // Data skipping stats configuration. Neither property enables a feature; both control
    // the per-file stats schema the writer collects.
    DATA_SKIPPING_NUM_INDEXED_COLS,
    DATA_SKIPPING_STATS_COLUMNS,
    // Property-driven feature enablement properties
    ENABLE_DELETION_VECTORS,
    ENABLE_CHANGE_DATA_FEED,
    ENABLE_TYPE_WIDENING,
    APPEND_ONLY,
    ENABLE_ROW_TRACKING,
    // Set transaction retention duration: controls expiration of txn identifiers
    SET_TRANSACTION_RETENTION_DURATION,
    // Parquet format version: controls the Parquet writer version for data files
    PARQUET_FORMAT_VERSION,
    // Advisory compression setting for the connector's Parquet writer.
    PARQUET_COMPRESSION_CODEC,
    // IcebergCompatV2 enablement: triggers auto-enablement of ColumnMapping.
    ENABLE_ICEBERG_COMPAT_V2,
    // IcebergCompatV3 enablement: triggers auto-enablement of ColumnMapping,
    // RowTracking, DomainMetadata.
    ENABLE_ICEBERG_COMPAT_V3,
    // IcebergNativeV4 enablement: triggers auto-enablement of ColumnMapping (id), RowTracking,
    // DeletionVectors, InCommitTimestamp, DomainMetadata, and adaptiveMetadata.
    #[cfg(feature = "adaptive-metadata-in-dev")]
    ENABLE_ICEBERG_NATIVE_V4,
    // Log/checkpoint maintenance configs: stored verbatim, consumed by log cleanup / checkpoint
    // scheduling.
    LOG_RETENTION_DURATION,
    DELETED_FILE_RETENTION_DURATION,
    ENABLE_EXPIRED_LOG_CLEANUP,
    CHECKPOINT_INTERVAL,
    // Checkpoint policy: "v2" auto-enables the v2Checkpoint feature.
    CHECKPOINT_POLICY,
];

/// Result of validating and transforming table properties.
pub(super) struct ValidatedTableProperties {
    /// Table properties with feature signals removed (to be stored in metadata)
    pub(super) properties: HashMap<String, String>,
    /// Reader features extracted from feature signals (for ReaderWriter features)
    pub(super) reader_features: Vec<TableFeature>,
    /// Writer features extracted from feature signals (for all features)
    pub(super) writer_features: Vec<TableFeature>,
}

impl ValidatedTableProperties {
    /// Returns `true` iff `properties[key] == "true"`.
    pub(super) fn is_property_true(&self, key: &str) -> bool {
        self.properties.get(key).map(String::as_str) == Some("true")
    }
}

/// Conditionally adds the `variantType` feature to the protocol when the schema contains Variant
/// columns anywhere in the schema tree (top-level, nested structs, arrays, maps).
pub(super) fn maybe_enable_variant_type(
    schema: &SchemaRef,
    validated: &mut ValidatedTableProperties,
) {
    if schema_contains_variant_type(schema) {
        add_feature_to_lists(
            TableFeature::VariantType,
            &mut validated.reader_features,
            &mut validated.writer_features,
        );
    }
}

/// Conditionally adds the `timestampNtz` feature to the protocol when the schema contains
/// TimestampNTZ columns anywhere in the schema tree (top-level, nested structs, arrays, maps).
pub(super) fn maybe_enable_timestamp_ntz(
    schema: &SchemaRef,
    validated: &mut ValidatedTableProperties,
) {
    if schema_contains_timestamp_ntz(schema) {
        add_feature_to_lists(
            TableFeature::TimestampWithoutTimezone,
            &mut validated.reader_features,
            &mut validated.writer_features,
        );
    }
}

/// Conditionally adds the `fileType-preview` feature to the protocol when the schema contains File
/// columns anywhere in the schema tree (top-level, nested structs, arrays, maps).
///
/// The stable `fileType` feature is deliberately not written: the protocol change is still only
/// proposed, and a stable feature name is permanent once it is in a table's protocol.
pub(super) fn maybe_enable_file_type_preview(
    schema: &SchemaRef,
    validated: &mut ValidatedTableProperties,
) {
    if schema_contains_file_type(schema) {
        add_feature_to_lists(
            TableFeature::FileTypePreview,
            &mut validated.reader_features,
            &mut validated.writer_features,
        );
    }
}

/// Adds the stable collation feature for annotated schemas.
pub(super) fn maybe_enable_collations(
    schema: &SchemaRef,
    validated: &mut ValidatedTableProperties,
) {
    if schema_has_collations(schema) {
        add_feature_to_lists(
            TableFeature::Collations,
            &mut validated.reader_features,
            &mut validated.writer_features,
        );
    }
}

/// Auto-enables allowed property-driven features from the table properties (see
/// [`auto_enable_property_driven_features`]).
pub(super) fn maybe_auto_enable_property_driven_features(validated: &mut ValidatedTableProperties) {
    let table_properties = TableProperties::from(validated.properties.iter());
    auto_enable_property_driven_features(
        ALLOWED_DELTA_FEATURES,
        &table_properties,
        &mut validated.reader_features,
        &mut validated.writer_features,
    );
}

/// Ensures that `inCommitTimestamp` is enabled when `catalogManaged` is present. Adds the ICT
/// feature to the protocol and sets the enablement property if not already present.
pub(super) fn maybe_enable_ict_for_catalog_managed(
    validated: &mut ValidatedTableProperties,
) -> KernelResult<()> {
    let has_catalog_managed = validated
        .writer_features
        .contains(&TableFeature::CatalogManaged);
    if has_catalog_managed {
        if validated
            .properties
            .get(ENABLE_IN_COMMIT_TIMESTAMPS)
            .is_some_and(|v| v != "true")
        {
            return Err(KernelError::generic(format!(
                "Catalog-managed tables require '{ENABLE_IN_COMMIT_TIMESTAMPS}=true', \
                 but it was explicitly set to '{}'",
                validated.properties[ENABLE_IN_COMMIT_TIMESTAMPS]
            )));
        }
        add_feature_to_lists(
            TableFeature::InCommitTimestamp,
            &mut validated.reader_features,
            &mut validated.writer_features,
        );
        validated
            .properties
            .entry(ENABLE_IN_COMMIT_TIMESTAMPS.to_string())
            .or_insert_with(|| "true".to_string());
    }
    Ok(())
}

pub(super) fn maybe_enable_v2_checkpoint_for_policy(validated: &mut ValidatedTableProperties) {
    let is_v2_policy = validated
        .properties
        .get(CHECKPOINT_POLICY)
        .and_then(|v| CheckpointPolicy::try_from(v.as_str()).ok())
        == Some(CheckpointPolicy::V2);
    if is_v2_policy {
        add_feature_to_lists(
            TableFeature::V2Checkpoint,
            &mut validated.reader_features,
            &mut validated.writer_features,
        );
    }
}

/// Validates and transforms table properties for CREATE TABLE.
///
/// This function:
/// 1. Validates feature signals (`delta.feature.*`) against `ALLOWED_DELTA_FEATURES`
/// 2. Validates delta property keys (`delta.*`) against `ALLOWED_DELTA_PROPERTIES` and checks
///    compression codec values with the shared codec parser (case-insensitive, without trimming)
/// 3. Removes feature signals from properties (they shouldn't be stored in metadata)
/// 4. Extracts reader/writer features from validated feature signals
///
/// Non-delta properties (user/application properties) are always allowed.
///
/// Note: This function does not auto-set enablement properties. A feature signal like
/// `delta.feature.deletionVectors=supported` adds the feature to the protocol but does
/// not insert `delta.enableDeletionVectors=true` into the properties. Property-driven
/// auto-enablement is handled separately by [`maybe_auto_enable_property_driven_features`]
/// called after validation.
pub(super) fn validate_extract_table_features_and_properties(
    properties: HashMap<String, String>,
) -> KernelResult<ValidatedTableProperties> {
    validate_extract_features_and_properties_for(properties, "CREATE TABLE")
}

/// Like [`validate_extract_table_features_and_properties`], but names `operation` in errors.
pub(super) fn validate_extract_features_and_properties_for(
    properties: HashMap<String, String>,
    operation: &str,
) -> KernelResult<ValidatedTableProperties> {
    let mut reader_features = Vec::new();
    let mut writer_features = Vec::new();

    // Partition properties into feature signals and regular properties
    // Feature signals (delta.feature.X=supported) are processed but not stored in metadata
    // Feature signals are removed from the properties map.
    let (feature_signals, properties): (HashMap<_, _>, HashMap<_, _>) = properties
        .into_iter()
        .partition(|(k, _)| k.starts_with(SET_TABLE_FEATURE_SUPPORTED_PREFIX));

    // Process and validate feature signals
    for (key, value) in &feature_signals {
        // Safe: we partitioned for keys starting with this prefix above
        let Some(feature_name) = key.strip_prefix(SET_TABLE_FEATURE_SUPPORTED_PREFIX) else {
            continue;
        };

        // Validate that the value is "supported"
        if value != SET_TABLE_FEATURE_SUPPORTED_VALUE {
            return Err(KernelError::generic(format!(
                "Invalid value '{value}' for '{key}'. Only '{SET_TABLE_FEATURE_SUPPORTED_VALUE}' is allowed."
            )));
        }

        // Parse feature name to TableFeature (unknown features become TableFeature::Unknown)
        let feature: TableFeature = feature_name
            .parse()
            .unwrap_or_else(|_| TableFeature::Unknown(feature_name.to_string()));

        if !ALLOWED_DELTA_FEATURES.contains(&feature) {
            return Err(KernelError::generic(format!(
                "Enabling feature '{feature_name}' via '{key}' is not supported during {operation}"
            )));
        }

        // VariantShredding requires VariantType as a dependency
        if feature == TableFeature::VariantShredding {
            add_feature_to_lists(
                TableFeature::VariantType,
                &mut reader_features,
                &mut writer_features,
            );
        }

        // Add to appropriate feature lists based on feature type
        add_feature_to_lists(feature, &mut reader_features, &mut writer_features);
    }

    // Validate remaining delta.* properties against the allow list
    for (key, value) in &properties {
        if strip_check_constraint_prefix(key).is_some() {
            return Err(KernelError::generic(format!(
                "Setting CHECK constraint property '{key}' is not supported during {operation}"
            )));
        }
        if key.starts_with(DELTA_PROPERTY_PREFIX)
            && !ALLOWED_DELTA_PROPERTIES.contains(&key.as_str())
        {
            return Err(KernelError::generic(format!(
                "Setting delta property '{key}' is not supported during {operation}"
            )));
        }
        if key == PARQUET_COMPRESSION_CODEC {
            // Reject unsupported codecs at creation; snapshot parsing remains tolerant of them.
            ParquetCompressionCodec::try_from(value.as_str()).map_err(|_| {
                KernelError::generic(format!("Invalid value '{value}' for '{key}'"))
            })?;
        }
    }

    Ok(ValidatedTableProperties {
        properties,
        reader_features,
        writer_features,
    })
}

/// Canonicalizes the key case of recognized `delta.*` properties and rejects malformed values.
///
/// Recognized keys are those in [`ALLOWED_DELTA_PROPERTIES`], matched case-insensitively. A value
/// is malformed when it does not parse as the property's type. Feature signals
/// (`delta.feature.<name>`) are matched case-insensitively in both the prefix and the name of an
/// allowed feature. Other unrecognized keys with the exact `delta.` prefix pass through unchanged
/// for [`validate_extract_features_and_properties_for`] to validate. Unrecognized keys whose prefix
/// is a case variant of `delta.` are rejected. Application properties keep their original case.
pub(super) fn validate_and_normalize_delta_properties(
    properties: HashMap<String, String>,
    operation: &str,
) -> KernelResult<HashMap<String, String>> {
    let mut normalized = HashMap::with_capacity(properties.len());
    for (key, value) in properties {
        let lowercase_key = key.to_ascii_lowercase();
        let canonical_key = if !lowercase_key.starts_with(DELTA_PROPERTY_PREFIX) {
            key
        } else if lowercase_key.starts_with(SET_TABLE_FEATURE_SUPPORTED_PREFIX) {
            // Like Java's case-insensitive `TableFeatures.getTableFeature`, match the feature name
            // in any case. Unknown names are kept for the feature-signal validation to reject.
            let name = &key[SET_TABLE_FEATURE_SUPPORTED_PREFIX.len()..];
            let canonical_name = ALLOWED_DELTA_FEATURES
                .iter()
                .chain(UPDATE_ONLY_FEATURE_SIGNALS)
                .map(ToString::to_string)
                .find(|feature| feature.eq_ignore_ascii_case(name))
                .unwrap_or_else(|| name.to_string());
            format!("{SET_TABLE_FEATURE_SUPPORTED_PREFIX}{canonical_name}")
        } else if let Some(canonical) = ALLOWED_DELTA_PROPERTIES
            .iter()
            .find(|allowed| allowed.eq_ignore_ascii_case(&key))
        {
            // A recognized key must parse to a typed value; malformed values fall through to
            // `unknown_properties`, or leave every field unset for leniently parsed enums.
            let parsed = TableProperties::from([(*canonical, value.as_str())]);
            if parsed.unknown_properties.contains_key(*canonical)
                || parsed == TableProperties::default()
            {
                return Err(KernelError::generic(format!(
                    "Invalid value '{value}' for table property '{canonical}'"
                )));
            }
            canonical.to_string()
        } else if key.starts_with(DELTA_PROPERTY_PREFIX) {
            key
        } else {
            return Err(KernelError::generic(format!(
                "Setting delta property '{key}' is not supported during {operation}"
            )));
        };
        if normalized.insert(canonical_key.clone(), value).is_some() {
            return Err(KernelError::generic(format!(
                "Table property '{canonical_key}' is supplied more than once with different key \
                 case"
            )));
        }
    }
    Ok(normalized)
}
