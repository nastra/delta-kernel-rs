//! Validates a complete replacement schema for an existing table and derives the metadata that
//! commits it.
//!
//! Fields are identified by column-mapping ID, so the supplied schema must carry the IDs and
//! physical names of the fields it keeps (as returned by
//! [`Snapshot::schema`](crate::Snapshot::schema)). Fields without column-mapping metadata are new
//! and receive fresh identities; Kernel never accepts a caller-chosen identity for a new field.
//! This mirrors Delta Kernel Java's `SchemaUtils.validateUpdatedSchemaAndGetUpdatedSchema`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use self::type_widening::AllowedTypeWidening;
use crate::actions::Metadata;
use crate::expressions::ColumnName;
use crate::schema::diff::{FieldChange, FieldChangeType, FieldUpdate, SchemaDiff};
use crate::schema::validation::validate_schema;
use crate::schema::{
    ArrayType, ColumnMetadataKey, DataType, MapType, SchemaRef, StructField, StructType,
};
use crate::table_configuration::TableConfiguration;
use crate::table_features::{
    ColumnMappingMode, SET_TABLE_FEATURE_SUPPORTED_PREFIX, TableFeature,
    find_max_column_id_in_schema, physical_to_logical_column_name_and_type,
    try_assign_flat_column_mapping_info, validate_column_mapping_id,
};
use crate::table_properties::{
    COLUMN_MAPPING_MAX_COLUMN_ID, ENABLE_ICEBERG_COMPAT_V1, ENABLE_ICEBERG_COMPAT_V2,
    ENABLE_ICEBERG_COMPAT_V3, ENABLE_TYPE_WIDENING,
};
use crate::transaction::schema_evolution::reject_generated_or_identity_metadata;
use crate::transforms::{SchemaTransform, transform_output_type};
use crate::utils::{FoldWithOption as _, require};
use crate::{KernelError, KernelResult};

mod type_widening;

/// Replaces the table schema with `schema` and returns the evolved metadata and logical schema.
///
/// Supported changes are nullable field additions, drops, renames, reordering within a struct,
/// non-protocol metadata changes such as comments, nullability relaxation, and type widening (see
/// [`type_widening::apply_type_changes`]). Existing fields
/// keep their column-mapping IDs and physical names; new fields get fresh ones and advance
/// `delta.columnMapping.maxColumnId`, which never decreases.
///
/// The result is not validated against the table protocol; callers derive any required protocol
/// features and validate the final table configuration.
///
/// `properties_added` are the table properties set in the same request. Like Java's
/// `validateUpdatedSchemaAndGetUpdatedSchema`, which reads the new metadata, they decide whether
/// type widening is enabled and whether only Iceberg V2 widenings are allowed.
pub(crate) fn evolve_metadata_to_schema(
    table_config: &TableConfiguration,
    schema: &StructType,
    properties_added: &HashMap<String, String>,
) -> KernelResult<(Metadata, SchemaRef)> {
    let column_mapping_mode = table_config.column_mapping_mode();
    // Without column mapping, fields have no stable identity across renames and drops.
    require!(
        column_mapping_mode != ColumnMappingMode::None,
        KernelError::unsupported(
            "Updating the table schema requires column mapping to be enabled on the table"
        )
    );
    let current_schema = table_config.logical_schema();
    let configured_max_column_id = table_config
        .table_properties()
        .column_mapping_max_column_id
        .ok_or_else(|| {
            KernelError::invalid_protocol(
                "Column mapping is enabled but delta.columnMapping.maxColumnId is not set in \
                 table properties",
            )
        })?;
    validate_column_mapping_id(configured_max_column_id).map_err(|e| {
        KernelError::invalid_protocol(format!(
            "Table property `delta.columnMapping.maxColumnId`: {e}"
        ))
    })?;
    // IDs up to this value may belong to existing or dropped fields. Take the schema's max too in
    // case a non-conforming writer left the property stale.
    let mut max_column_id =
        configured_max_column_id.max(find_max_column_id_in_schema(&current_schema).unwrap_or(0));

    for field in schema.fields() {
        require!(
            !field.is_metadata_column(),
            KernelError::schema(format!(
                "Cannot update the schema: '{}' is a metadata column, which is not allowed in a \
                 table schema",
                field.name()
            ))
        );
        StructType::ensure_no_metadata_columns_in_field(field)?;
    }
    let mut current_ids = HashSet::new();
    for_each_struct_field(&current_schema, |field| {
        current_ids.extend(field.column_mapping_id());
        Ok(())
    })?;
    for_each_struct_field(schema, |field| {
        reject_caller_assigned_column_mapping_metadata(field, &current_ids)
    })?;

    let schema = assign_new_field_identities(schema, &mut max_column_id)?;
    let mut current_fields = HashMap::new();
    for_each_struct_field(&current_schema, |field| {
        current_fields.extend(field.column_mapping_id().map(|id| (id, field)));
        Ok(())
    })?;
    let schema = type_widening::apply_type_changes(
        &current_fields,
        &schema,
        allowed_type_widening(table_config, properties_added),
    )?;
    let diff = SchemaDiff::new(&current_schema, &schema)
        .map_err(|e| KernelError::schema(format!("Invalid schema update: {e}")))?;
    for added in &diff.added_fields {
        validate_added_field(added)?;
    }
    for update in &diff.updated_fields {
        validate_field_update(update)?;
    }
    validate_partition_columns_unchanged(table_config.metadata(), &current_schema, &schema)?;
    type_widening::validate_partition_column_type_changes(
        table_config.metadata().partition_columns(),
        &current_schema,
        &schema,
    )?;
    if let Some(columns) = &table_config.table_properties().data_skipping_stats_columns {
        validate_stats_columns_unchanged(columns, &current_schema, &schema)?;
    }
    validate_schema(
        &schema,
        column_mapping_mode,
        table_config.is_feature_enabled(&TableFeature::ChangeDataFeed),
    )?;

    let new_max_column_id = (max_column_id > configured_max_column_id).then_some(max_column_id);
    let schema = Arc::new(schema);
    let metadata = table_config
        .metadata()
        .clone()
        .with_schema(schema.clone())?
        .fold_with(new_max_column_id, |metadata, id| {
            metadata.with_configuration_entry(COLUMN_MAPPING_MAX_COLUMN_ID, id.to_string())
        });
    Ok((metadata, schema))
}

/// Returns the type changes the table permits once `properties_added` are applied.
///
/// Type widening is enabled by `delta.enableTypeWidening`, as in Java. The Iceberg V2 subset
/// applies while `icebergCompatV1` or `icebergCompatV2` is supported, as the protocol's type
/// widening writer requirements state, or while `icebergCompatV3` is enabled, as its RFC states.
fn allowed_type_widening(
    table_config: &TableConfiguration,
    properties_added: &HashMap<String, String>,
) -> AllowedTypeWidening {
    // The value from the request, or else the table's.
    let is_true = |key: &str| match properties_added.get(key) {
        Some(value) => value == "true",
        None => {
            table_config
                .metadata()
                .configuration()
                .get(key)
                .map(String::as_str)
                == Some("true")
        }
    };
    let signals_support = |feature: &TableFeature| {
        properties_added
            .get(&format!("{SET_TABLE_FEATURE_SUPPORTED_PREFIX}{feature}"))
            .is_some()
    };
    let enabled = is_true(ENABLE_TYPE_WIDENING);
    let iceberg_compat = [
        (TableFeature::IcebergCompatV1, ENABLE_ICEBERG_COMPAT_V1),
        (TableFeature::IcebergCompatV2, ENABLE_ICEBERG_COMPAT_V2),
    ]
    .iter()
    .any(|(feature, key)| {
        table_config.is_feature_supported(feature) || signals_support(feature) || is_true(key)
    }) || is_true(ENABLE_ICEBERG_COMPAT_V3);
    match (enabled, iceberg_compat) {
        (false, _) => AllowedTypeWidening::None,
        (true, false) => AllowedTypeWidening::Delta,
        (true, true) => AllowedTypeWidening::IcebergV2,
    }
}

/// Rejects an updated table configuration whose schema no longer contains every clustering column.
///
/// `physical_clustering_columns` are the clustering columns in effect after the transaction, by
/// physical name. Renaming a clustering column keeps its physical name and therefore its
/// clustering reference; dropping it would leave a dangling reference.
pub(crate) fn validate_clustering_columns_retained(
    read_config: &TableConfiguration,
    updated_config: &TableConfiguration,
    physical_clustering_columns: &[ColumnName],
) -> KernelResult<()> {
    let column_mapping_mode = updated_config.column_mapping_mode();
    let updated_schema = updated_config.logical_schema();
    for column in physical_clustering_columns {
        if physical_to_logical_column_name_and_type(&updated_schema, column, column_mapping_mode)
            .is_err()
        {
            let logical_name = physical_to_logical_column_name_and_type(
                &read_config.logical_schema(),
                column,
                read_config.column_mapping_mode(),
            )
            .map_or_else(|_| column.clone(), |(name, _)| name);
            return Err(KernelError::schema(format!(
                "Cannot drop clustering column '{logical_name}'"
            )));
        }
    }
    Ok(())
}

/// Rejects column-mapping metadata that does not identify an existing field.
///
/// Kernel assigns the identity of new fields. A caller-chosen ID or physical name could belong to
/// a field dropped earlier, whose values existing data files still contain.
fn reject_caller_assigned_column_mapping_metadata(
    field: &StructField,
    current_ids: &HashSet<i64>,
) -> KernelResult<()> {
    let annotations = field.validate_and_extract_existing_column_mapping_annotations()?;
    match (annotations.id, annotations.physical_name) {
        (None, None) => Ok(()),
        (Some(id), Some(_)) if current_ids.contains(&id) => Ok(()),
        (Some(id), Some(_)) => Err(KernelError::schema(format!(
            "Field '{}' has column mapping id {id}, which does not belong to an existing field. \
             Omit column mapping metadata from new fields; Kernel assigns it",
            field.name()
        ))),
        _ => Err(KernelError::schema(format!(
            "Field '{}' must carry both `{}` and `{}`, or neither",
            field.name(),
            ColumnMetadataKey::ColumnMappingId.as_ref(),
            ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
        ))),
    }
}

/// Assigns a fresh column-mapping ID and physical name to every field without one, advancing
/// `max_column_id`. Fields that keep their identity are left unchanged.
fn assign_new_field_identities(
    schema: &StructType,
    max_column_id: &mut i64,
) -> KernelResult<StructType> {
    let fields = schema
        .fields()
        .map(|field| {
            if field.column_mapping_id().is_none() {
                return try_assign_flat_column_mapping_info(field, max_column_id);
            }
            let mut field = field.clone();
            field.data_type = assign_new_identities_in_type(&field.data_type, max_column_id)?;
            Ok(field)
        })
        .collect::<KernelResult<Vec<_>>>()?;
    StructType::try_new(fields)
}

fn assign_new_identities_in_type(
    data_type: &DataType,
    max_column_id: &mut i64,
) -> KernelResult<DataType> {
    Ok(match data_type {
        DataType::Struct(struct_type) => {
            assign_new_field_identities(struct_type, max_column_id)?.into()
        }
        DataType::Array(array_type) => ArrayType::new(
            assign_new_identities_in_type(array_type.element_type(), max_column_id)?,
            array_type.contains_null(),
        )
        .into(),
        DataType::Map(map_type) => MapType::new(
            assign_new_identities_in_type(map_type.key_type(), max_column_id)?,
            assign_new_identities_in_type(map_type.value_type(), max_column_id)?,
            map_type.value_contains_null(),
        )
        .into(),
        other => other.clone(),
    })
}

/// Validates a field that is new in the updated schema, including its nested fields.
fn validate_added_field(added: &FieldChange) -> KernelResult<()> {
    let path = &added.path;
    // Existing data files do not contain the field, so readers must be able to return NULL.
    require!(
        added.field.is_nullable(),
        KernelError::schema(format!(
            "Cannot add non-nullable field '{path}': existing data files do not contain it"
        ))
    );
    reject_generated_or_identity_metadata(&added.field)?;
    for_each_struct_field_in(&added.field, |field| {
        require!(
            field
                .get_config_value(&ColumnMetadataKey::TypeChanges)
                .is_none(),
            KernelError::schema(format!(
                "Cannot add field '{}' under '{path}' with `{}` metadata",
                field.name(),
                ColumnMetadataKey::TypeChanges.as_ref()
            ))
        );
        require!(
            field
                .get_config_value(&ColumnMetadataKey::CurrentDefault)
                .is_none(),
            KernelError::schema(format!(
                "Cannot add field '{}' under '{path}' with a default value",
                field.name()
            ))
        );
        Ok(())
    })
}

/// Validates the changes to a field that exists in both schemas.
fn validate_field_update(update: &FieldUpdate) -> KernelResult<()> {
    let path = &update.path;
    for change in &update.change_types {
        match change {
            // Collation annotations are keyed by the field's logical name.
            FieldChangeType::Renamed
                if update
                    .before
                    .get_config_value(&ColumnMetadataKey::Collations)
                    .is_some() =>
            {
                return Err(KernelError::unsupported(format!(
                    "Renaming field '{}' with a collation is not supported",
                    update.before.name()
                )));
            }
            FieldChangeType::Renamed
            | FieldChangeType::NullabilityLoosened
            | FieldChangeType::ContainerNullabilityLoosened => {}
            FieldChangeType::NullabilityTightened
            | FieldChangeType::ContainerNullabilityTightened => {
                return Err(KernelError::schema(format!(
                    "Cannot tighten the nullability of existing field '{path}'"
                )));
            }
            // Validated and recorded by `type_widening::apply_type_changes`.
            FieldChangeType::TypeChanged => {}
            FieldChangeType::MetadataChanged => validate_metadata_change(update)?,
        }
    }
    Ok(())
}

/// Rejects changes to field metadata that the Delta protocol or Kernel interprets, such as
/// generation expressions, identity, invariants, defaults, and collations. Other metadata, such as
/// comments, may change freely. Column-mapping metadata is validated by the schema diff.
fn validate_metadata_change(update: &FieldUpdate) -> KernelResult<()> {
    let before = &update.before.metadata;
    let after = &update.after.metadata;
    let changed_key = before
        .keys()
        .chain(after.keys())
        .filter(|key| before.get(*key) != after.get(*key))
        // The type-change history is validated by `type_widening::apply_type_changes`.
        .filter(|key| key.as_str() != ColumnMetadataKey::TypeChanges.as_ref())
        .find(|key| is_protocol_metadata_key(key.as_str()));
    match changed_key {
        Some(key) => Err(KernelError::schema(format!(
            "Cannot change metadata '{key}' of existing field '{}'",
            update.path
        ))),
        None => Ok(()),
    }
}

fn is_protocol_metadata_key(key: &str) -> bool {
    key.starts_with("delta.") || key.parse::<ColumnMetadataKey>().is_ok()
}

/// Rejects dropping or renaming a partition column, including a rename that only changes case.
/// Partition columns are recorded by logical name in the metadata, and Kernel resolves them by
/// exact name, so a rename would orphan the partition values of existing files.
fn validate_partition_columns_unchanged(
    metadata: &Metadata,
    current_schema: &StructType,
    schema: &StructType,
) -> KernelResult<()> {
    for column in metadata.partition_columns() {
        let Some(id) = current_schema
            .fields()
            .find(|field| field.name().eq_ignore_ascii_case(column))
            .and_then(StructField::column_mapping_id)
        else {
            continue;
        };
        let kept = schema
            .fields()
            .any(|field| field.column_mapping_id() == Some(id) && field.name() == column);
        require!(
            kept,
            KernelError::schema(format!("Cannot drop or rename partition column '{column}'"))
        );
    }
    Ok(())
}

/// Rejects dropping or renaming a column listed in `delta.dataSkippingStatsColumns`, which names
/// columns by logical path.
fn validate_stats_columns_unchanged(
    columns: &[ColumnName],
    current_schema: &StructType,
    schema: &StructType,
) -> KernelResult<()> {
    for column in columns {
        let Some(ids) = column_mapping_ids_at_path(current_schema, column) else {
            continue;
        };
        require!(
            column_mapping_ids_at_path(schema, column).as_ref() == Some(&ids),
            KernelError::schema(format!(
                "Cannot drop or rename column '{column}': it is listed in \
                 delta.dataSkippingStatsColumns"
            ))
        );
    }
    Ok(())
}

/// Returns the column-mapping IDs of the struct fields along `path`, matching names
/// case-insensitively, or `None` if the path does not resolve.
fn column_mapping_ids_at_path(schema: &StructType, path: &ColumnName) -> Option<Vec<i64>> {
    let mut ids = Vec::new();
    let mut current = Some(schema);
    for segment in path.path() {
        let field = current?
            .fields()
            .find(|field| field.name().eq_ignore_ascii_case(segment))?;
        ids.push(field.column_mapping_id()?);
        current = match field.data_type() {
            DataType::Struct(struct_type) => Some(struct_type),
            _ => None,
        };
    }
    Some(ids)
}

/// Calls `f` for every struct field in `schema`, at any nesting depth.
fn for_each_struct_field<'a>(
    schema: &'a StructType,
    f: impl FnMut(&'a StructField) -> KernelResult<()>,
) -> KernelResult<()> {
    StructFieldVisitor(f).transform_struct(schema)
}

/// Calls `f` for `field` and every struct field nested in it.
fn for_each_struct_field_in<'a>(
    field: &'a StructField,
    f: impl FnMut(&'a StructField) -> KernelResult<()>,
) -> KernelResult<()> {
    StructFieldVisitor(f).transform_struct_field(field)
}

struct StructFieldVisitor<F>(F);

impl<'a, F: FnMut(&'a StructField) -> KernelResult<()>> SchemaTransform<'a>
    for StructFieldVisitor<F>
{
    transform_output_type!(|'a, T| KernelResult<()>);

    fn transform_struct_field(&mut self, field: &'a StructField) -> KernelResult<()> {
        (self.0)(field)?;
        self.recurse_into_struct_field(field)
    }
}
