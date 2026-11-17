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
    ColumnMappingMode, TableFeature, find_max_column_id_in_schema,
    physical_to_logical_column_name_and_type, try_assign_flat_column_mapping_info,
    validate_column_mapping_id,
};
use crate::table_properties::COLUMN_MAPPING_MAX_COLUMN_ID;
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
/// Whether type widening is enabled, and whether only Iceberg V2 widenings are allowed, is read
/// from `table_config`. Property updates cannot change the keys involved on an existing table, so
/// this is also the final configuration, which Java's `validateUpdatedSchemaAndGetUpdatedSchema`
/// reads.
pub(crate) fn evolve_metadata_to_schema(
    table_config: &TableConfiguration,
    schema: &StructType,
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

    // Like Java's `SchemaUtils.validateSchema`: dropping every column would leave a table that can
    // no longer be scanned, and re-added columns get fresh identities, so the data would be lost.
    require!(
        schema.fields().next().is_some(),
        KernelError::schema(
            "Cannot update the schema: the schema must contain at least one column"
        )
    );
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
    let schema = inherit_kernel_managed_metadata(&schema, &current_fields)?;
    let schema = type_widening::apply_type_changes(
        &current_fields,
        &schema,
        allowed_type_widening(table_config),
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
    // Property updates cannot enable change data feed on an existing table, so the read
    // configuration decides the reserved-name check. The `delta.dataSkippingStatsColumns` check
    // runs on the final configuration in the builder, see `validate_stats_columns_unchanged`.
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

/// Returns the type changes `table_config` permits.
///
/// Type widening is enabled by `delta.enableTypeWidening`, as in Java. The Iceberg V2 subset
/// applies while `icebergCompatV1` or `icebergCompatV2` is supported, as the protocol's type
/// widening writer requirements state, or while `icebergCompatV3` is enabled, as its RFC states.
fn allowed_type_widening(table_config: &TableConfiguration) -> AllowedTypeWidening {
    let properties = table_config.table_properties();
    let iceberg_compat = [TableFeature::IcebergCompatV1, TableFeature::IcebergCompatV2]
        .iter()
        .any(|feature| table_config.is_feature_supported(feature))
        || properties.enable_iceberg_compat_v3 == Some(true);
    match (
        properties.enable_type_widening == Some(true),
        iceberg_compat,
    ) {
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

/// Returns `true` for field metadata that Kernel maintains for existing fields: the Parquet field
/// ids and the IcebergCompat nested column-mapping ids.
fn is_kernel_maintained_id_key(key: &str) -> bool {
    key.starts_with("parquet.field.") || key == ColumnMetadataKey::ColumnMappingNestedIds.as_ref()
}

/// Carries the Parquet field ids and nested column-mapping ids of each retained field over from the
/// current schema, so a caller's copy can neither drop nor change them. The `delta.typeChanges`
/// history is handled by [`type_widening::apply_type_changes`].
fn inherit_kernel_managed_metadata(
    schema: &StructType,
    current_fields: &HashMap<i64, &StructField>,
) -> KernelResult<StructType> {
    let fields = schema
        .fields()
        .map(|field| {
            let mut field = field.clone();
            if let Some(current) = field
                .column_mapping_id()
                .and_then(|id| current_fields.get(&id))
            {
                field
                    .metadata
                    .retain(|key, _| !is_kernel_maintained_id_key(key.as_str()));
                field.metadata.extend(
                    current
                        .metadata
                        .iter()
                        .filter(|(key, _)| is_kernel_maintained_id_key(key.as_str()))
                        .map(|(key, value)| (key.clone(), value.clone())),
                );
            }
            field.data_type =
                inherit_kernel_managed_metadata_in_type(&field.data_type, current_fields)?;
            Ok(field)
        })
        .collect::<KernelResult<Vec<_>>>()?;
    StructType::try_new(fields)
}

fn inherit_kernel_managed_metadata_in_type(
    data_type: &DataType,
    current_fields: &HashMap<i64, &StructField>,
) -> KernelResult<DataType> {
    Ok(match data_type {
        DataType::Struct(struct_type) => {
            inherit_kernel_managed_metadata(struct_type, current_fields)?.into()
        }
        DataType::Array(array_type) => ArrayType::new(
            inherit_kernel_managed_metadata_in_type(array_type.element_type(), current_fields)?,
            array_type.contains_null(),
        )
        .into(),
        DataType::Map(map_type) => MapType::new(
            inherit_kernel_managed_metadata_in_type(map_type.key_type(), current_fields)?,
            inherit_kernel_managed_metadata_in_type(map_type.value_type(), current_fields)?,
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
        // As in Java's `validateUpdatedSchemaCompatibility`, every new struct field must be
        // nullable, including fields nested in structs, arrays, and maps of the added field.
        require!(
            field.is_nullable(),
            KernelError::schema(format!(
                "Cannot add non-nullable field '{}' under '{path}': existing data files do not \
                 contain it",
                field.name()
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
        // Kernel maintains these for existing fields; a new field has no history or file ids.
        let managed_key = [
            ColumnMetadataKey::TypeChanges,
            ColumnMetadataKey::ParquetFieldId,
            ColumnMetadataKey::ParquetFieldNestedIds,
            ColumnMetadataKey::InternalColumn,
        ]
        .into_iter()
        .find(|key| field.get_config_value(key).is_some());
        if let Some(key) = managed_key {
            return Err(KernelError::schema(format!(
                "Cannot add field '{}' under '{path}' with Kernel-managed metadata '{}'",
                field.name(),
                key.as_ref()
            )));
        }
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
        let Some(current) = current_schema
            .fields()
            .find(|field| field.name().eq_ignore_ascii_case(column))
        else {
            continue;
        };
        let Some(id) = current.column_mapping_id() else {
            continue;
        };
        // Compare with the current field's name: the metadata may spell the column in another
        // case. Matching by ID also rejects dropping the column and re-adding one with the same
        // name, whose new physical name would orphan the existing partition values.
        let kept = schema
            .fields()
            .any(|field| field.column_mapping_id() == Some(id) && field.name() == current.name());
        require!(
            kept,
            KernelError::schema(format!("Cannot drop or rename partition column '{column}'"))
        );
    }
    Ok(())
}

/// Rejects dropping or renaming a column listed in `delta.dataSkippingStatsColumns`, which names
/// columns by logical path.
pub(crate) fn validate_stats_columns_unchanged(
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::*;
    use crate::schema::{DataType, MetadataValue};

    fn field_with_id(name: &str, id: i64) -> StructField {
        let mut field = StructField::nullable(name, DataType::STRING);
        field.metadata.insert(
            ColumnMetadataKey::ColumnMappingId.as_ref().into(),
            MetadataValue::Number(id),
        );
        field
    }

    #[rstest::rstest]
    #[case::unchanged("part", true)]
    #[case::renamed("partition", false)]
    fn partition_columns_recorded_in_another_case_are_matched_by_the_field_name(
        #[case] updated_name: &str,
        #[case] accepted: bool,
    ) {
        let current = StructType::try_new([field_with_id("part", 1)]).unwrap();
        let updated = StructType::try_new([field_with_id(updated_name, 1)]).unwrap();
        // The metadata spells the partition column in a different case than the schema.
        let metadata = Metadata::try_new(
            None,
            None,
            Arc::new(current.clone()),
            vec!["PART".to_string()],
            0,
            HashMap::new(),
        )
        .unwrap();
        let result = validate_partition_columns_unchanged(&metadata, &current, &updated);
        assert_eq!(result.is_ok(), accepted, "{result:?}");
    }

    fn table_config(properties: &[(&str, &str)]) -> TableConfiguration {
        use crate::committer::FileSystemCommitter;
        use crate::engine::sync::SyncEngine;
        use crate::object_store::memory::InMemory;
        use crate::transaction::create_table::create_table;

        let engine = SyncEngine::new_with_store(Arc::new(InMemory::new()));
        let schema =
            Arc::new(StructType::try_new([StructField::nullable("i", DataType::INTEGER)]).unwrap());
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

    #[rstest::rstest]
    #[case::not_enabled(&[], AllowedTypeWidening::None)]
    #[case::enabled(&[("delta.enableTypeWidening", "true")], AllowedTypeWidening::Delta)]
    #[case::iceberg_compat_v2_supported_not_enabled(
        &[
            ("delta.enableTypeWidening", "true"),
            ("delta.columnMapping.mode", "name"),
            ("delta.feature.icebergCompatV2", "supported"),
        ],
        AllowedTypeWidening::IcebergV2
    )]
    #[case::iceberg_compat_v3_enabled(
        &[
            ("delta.enableTypeWidening", "true"),
            ("delta.columnMapping.mode", "name"),
            ("delta.enableIcebergCompatV3", "true"),
        ],
        AllowedTypeWidening::IcebergV2
    )]
    fn allowed_type_widening_follows_type_widening_and_iceberg_compat(
        #[case] properties: &[(&str, &str)],
        #[case] expected: AllowedTypeWidening,
    ) {
        assert_eq!(allowed_type_widening(&table_config(properties)), expected);
    }
}
