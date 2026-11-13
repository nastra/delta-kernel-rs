//! Type widening for complete schema updates: validates type changes of retained fields and
//! records them in the `delta.typeChanges` history of each field.
//!
//! Mirrors Delta Kernel Java's `TypeWideningChecker.isWideningSupported` and the type-change
//! bookkeeping in `SchemaUtils.computeSchemaChangesById`.

use std::collections::HashMap;

use serde_json::Value;

use crate::schema::{
    ArrayType, ColumnMetadataKey, DataType, DecimalType, MapType, MetadataValue, PrimitiveType,
    StructField, StructType,
};
use crate::table_features::is_iceberg_allowed_type_change;
use crate::utils::require;
use crate::{KernelError, KernelResult};

/// Validates the type of every field of `schema` whose column-mapping ID appears in
/// `current_fields`, and returns `schema` with each field's type-change history up to date.
///
/// A retained field inherits the `delta.typeChanges` history of its current version when it
/// omits the key, and must not alter it otherwise. A primitive type of the field, or of an array
/// element or map key or value inside it, may only change by a widening the Delta protocol
/// supports, and only when `widening` permits type changes. Each such change is appended to the
/// history, with a `fieldPath` for changes inside an array or map. Nested struct fields carry their
/// own history.
pub(super) fn apply_type_changes(
    current_fields: &HashMap<i64, &StructField>,
    schema: &StructType,
    widening: AllowedTypeWidening,
) -> KernelResult<StructType> {
    TypeChangeRecorder {
        current_fields,
        widening,
    }
    .apply_to_struct(schema, &[])
}

/// The type changes a table permits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AllowedTypeWidening {
    /// Type widening is not enabled.
    None,
    /// Every widening the Delta protocol supports.
    Delta,
    /// The Iceberg V2 subset, required while `icebergCompatV1` or `icebergCompatV2` is supported,
    /// even if it is not enabled, or while `icebergCompatV3` is enabled.
    IcebergV2,
}

impl AllowedTypeWidening {
    fn allows(self, from: &DataType, to: &DataType) -> bool {
        match (self, from, to) {
            (Self::IcebergV2, _, _) => is_iceberg_allowed_type_change(from, to),
            (_, DataType::Primitive(from), DataType::Primitive(to)) => {
                is_supported_type_widening(from, to)
            }
            _ => false,
        }
    }
}

/// Rejects type changes of partition columns whose existing partition values Kernel cannot parse
/// as the new type. Partition values are stored as strings in the log, and Kernel parses neither a
/// date string as a `timestamp_ntz` nor a decimal string whose scale differs from the target scale.
pub(super) fn validate_partition_column_type_changes<'a>(
    partition_columns: impl IntoIterator<Item = &'a String>,
    current_schema: &StructType,
    schema: &StructType,
) -> KernelResult<()> {
    use PrimitiveType::*;
    for column in partition_columns {
        let (Some(current), Some(updated)) = (current_schema.field(column), schema.field(column))
        else {
            continue;
        };
        let (DataType::Primitive(from), DataType::Primitive(to)) =
            (current.data_type(), updated.data_type())
        else {
            continue;
        };
        let readable = match (from, to) {
            (Decimal(from), Decimal(to)) => from.scale() == to.scale(),
            (Byte | Short | Integer | Long, Decimal(to)) => to.scale() == 0,
            (Date, TimestampNtz) => false,
            _ => true,
        };
        require!(
            readable,
            KernelError::unsupported(format!(
                "Cannot change the type of partition column '{column}' from {from} to {to}: \
                 Kernel cannot read the existing partition values as the new type"
            ))
        );
    }
    Ok(())
}

/// Returns `true` if the Delta protocol allows widening a column from `from` to `to`.
fn is_supported_type_widening(from: &PrimitiveType, to: &PrimitiveType) -> bool {
    use PrimitiveType::*;
    // `Decimal(p, s)` is reachable when `p - s` integer digits and `s` fractional digits fit.
    let fits_decimal = |integer_digits: u8, scale: u8, to: &DecimalType| {
        to.scale() >= scale && to.precision() - to.scale() >= integer_digits
    };
    match (from, to) {
        (Byte, Short | Integer | Long | Double)
        | (Short, Integer | Long | Double)
        | (Integer, Long | Double)
        | (Float, Double)
        | (Date, TimestampNtz) => true,
        (Decimal(from), Decimal(to)) => {
            fits_decimal(from.precision() - from.scale(), from.scale(), to)
        }
        (Byte | Short | Integer, Decimal(to)) => fits_decimal(10, 0, to),
        (Long, Decimal(to)) => fits_decimal(20, 0, to),
        _ => false,
    }
}

struct TypeChangeRecorder<'a> {
    current_fields: &'a HashMap<i64, &'a StructField>,
    widening: AllowedTypeWidening,
}

/// A type change recorded in `delta.typeChanges`.
struct TypeChange {
    from_type: DataType,
    to_type: DataType,
    /// Path from the field to the changed array element or map key or value, if any.
    field_path: Option<String>,
}

impl TypeChangeRecorder<'_> {
    fn apply_to_struct(&self, schema: &StructType, parent: &[&str]) -> KernelResult<StructType> {
        let fields = schema
            .fields()
            .map(|field| self.apply_to_field(field, parent))
            .collect::<KernelResult<Vec<_>>>()?;
        StructType::try_new(fields)
    }

    fn apply_to_field(&self, field: &StructField, parent: &[&str]) -> KernelResult<StructField> {
        let path = [parent, &[field.name().as_str()]].concat();
        let mut field = field.clone();
        if let Some(current) = field
            .column_mapping_id()
            .and_then(|id| self.current_fields.get(&id))
        {
            self.record_type_changes(current, &mut field, &path.join("."))?;
        }
        field.data_type = self.apply_to_type(&field.data_type, &path)?;
        Ok(field)
    }

    fn apply_to_type(&self, data_type: &DataType, path: &[&str]) -> KernelResult<DataType> {
        Ok(match data_type {
            DataType::Struct(struct_type) => self.apply_to_struct(struct_type, path)?.into(),
            DataType::Array(array_type) => ArrayType::new(
                self.apply_to_type(array_type.element_type(), &[path, &["element"]].concat())?,
                array_type.contains_null(),
            )
            .into(),
            DataType::Map(map_type) => MapType::new(
                self.apply_to_type(map_type.key_type(), &[path, &["key"]].concat())?,
                self.apply_to_type(map_type.value_type(), &[path, &["value"]].concat())?,
                map_type.value_contains_null(),
            )
            .into(),
            other => other.clone(),
        })
    }

    /// Validates the type change from `current` to `field` and updates the history of `field`.
    fn record_type_changes(
        &self,
        current: &StructField,
        field: &mut StructField,
        path: &str,
    ) -> KernelResult<()> {
        let key = ColumnMetadataKey::TypeChanges.as_ref();
        match (current.metadata.get(key), field.metadata.get(key)) {
            // Schemas built by the caller may omit the history; the protocol requires keeping it.
            (Some(history), None) => {
                field.metadata.insert(key.into(), history.clone());
            }
            (current_history, history) if current_history != history => {
                return Err(KernelError::schema(format!(
                    "Cannot change metadata '{key}' of existing field '{path}'"
                )));
            }
            _ => {}
        }

        let mut changes = Vec::new();
        collect_type_changes(
            current.data_type(),
            field.data_type(),
            &mut Vec::new(),
            path,
            &mut changes,
        )?;
        if let Some(change) = changes.iter().find(|change| {
            self.widening == AllowedTypeWidening::IcebergV2
                && !self.widening.allows(&change.from_type, &change.to_type)
        }) {
            return Err(KernelError::schema(format!(
                "Cannot change the type of existing field '{path}' from {} to {}: tables that \
                 support icebergCompatV1 or V2, or enable icebergCompatV3, only allow type \
                 changes that Iceberg V2 supports",
                change.from_type, change.to_type
            )));
        }
        let Some(first) = changes.first() else {
            return Ok(());
        };
        require!(
            self.widening != AllowedTypeWidening::None,
            KernelError::schema(format!(
                "Cannot change the type of existing field '{path}' from {} to {}: type widening \
                 is not enabled on the table",
                first.from_type, first.to_type
            ))
        );
        let mut history = match field.metadata.get(key) {
            None => Vec::new(),
            Some(MetadataValue::Other(Value::Array(history))) => history.clone(),
            Some(other) => {
                return Err(KernelError::schema(format!(
                    "Field '{path}' has a non-array `{key}` annotation: {other}"
                )));
            }
        };
        for change in changes {
            history.push(change.into_json()?);
        }
        field
            .metadata
            .insert(key.into(), MetadataValue::Other(Value::Array(history)));
        Ok(())
    }
}

impl TypeChange {
    fn into_json(self) -> KernelResult<Value> {
        let mut change = serde_json::Map::new();
        change.insert("fromType".into(), serde_json::to_value(&self.from_type)?);
        change.insert("toType".into(), serde_json::to_value(&self.to_type)?);
        if let Some(path) = self.field_path {
            change.insert("fieldPath".into(), Value::String(path));
        }
        Ok(Value::Object(change))
    }
}

/// Collects the primitive type changes between `current` and `updated`, descending through array
/// elements and map keys and values but not into structs, whose fields have their own identity.
fn collect_type_changes(
    current: &DataType,
    updated: &DataType,
    container_path: &mut Vec<&'static str>,
    field_path: &str,
    out: &mut Vec<TypeChange>,
) -> KernelResult<()> {
    let mut descend = |segment: &'static str,
                       current: &DataType,
                       updated: &DataType,
                       out: &mut Vec<TypeChange>| {
        container_path.push(segment);
        let result = collect_type_changes(current, updated, container_path, field_path, out);
        container_path.pop();
        result
    };
    match (current, updated) {
        (DataType::Struct(_), DataType::Struct(_)) => Ok(()),
        (DataType::Array(current), DataType::Array(updated)) => descend(
            "element",
            current.element_type(),
            updated.element_type(),
            out,
        ),
        (DataType::Map(current), DataType::Map(updated)) => {
            descend("key", current.key_type(), updated.key_type(), out)?;
            descend("value", current.value_type(), updated.value_type(), out)
        }
        _ if current == updated => Ok(()),
        (DataType::Primitive(from), DataType::Primitive(to))
            if is_supported_type_widening(from, to) =>
        {
            out.push(TypeChange {
                from_type: current.clone(),
                to_type: updated.clone(),
                field_path: (!container_path.is_empty()).then(|| container_path.join(".")),
            });
            Ok(())
        }
        _ => {
            let location = match container_path.is_empty() {
                true => String::new(),
                false => format!(" at '{}'", container_path.join(".")),
            };
            Err(KernelError::schema(format!(
                "Cannot change the type of existing field '{field_path}'{location} from {current} \
                 to {updated}"
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn decimal(precision: u8, scale: u8) -> PrimitiveType {
        PrimitiveType::Decimal(DecimalType::try_new(precision, scale).unwrap())
    }

    #[rstest]
    #[case::byte_to_short(PrimitiveType::Byte, PrimitiveType::Short, true)]
    #[case::byte_to_long(PrimitiveType::Byte, PrimitiveType::Long, true)]
    #[case::short_to_integer(PrimitiveType::Short, PrimitiveType::Integer, true)]
    #[case::integer_to_long(PrimitiveType::Integer, PrimitiveType::Long, true)]
    #[case::integer_to_double(PrimitiveType::Integer, PrimitiveType::Double, true)]
    #[case::float_to_double(PrimitiveType::Float, PrimitiveType::Double, true)]
    #[case::date_to_timestamp_ntz(PrimitiveType::Date, PrimitiveType::TimestampNtz, true)]
    #[case::decimal_precision(decimal(10, 2), decimal(12, 2), true)]
    #[case::decimal_precision_and_scale(decimal(10, 2), decimal(12, 4), true)]
    #[case::integer_to_decimal(PrimitiveType::Integer, decimal(10, 0), true)]
    #[case::short_to_wide_decimal(PrimitiveType::Short, decimal(12, 2), true)]
    #[case::long_to_decimal(PrimitiveType::Long, decimal(20, 0), true)]
    #[case::long_to_narrow_decimal(PrimitiveType::Long, decimal(19, 0), false)]
    #[case::integer_to_narrow_decimal(PrimitiveType::Integer, decimal(11, 2), false)]
    #[case::decimal_scale_without_precision(decimal(10, 2), decimal(10, 4), false)]
    #[case::decimal_narrowing(decimal(10, 2), decimal(8, 2), false)]
    #[case::long_to_integer(PrimitiveType::Long, PrimitiveType::Integer, false)]
    #[case::long_to_double(PrimitiveType::Long, PrimitiveType::Double, false)]
    #[case::double_to_float(PrimitiveType::Double, PrimitiveType::Float, false)]
    #[case::integer_to_string(PrimitiveType::Integer, PrimitiveType::String, false)]
    #[case::date_to_timestamp(PrimitiveType::Date, PrimitiveType::Timestamp, false)]
    #[case::timestamp_to_timestamp_ntz(
        PrimitiveType::Timestamp,
        PrimitiveType::TimestampNtz,
        false
    )]
    fn type_widening_follows_the_protocol(
        #[case] from: PrimitiveType,
        #[case] to: PrimitiveType,
        #[case] supported: bool,
    ) {
        assert_eq!(is_supported_type_widening(&from, &to), supported);
    }

    #[rstest]
    #[case::top_level(
        DataType::INTEGER,
        DataType::LONG,
        json!({ "fromType": "integer", "toType": "long" })
    )]
    #[case::array_element(
        ArrayType::new(DataType::INTEGER, true).into(),
        ArrayType::new(DataType::LONG, true).into(),
        json!({ "fromType": "integer", "toType": "long", "fieldPath": "element" })
    )]
    #[case::map_value_in_array(
        ArrayType::new(MapType::new(DataType::STRING, DataType::FLOAT, true), true).into(),
        ArrayType::new(MapType::new(DataType::STRING, DataType::DOUBLE, true), true).into(),
        json!({ "fromType": "float", "toType": "double", "fieldPath": "element.value" })
    )]
    fn type_changes_record_the_container_path(
        #[case] current: DataType,
        #[case] updated: DataType,
        #[case] expected: Value,
    ) {
        let mut changes = Vec::new();
        collect_type_changes(&current, &updated, &mut Vec::new(), "f", &mut changes).unwrap();
        let changes: Vec<_> = changes
            .into_iter()
            .map(|change| change.into_json().unwrap())
            .collect();
        assert_eq!(changes, [expected]);
    }

    #[rstest]
    #[case::array_to_map(
        ArrayType::new(DataType::INTEGER, true).into(),
        MapType::new(DataType::INTEGER, DataType::INTEGER, true).into(),
        "Cannot change the type of existing field 'f' from array<integer> to map<integer, integer>"
    )]
    #[case::narrowing_map_key(
        MapType::new(DataType::LONG, DataType::STRING, true).into(),
        MapType::new(DataType::INTEGER, DataType::STRING, true).into(),
        "Cannot change the type of existing field 'f' at 'key' from long to integer"
    )]
    fn unsupported_type_changes_are_rejected(
        #[case] current: DataType,
        #[case] updated: DataType,
        #[case] expected: &str,
    ) {
        let error = collect_type_changes(&current, &updated, &mut Vec::new(), "f", &mut Vec::new())
            .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}
