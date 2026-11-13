//! Integration tests for replacing a table schema with `with_updated_schema`.

use std::collections::HashMap;
use std::sync::Arc;

use delta_kernel::arrow::array::{Array, AsArray as _, Int32Array, StringArray};
use delta_kernel::arrow::datatypes::Int32Type;
use delta_kernel::arrow::record_batch::RecordBatch;
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::engine::arrow_conversion::TryIntoArrow as _;
use delta_kernel::schema::{
    ArrayType, ColumnMetadataKey, DataType, MetadataValue, SchemaRef, StructField, StructType,
    schema_ref,
};
use delta_kernel::snapshot::Snapshot;
use delta_kernel::table_features::TableFeature;
use delta_kernel::transaction::UpdateTableOperation;
use delta_kernel::transaction::create_table::create_table;
use delta_kernel::transaction::data_layout::DataLayout;
use delta_kernel::{Engine, KernelError, SnapshotRef};
use rstest::rstest;
use test_utils::{
    assert_result_error_with_message, read_scan, test_table_setup, write_batch_to_table,
};

fn flat_schema() -> SchemaRef {
    schema_ref! {
        not_null "id": INTEGER,
        nullable "name": STRING,
        nullable "zip": STRING,
    }
}

fn nested_schema() -> SchemaRef {
    let tags = ArrayType::new(
        StructType::try_new([
            StructField::nullable("label", DataType::STRING),
            StructField::not_null("weight", DataType::INTEGER),
        ])
        .unwrap(),
        true,
    );
    let address = StructType::try_new([
        StructField::not_null("city", DataType::STRING),
        StructField::nullable("zip", DataType::STRING),
    ])
    .unwrap();
    Arc::new(
        StructType::try_new([
            StructField::not_null("id", DataType::INTEGER),
            StructField::nullable("address", address),
            StructField::nullable("tags", tags),
            StructField::nullable("part", DataType::STRING),
        ])
        .unwrap(),
    )
}

fn create(
    table_path: &str,
    engine: &dyn Engine,
    schema: SchemaRef,
    properties: &[(&str, &str)],
    layout: Option<DataLayout>,
) -> delta_kernel::Result<SnapshotRef> {
    let mut builder = create_table(table_path, schema, "test-engine", true)
        .with_table_properties(properties.to_vec());
    if let Some(layout) = layout {
        builder = builder.with_data_layout(layout);
    }
    Ok(builder
        .build(engine, Box::new(FileSystemCommitter::new()))?
        .commit(engine)?
        .unwrap_post_commit_snapshot())
}

fn update_schema(
    snapshot: &SnapshotRef,
    engine: &dyn Engine,
    schema: StructType,
) -> delta_kernel::Result<SnapshotRef> {
    Ok(snapshot
        .clone()
        .update_table(UpdateTableOperation::AlterTable, "test-engine", false)
        .with_updated_schema(Arc::new(schema))
        .build(engine, Box::new(FileSystemCommitter::new()))?
        .commit(engine)?
        .unwrap_post_commit_snapshot())
}

fn field(schema: &StructType, name: &str) -> StructField {
    schema.field(name).unwrap().clone()
}

/// Returns the fields of `schema` followed by `added`.
fn appended(schema: &StructType, added: StructField) -> Vec<StructField> {
    schema.fields().cloned().chain([added]).collect()
}

fn with_fields(fields: impl IntoIterator<Item = StructField>) -> StructType {
    StructType::try_new(fields).unwrap()
}

fn physical_name(field: &StructField) -> &MetadataValue {
    field
        .get_config_value(&ColumnMetadataKey::ColumnMappingPhysicalName)
        .unwrap()
}

fn max_column_id(snapshot: &Snapshot) -> i64 {
    snapshot
        .table_properties()
        .column_mapping_max_column_id
        .unwrap()
}

/// Asserts that `after` keeps the column-mapping identity of `before`.
fn assert_same_identity(before: &StructField, after: &StructField) {
    assert_eq!(after.column_mapping_id(), before.column_mapping_id());
    assert_eq!(physical_name(after), physical_name(before));
}

#[rstest]
#[tokio::test]
async fn updated_schema_renames_drops_reorders_and_adds_columns_preserving_data(
    #[values("name", "id")] cm_mode: &str,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        flat_schema(),
        &[("delta.columnMapping.mode", cm_mode)],
        None,
    )?;
    let batch = RecordBatch::try_new(
        Arc::new(
            snapshot
                .schema()
                .as_ref()
                .try_into_arrow()
                .map_err(KernelError::from)?,
        ),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(StringArray::from(vec!["z1", "z2"])),
        ],
    )
    .map_err(KernelError::from)?;
    let snapshot = write_batch_to_table(&snapshot, engine.as_ref(), batch, HashMap::new()).await?;
    let old_schema = snapshot.schema();
    let old_max_id = max_column_id(&snapshot);

    let full_name = field(&old_schema, "name")
        .with_name("full_name")
        .add_metadata([("comment", MetadataValue::String("display name".into()))]);
    let mut id = field(&old_schema, "id");
    id.nullable = true;
    let updated = with_fields([
        StructField::nullable("country", DataType::STRING),
        full_name,
        id,
    ]);
    update_schema(&snapshot, engine.as_ref(), updated)?;

    let reloaded = Snapshot::builder_for(&table_path).build(engine.as_ref())?;
    let schema = reloaded.schema();
    let names: Vec<_> = schema.fields().map(StructField::name).collect();
    assert_eq!(names, ["country", "full_name", "id"]);
    assert_same_identity(&field(&old_schema, "name"), &field(&schema, "full_name"));
    assert_same_identity(&field(&old_schema, "id"), &field(&schema, "id"));
    assert!(field(&schema, "id").is_nullable());
    assert_eq!(
        field(&schema, "full_name").metadata().get("comment"),
        Some(&MetadataValue::String("display name".into()))
    );
    let country = field(&schema, "country");
    assert_eq!(country.column_mapping_id(), Some(old_max_id + 1));
    assert_eq!(max_column_id(&reloaded), old_max_id + 1);

    let scan = reloaded.scan_builder().build()?;
    let batches = read_scan(&scan, engine.clone())?;
    let batch = &batches[0];
    assert_eq!(batches.len(), 1);
    assert_eq!(batch.column(0).null_count(), 2);
    let full_names = batch.column(1).as_string::<i32>();
    assert_eq!(
        full_names.iter().collect::<Vec<_>>(),
        [Some("a"), Some("b")]
    );
    let ids = batch.column(2).as_primitive::<Int32Type>();
    assert_eq!(ids.values(), &[1, 2]);
    Ok(())
}

#[tokio::test]
async fn updated_schema_changes_nested_fields_preserving_identity() -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        nested_schema(),
        &[("delta.columnMapping.mode", "name")],
        None,
    )?;
    let old_schema = snapshot.schema();
    let old_address = field(&old_schema, "address");
    let DataType::Struct(old_address_type) = old_address.data_type() else {
        panic!("address is a struct");
    };
    let old_tags = field(&old_schema, "tags");
    let DataType::Array(old_tags_type) = old_tags.data_type() else {
        panic!("tags is an array");
    };
    let DataType::Struct(old_tag_type) = old_tags_type.element_type() else {
        panic!("tags holds structs");
    };

    // Drop `address.zip`, rename and relax `address.city`, add `address.street`, and rename and
    // relax `tags.element.weight`.
    let mut town = field(old_address_type, "city").with_name("town");
    town.nullable = true;
    let mut address = old_address.clone();
    address.data_type = with_fields([
        StructField::nullable("street", DataType::STRING),
        town.clone(),
    ])
    .into();
    let mut score = field(old_tag_type, "weight").with_name("score");
    score.nullable = true;
    let mut tags = old_tags.clone();
    tags.data_type = ArrayType::new(
        with_fields([field(old_tag_type, "label"), score.clone()]),
        true,
    )
    .into();
    let updated = with_fields([
        field(&old_schema, "id"),
        tags,
        address,
        field(&old_schema, "part"),
    ]);
    let committed = update_schema(&snapshot, engine.as_ref(), updated)?;

    let reloaded = Snapshot::builder_for(&table_path).build(engine.as_ref())?;
    assert_eq!(reloaded.schema(), committed.schema());
    let schema = reloaded.schema();
    let city_path = ["address".to_string(), "town".to_string()];
    let new_town = schema.field_at_path(&city_path);
    assert_same_identity(&field(old_address_type, "city"), new_town);
    assert!(new_town.is_nullable());
    let street = schema.field_at_path(&["address".to_string(), "street".to_string()]);
    assert_eq!(
        street.column_mapping_id(),
        Some(max_column_id(&snapshot) + 1)
    );
    let DataType::Array(new_tags) = field(&schema, "tags").data_type().clone() else {
        panic!("tags is an array");
    };
    let DataType::Struct(new_tag) = new_tags.element_type() else {
        panic!("tags holds structs");
    };
    assert_same_identity(&field(old_tag_type, "weight"), &field(new_tag, "score"));
    assert!(field(new_tag, "score").is_nullable());
    Ok(())
}

#[tokio::test]
async fn updated_schema_drop_keeps_max_column_id() -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        flat_schema(),
        &[("delta.columnMapping.mode", "name")],
        None,
    )?;
    let old_schema = snapshot.schema();
    let updated = with_fields([field(&old_schema, "id"), field(&old_schema, "name")]);
    let committed = update_schema(&snapshot, engine.as_ref(), updated)?;

    assert!(!committed.schema().contains("zip"));
    assert_eq!(max_column_id(&committed), max_column_id(&snapshot));
    for name in ["id", "name"] {
        assert_eq!(field(&committed.schema(), name), field(&old_schema, name));
    }
    Ok(())
}

#[tokio::test]
async fn updated_schema_rejects_reusing_a_dropped_column_mapping_id() -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        flat_schema(),
        &[("delta.columnMapping.mode", "name")],
        None,
    )?;
    let old_schema = snapshot.schema();
    let zip_id = field(&old_schema, "zip").column_mapping_id().unwrap();
    let dropped = update_schema(
        &snapshot,
        engine.as_ref(),
        with_fields([field(&old_schema, "id"), field(&old_schema, "name")]),
    )?;

    let reused = StructField::nullable("zip_again", DataType::STRING).add_metadata([
        ("delta.columnMapping.id", MetadataValue::Number(zip_id)),
        (
            "delta.columnMapping.physicalName",
            MetadataValue::String("col-reused".into()),
        ),
    ]);
    let mut fields: Vec<_> = dropped.schema().fields().cloned().collect();
    fields.push(reused);
    let result = update_schema(&dropped, engine.as_ref(), with_fields(fields));
    assert_result_error_with_message(result, "does not belong to an existing field");
    Ok(())
}

type SchemaEdit = fn(&StructType) -> Vec<StructField>;

#[rstest]
#[case::tighten_nullability(
    |s: &StructType| vec![field(s, "id"), StructField { nullable: false, ..field(s, "name") }, field(s, "zip")],
    "Cannot tighten the nullability of existing field 'name'"
)]
#[case::change_type(
    |s: &StructType| vec![
        field(s, "id"),
        StructField { data_type: DataType::INTEGER, ..field(s, "name") },
        field(s, "zip"),
    ],
    "Cannot change the type of existing field 'name' from string to integer"
)]
#[case::add_non_nullable(
    |s: &StructType| appended(s, StructField::not_null("new", DataType::STRING)),
    "Cannot add non-nullable field 'new'"
)]
#[case::add_with_default(
    |s: &StructType| appended(s, StructField::nullable("new", DataType::STRING)
            .add_metadata([("CURRENT_DEFAULT", MetadataValue::String("'x'".into()))])),
    "with a default value"
)]
#[case::add_with_generation_expression(
    |s: &StructType| appended(s, StructField::nullable("new", DataType::STRING).add_metadata([(
            "delta.generationExpression",
            MetadataValue::String("upper(name)".into()),
        )])),
    "delta.generationExpression"
)]
#[case::change_protocol_metadata(
    |s: &StructType| vec![
        field(s, "id"),
        field(s, "name").add_metadata([(
            "delta.generationExpression",
            MetadataValue::String("upper(zip)".into()),
        )]),
        field(s, "zip"),
    ],
    "Cannot change metadata 'delta.generationExpression' of existing field 'name'"
)]
#[case::change_physical_name(
    |s: &StructType| vec![
        field(s, "id"),
        field(s, "name").add_metadata([(
            "delta.columnMapping.physicalName",
            MetadataValue::String("col-other".into()),
        )]),
        field(s, "zip"),
    ],
    "inconsistent physical names"
)]
#[case::new_field_with_column_mapping_metadata(
    |s: &StructType| appended(
        s,
        StructField::nullable("new", DataType::STRING).add_metadata([
            ("delta.columnMapping.id", MetadataValue::Number(100)),
            ("delta.columnMapping.physicalName", MetadataValue::String("col-new".into())),
        ]),
    ),
    "does not belong to an existing field"
)]
#[case::new_field_with_physical_name_only(
    |s: &StructType| appended(
        s,
        StructField::nullable("new", DataType::STRING).add_metadata([(
            "delta.columnMapping.physicalName",
            physical_name(&field(s, "zip")).clone(),
        )]),
    ),
    "must carry both"
)]
#[case::incomplete_column_mapping_metadata(
    |s: &StructType| appended(s, StructField::nullable("new", DataType::STRING)
            .add_metadata([("delta.columnMapping.id", MetadataValue::Number(100))])),
    "must carry both"
)]
#[case::duplicate_column_mapping_id(
    |s: &StructType| appended(s, field(s, "zip").with_name("zip_copy")),
    "Duplicate field ID"
)]
#[case::move_into_struct(
    |s: &StructType| vec![
        field(s, "id"),
        field(s, "name"),
        StructField::nullable("wrapper", StructType::try_new([field(s, "zip")]).unwrap()),
    ],
    "Fields cannot move to a different parent"
)]
#[case::metadata_column(
    |s: &StructType| appended(s, StructField::create_metadata_column(
            "row_index",
            delta_kernel::schema::MetadataColumnSpec::RowIndex,
        )),
    "metadata column"
)]
#[tokio::test]
async fn updated_schema_rejects_unsupported_changes(
    #[case] edit: SchemaEdit,
    #[case] expected: &str,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        flat_schema(),
        &[("delta.columnMapping.mode", "name")],
        None,
    )?;
    let updated = with_fields(edit(&snapshot.schema()));
    assert_result_error_with_message(update_schema(&snapshot, engine.as_ref(), updated), expected);
    Ok(())
}

#[tokio::test]
async fn updated_schema_rejects_moving_a_nested_field_to_another_struct() -> delta_kernel::Result<()>
{
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        nested_schema(),
        &[("delta.columnMapping.mode", "name")],
        None,
    )?;
    // Move `address.zip` into the struct elements of `tags`.
    let old_schema = snapshot.schema();
    let mut address = field(&old_schema, "address");
    let DataType::Struct(address_type) = address.data_type().clone() else {
        panic!("address is a struct");
    };
    address.data_type = with_fields([field(&address_type, "city")]).into();
    let mut tags = field(&old_schema, "tags");
    let DataType::Array(tags_type) = tags.data_type().clone() else {
        panic!("tags is an array");
    };
    let DataType::Struct(tag_type) = tags_type.element_type() else {
        panic!("tags holds structs");
    };
    tags.data_type = ArrayType::new(
        with_fields(appended(tag_type, field(&address_type, "zip"))),
        true,
    )
    .into();
    let updated = with_fields([
        field(&old_schema, "id"),
        address,
        tags,
        field(&old_schema, "part"),
    ]);
    assert_result_error_with_message(
        update_schema(&snapshot, engine.as_ref(), updated),
        "moved from 'address.zip' to 'tags.element.zip'",
    );
    Ok(())
}

#[rstest]
#[case::drop(|s: &StructType| vec![field(s, "id"), field(s, "name")])]
#[case::rename(|s: &StructType| vec![field(s, "id"), field(s, "name"), field(s, "zip").with_name("postal")])]
#[case::add(|s: &StructType| appended(s, StructField::nullable("new", DataType::STRING)))]
#[tokio::test]
async fn updated_schema_requires_column_mapping(
    #[case] edit: SchemaEdit,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create(&table_path, engine.as_ref(), flat_schema(), &[], None)?;
    let updated = with_fields(edit(&snapshot.schema()));
    assert_result_error_with_message(
        update_schema(&snapshot, engine.as_ref(), updated),
        "Updating the table schema requires column mapping",
    );
    Ok(())
}

#[rstest]
#[case::drop(|s: &StructType| vec![field(s, "id"), field(s, "address"), field(s, "tags")])]
#[case::rename(|s: &StructType| vec![
    field(s, "id"),
    field(s, "address"),
    field(s, "tags"),
    field(s, "part").with_name("partition"),
])]
#[tokio::test]
async fn updated_schema_rejects_dropping_or_renaming_partition_columns(
    #[case] edit: SchemaEdit,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        nested_schema(),
        &[("delta.columnMapping.mode", "name")],
        Some(DataLayout::partitioned(["part"])),
    )?;
    let updated = with_fields(edit(&snapshot.schema()));
    assert_result_error_with_message(
        update_schema(&snapshot, engine.as_ref(), updated),
        "Cannot drop or rename partition column 'part'",
    );
    Ok(())
}

#[tokio::test]
async fn updated_schema_cannot_be_combined_with_incremental_schema_changes()
-> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        flat_schema(),
        &[("delta.columnMapping.mode", "name")],
        None,
    )?;
    let result = snapshot
        .clone()
        .update_table(UpdateTableOperation::AlterTable, "test-engine", false)
        .add_column(StructField::nullable("new", DataType::STRING))
        .with_updated_schema(snapshot.schema())
        .build(engine.as_ref(), Box::new(FileSystemCommitter::new()));
    assert_result_error_with_message(result, "with_updated_schema cannot be combined");
    Ok(())
}

#[tokio::test]
async fn updated_schema_last_call_wins() -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        flat_schema(),
        &[("delta.columnMapping.mode", "name")],
        None,
    )?;
    let old_schema = snapshot.schema();
    let dropped = with_fields([field(&old_schema, "id"), field(&old_schema, "name")]);
    let renamed = with_fields([
        field(&old_schema, "id"),
        field(&old_schema, "name"),
        field(&old_schema, "zip").with_name("postal"),
    ]);
    let committed = snapshot
        .clone()
        .update_table(UpdateTableOperation::AlterTable, "test-engine", false)
        .with_updated_schema(Arc::new(dropped))
        .with_updated_schema(Arc::new(renamed))
        .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?
        .commit(engine.as_ref())?
        .unwrap_post_commit_snapshot();
    let names: Vec<_> = committed
        .schema()
        .fields()
        .map(|field| field.name().clone())
        .collect();
    assert_eq!(names, ["id", "name", "postal"]);
    Ok(())
}

#[tokio::test]
async fn updated_schema_derives_required_table_features() -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        flat_schema(),
        &[("delta.columnMapping.mode", "name")],
        None,
    )?;
    assert!(
        !snapshot
            .table_configuration()
            .is_feature_supported(&TableFeature::TimestampWithoutTimezone)
    );
    let mut fields: Vec<_> = snapshot.schema().fields().cloned().collect();
    fields.push(StructField::nullable("ts", DataType::TIMESTAMP_NTZ));
    update_schema(&snapshot, engine.as_ref(), with_fields(fields))?;

    let reloaded = Snapshot::builder_for(&table_path).build(engine.as_ref())?;
    assert!(
        reloaded
            .table_configuration()
            .is_feature_supported(&TableFeature::TimestampWithoutTimezone)
    );
    assert!(reloaded.schema().contains("ts"));
    Ok(())
}

#[tokio::test]
async fn updated_schema_renames_a_field_to_the_name_of_a_dropped_field() -> delta_kernel::Result<()>
{
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        nested_schema(),
        &[("delta.columnMapping.mode", "name")],
        None,
    )?;
    let old_schema = snapshot.schema();
    // Drop `id` and rename the struct `address` to `id`.
    let updated = with_fields([
        field(&old_schema, "address").with_name("id"),
        field(&old_schema, "tags"),
        field(&old_schema, "part"),
    ]);
    let committed = update_schema(&snapshot, engine.as_ref(), updated)?;
    assert_same_identity(
        &field(&old_schema, "address"),
        &field(&committed.schema(), "id"),
    );
    Ok(())
}

#[tokio::test]
async fn updated_schema_rejects_renaming_a_collated_field() -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let collated = StructField::nullable("s", DataType::STRING).add_metadata([(
        "__COLLATIONS",
        MetadataValue::Other(serde_json::json!({ "s": "icu.UNICODE_CI" })),
    )]);
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        Arc::new(with_fields([collated])),
        &[("delta.columnMapping.mode", "name")],
        None,
    )?;
    let renamed = with_fields([field(&snapshot.schema(), "s").with_name("t")]);
    assert_result_error_with_message(
        update_schema(&snapshot, engine.as_ref(), renamed),
        "Renaming field 's' with a collation is not supported",
    );
    Ok(())
}

#[rstest]
#[case::drop(|s: &StructType| vec![field(s, "id"), field(s, "zip")])]
#[case::rename(|s: &StructType| vec![field(s, "id"), field(s, "name").with_name("n"), field(s, "zip")])]
#[tokio::test]
async fn updated_schema_rejects_dropping_or_renaming_stats_columns(
    #[case] edit: SchemaEdit,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        flat_schema(),
        &[
            ("delta.columnMapping.mode", "name"),
            ("delta.dataSkippingStatsColumns", "id,name"),
        ],
        None,
    )?;
    let updated = with_fields(edit(&snapshot.schema()));
    assert_result_error_with_message(
        update_schema(&snapshot, engine.as_ref(), updated),
        "Cannot drop or rename column 'name': it is listed in delta.dataSkippingStatsColumns",
    );
    Ok(())
}
