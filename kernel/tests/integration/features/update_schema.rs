//! Integration tests for replacing a table schema with `with_updated_schema`.

use std::collections::HashMap;
use std::sync::Arc;

use delta_kernel::arrow::array::{Array, AsArray as _, Int32Array, StringArray};
use delta_kernel::arrow::datatypes::{Float64Type, Int32Type, Int64Type, Schema as ArrowSchema};
use delta_kernel::arrow::json::ReaderBuilder;
use delta_kernel::arrow::record_batch::RecordBatch;
use delta_kernel::arrow::util::display::array_value_to_string;
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::engine::arrow_conversion::TryIntoArrow as _;
use delta_kernel::expressions::Scalar;
use delta_kernel::schema::{
    ArrayType, ColumnMetadataKey, DataType, MapType, MetadataValue, SchemaRef, StructField,
    StructType, schema_ref,
};
use delta_kernel::snapshot::Snapshot;
use delta_kernel::table_features::TableFeature;
use delta_kernel::transaction::UpdateTableOperation;
use delta_kernel::transaction::create_table::create_table;
use delta_kernel::transaction::data_layout::DataLayout;
use delta_kernel::{Engine, KernelError, SnapshotRef};
use rstest::rstest;
use test_utils::{
    assert_result_error_with_message, read_actions_from_commit, read_scan, test_table_setup,
    write_batch_to_table,
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

/// Asserts that updating `snapshot` to `schema` fails with `expected` and commits nothing.
fn assert_update_rejected(
    snapshot: &SnapshotRef,
    engine: &dyn Engine,
    schema: StructType,
    expected: &str,
) {
    assert_result_error_with_message(update_schema(snapshot, engine, schema), expected);
    assert_nothing_committed(snapshot, engine);
}

/// Asserts that the latest version of the table is still `snapshot`.
fn assert_nothing_committed(snapshot: &SnapshotRef, engine: &dyn Engine) {
    let reloaded = Snapshot::builder_for(snapshot.table_root())
        .build(engine)
        .unwrap();
    assert_eq!(reloaded.version(), snapshot.version());
    assert_eq!(reloaded.schema(), snapshot.schema());
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
    let batch = batch_from_json(
        &snapshot,
        r#"{"id": 1, "address": {"city": "Oslo", "zip": "0150"}, "tags": [{"label": "x", "weight": 3}], "part": "p"}"#,
    );
    let snapshot = write_batch_to_table(&snapshot, engine.as_ref(), batch, HashMap::new()).await?;
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

    // The renamed nested fields still read the values written under their old names.
    let batches = read_scan(&reloaded.scan_builder().build()?, engine.clone())?;
    let batch = &batches[0];
    let tags = batch.column(1).as_list::<i32>().value(0);
    let tag = tags.as_struct();
    assert_eq!(tag.column(0).as_string::<i32>().value(0), "x");
    assert_eq!(tag.column(1).as_primitive::<Int32Type>().value(0), 3);
    let address = batch.column(2).as_struct();
    assert!(address.column(0).is_null(0), "the added street is null");
    assert_eq!(address.column(1).as_string::<i32>().value(0), "Oslo");
    Ok(())
}

/// Decodes `rows` (one JSON object per line) into a batch with the table's Arrow schema.
fn batch_from_json(snapshot: &Snapshot, rows: &str) -> RecordBatch {
    let schema: ArrowSchema = snapshot.schema().as_ref().try_into_arrow().unwrap();
    ReaderBuilder::new(Arc::new(schema))
        .build(std::io::Cursor::new(rows))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
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
    let (before, after) = (
        snapshot.table_configuration(),
        committed.table_configuration(),
    );
    assert_eq!(
        after.metadata().configuration(),
        before.metadata().configuration()
    );
    assert_eq!(after.protocol(), before.protocol());
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
    assert_update_rejected(
        &dropped,
        engine.as_ref(),
        with_fields(fields),
        "does not belong to an existing field",
    );
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
#[case::add_struct_with_non_nullable_child(
    |s: &StructType| appended(s, StructField::nullable(
        "new",
        StructType::try_new([StructField::not_null("c", DataType::INTEGER)]).unwrap(),
    )),
    "Cannot add non-nullable field 'c' under 'new'"
)]
#[case::add_array_of_struct_with_non_nullable_child(
    |s: &StructType| appended(s, StructField::nullable(
        "new",
        ArrayType::new(
            StructType::try_new([StructField::not_null("c", DataType::INTEGER)]).unwrap(),
            true,
        ),
    )),
    "Cannot add non-nullable field 'c' under 'new'"
)]
#[case::add_field_with_newline_in_name(
    |s: &StructType| appended(s, StructField::nullable("bad\nname", DataType::STRING)),
    "contains a newline character"
)]
#[case::add_with_default(
    |s: &StructType| appended(s, StructField::nullable("new", DataType::STRING)
            .add_metadata([("CURRENT_DEFAULT", MetadataValue::String("'x'".into()))])),
    "the table does not support the allowColumnDefaults table feature"
)]
#[case::add_with_generation_expression(
    |s: &StructType| appended(s, StructField::nullable("new", DataType::STRING).add_metadata([(
            "delta.generationExpression",
            MetadataValue::String("upper(name)".into()),
        )])),
    "delta.generationExpression"
)]
#[case::change_existing_default(
    |s: &StructType| vec![
        field(s, "id"),
        field(s, "name").add_metadata([("CURRENT_DEFAULT", MetadataValue::String("'x'".into()))]),
        field(s, "zip"),
    ],
    "the table does not support the allowColumnDefaults table feature"
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
#[case::empty_schema(|_: &StructType| vec![], "must contain at least one column")]
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
    assert_update_rejected(&snapshot, engine.as_ref(), updated, expected);
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
    assert_update_rejected(
        &snapshot,
        engine.as_ref(),
        updated,
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
    assert_update_rejected(
        &snapshot,
        engine.as_ref(),
        updated,
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
#[case::case_only_rename(|s: &StructType| vec![
    field(s, "id"),
    field(s, "address"),
    field(s, "tags"),
    field(s, "part").with_name("PART"),
])]
// Dropping the partition column and adding a new one with the same name: the new column gets a
// new physical name, so the existing partition values would be orphaned. Java accepts this.
#[case::drop_and_readd_with_the_same_name(|s: &StructType| vec![
    field(s, "id"),
    field(s, "address"),
    field(s, "tags"),
    StructField::nullable("part", field(s, "part").data_type().clone()),
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
    assert_update_rejected(
        &snapshot,
        engine.as_ref(),
        updated,
        "Cannot drop or rename partition column 'part'",
    );
    Ok(())
}

#[tokio::test]
async fn updated_schema_rejects_change_data_feed_reserved_names() -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        flat_schema(),
        &[
            ("delta.columnMapping.mode", "name"),
            ("delta.enableChangeDataFeed", "true"),
        ],
        None,
    )?;
    let updated = with_fields(appended(
        &snapshot.schema(),
        StructField::nullable("_change_type", DataType::STRING),
    ));
    assert_update_rejected(
        &snapshot,
        engine.as_ref(),
        updated,
        "is reserved for Change Data Feed",
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn updated_schema_adds_variant_and_file_columns(
    #[values(DataType::unshredded_variant(), DataType::file_type())] data_type: DataType,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        flat_schema(),
        &[("delta.columnMapping.mode", "name")],
        None,
    )?;
    // The non-nullable fields inside variant and file values are not table columns.
    let added = StructField::nullable("added", data_type.clone());
    update_schema(
        &snapshot,
        engine.as_ref(),
        with_fields(appended(&snapshot.schema(), added)),
    )?;

    let reloaded = Snapshot::builder_for(&table_path).build(engine.as_ref())?;
    assert_eq!(field(&reloaded.schema(), "added").data_type(), &data_type);
    Ok(())
}

#[tokio::test]
async fn updated_schema_changes_only_a_comment() -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        flat_schema(),
        &[("delta.columnMapping.mode", "name")],
        None,
    )?;
    let schema = snapshot.schema();
    let commented = field(&schema, "name").add_metadata([("comment", "the customer name")]);
    let updated = with_fields([field(&schema, "id"), commented, field(&schema, "zip")]);

    let committed = update_schema(&snapshot, engine.as_ref(), updated)?;

    assert_eq!(committed.version(), snapshot.version() + 1);
    let reloaded = Snapshot::builder_for(&table_path).build(engine.as_ref())?;
    let name = field(&reloaded.schema(), "name");
    assert_eq!(
        name.metadata().get("comment"),
        Some(&MetadataValue::String("the customer name".to_string()))
    );
    assert_eq!(physical_name(&name), physical_name(&field(&schema, "name")));
    assert_eq!(field(&reloaded.schema(), "id"), field(&schema, "id"));
    assert_eq!(field(&reloaded.schema(), "zip"), field(&schema, "zip"));
    Ok(())
}

#[tokio::test]
async fn updated_schema_renames_and_reorders_fields_in_a_map_value_struct()
-> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let value = StructType::try_new([
        StructField::nullable("a", DataType::INTEGER),
        StructField::nullable("b", DataType::STRING),
    ])?;
    let schema = Arc::new(StructType::try_new([
        StructField::nullable("id", DataType::INTEGER),
        StructField::nullable("m", MapType::new(DataType::STRING, value, true)),
    ])?);
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        schema,
        &[("delta.columnMapping.mode", "name")],
        None,
    )?;
    let map_value_fields = |schema: &StructType| -> Vec<StructField> {
        let DataType::Map(map) = field(schema, "m").data_type().clone() else {
            panic!("m is a map");
        };
        let DataType::Struct(value) = map.value_type().clone() else {
            panic!("the map value is a struct");
        };
        value.fields().cloned().collect()
    };
    let current = map_value_fields(&snapshot.schema());
    // Rename `a` to `alpha` and move it after `b`.
    let renamed_value = StructType::try_new([current[1].clone(), current[0].with_name("alpha")])?;
    let mut m = field(&snapshot.schema(), "m");
    m.data_type = MapType::new(DataType::STRING, renamed_value, true).into();
    let updated = with_fields([field(&snapshot.schema(), "id"), m]);

    update_schema(&snapshot, engine.as_ref(), updated)?;

    let reloaded = Snapshot::builder_for(&table_path).build(engine.as_ref())?;
    let fields = map_value_fields(&reloaded.schema());
    let names: Vec<_> = fields.iter().map(|field| field.name().as_str()).collect();
    assert_eq!(names, ["b", "alpha"]);
    assert_eq!(fields[0], current[1]);
    assert_eq!(
        fields[1].column_mapping_id(),
        current[0].column_mapping_id()
    );
    assert_eq!(physical_name(&fields[1]), physical_name(&current[0]));
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
    assert_nothing_committed(&snapshot, engine.as_ref());
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
    assert_update_rejected(
        &snapshot,
        engine.as_ref(),
        renamed,
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
    assert_update_rejected(
        &snapshot,
        engine.as_ref(),
        updated,
        "Cannot drop or rename column 'name': it is listed in delta.dataSkippingStatsColumns",
    );
    Ok(())
}

// === Type widening ===

const TYPE_WIDENING: (&str, &str) = ("delta.enableTypeWidening", "true");

fn type_changes(field: &StructField) -> Option<&MetadataValue> {
    field.get_config_value(&ColumnMetadataKey::TypeChanges)
}

fn widening_schema() -> SchemaRef {
    let address = StructType::try_new([StructField::nullable("zip", DataType::SHORT)]).unwrap();
    Arc::new(
        StructType::try_new([
            StructField::nullable("i", DataType::INTEGER),
            StructField::nullable("s", DataType::STRING),
            StructField::nullable("scores", ArrayType::new(DataType::INTEGER, true)),
            StructField::nullable("address", address),
        ])
        .unwrap(),
    )
}

#[tokio::test]
async fn updated_schema_widens_types_and_records_type_changes() -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        schema_ref! { nullable "i": INTEGER, nullable "s": STRING },
        &[("delta.columnMapping.mode", "name"), TYPE_WIDENING],
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
        ],
    )
    .map_err(KernelError::from)?;
    let snapshot = write_batch_to_table(&snapshot, engine.as_ref(), batch, HashMap::new()).await?;
    let old_schema = snapshot.schema();
    let widened = with_fields([
        StructField {
            data_type: DataType::LONG,
            ..field(&old_schema, "i")
        },
        field(&old_schema, "s"),
    ]);
    update_schema(&snapshot, engine.as_ref(), widened)?;

    let reloaded = Snapshot::builder_for(&table_path).build(engine.as_ref())?;
    let i = field(&reloaded.schema(), "i");
    assert_eq!(i.data_type(), &DataType::LONG);
    assert_same_identity(&field(&old_schema, "i"), &i);
    assert_eq!(
        type_changes(&i),
        Some(&MetadataValue::Other(serde_json::json!([
            { "fromType": "integer", "toType": "long" }
        ])))
    );
    let batches = read_scan(&reloaded.scan_builder().build()?, engine.clone())?;
    let ids = batches[0].column(0).as_primitive::<Int64Type>();
    assert_eq!(ids.values(), &[1, 2]);
    Ok(())
}

#[tokio::test]
async fn updated_schema_records_nested_and_container_type_changes() -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        widening_schema(),
        &[("delta.columnMapping.mode", "name"), TYPE_WIDENING],
        None,
    )?;
    let batch = batch_from_json(
        &snapshot,
        r#"{"i": 1, "s": "a", "scores": [1, 2], "address": {"zip": 7}}"#,
    );
    let snapshot = write_batch_to_table(&snapshot, engine.as_ref(), batch, HashMap::new()).await?;
    let old_schema = snapshot.schema();
    let mut address = field(&old_schema, "address");
    let DataType::Struct(address_type) = address.data_type().clone() else {
        panic!("address is a struct");
    };
    address.data_type = with_fields([StructField {
        data_type: DataType::INTEGER,
        ..field(&address_type, "zip")
    }])
    .into();
    let updated = with_fields([
        field(&old_schema, "i"),
        field(&old_schema, "s"),
        StructField {
            data_type: ArrayType::new(DataType::DOUBLE, true).into(),
            ..field(&old_schema, "scores")
        },
        address,
    ]);
    let committed = update_schema(&snapshot, engine.as_ref(), updated)?;

    let schema = committed.schema();
    assert_eq!(
        type_changes(&field(&schema, "scores")),
        Some(&MetadataValue::Other(serde_json::json!([
            { "fromType": "integer", "toType": "double", "fieldPath": "element" }
        ])))
    );
    let zip = schema.field_at_path(&["address".to_string(), "zip".to_string()]);
    assert_eq!(
        type_changes(zip),
        Some(&MetadataValue::Other(serde_json::json!([
            { "fromType": "short", "toType": "integer" }
        ])))
    );
    assert_eq!(type_changes(&field(&schema, "address")), None);
    assert_eq!(type_changes(&field(&schema, "i")), None);

    // Values written before the widening read back as the wider types.
    let batches = read_scan(&committed.scan_builder().build()?, engine.clone())?;
    let scores = batches[0].column(2).as_list::<i32>().value(0);
    assert_eq!(scores.as_primitive::<Float64Type>().values(), &[1.0, 2.0]);
    let address = batches[0].column(3).as_struct();
    assert_eq!(address.column(0).as_primitive::<Int32Type>().value(0), 7);
    Ok(())
}

#[tokio::test]
async fn updated_schema_widens_map_keys_and_values() -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        Arc::new(StructType::try_new([
            StructField::nullable("id", DataType::INTEGER),
            StructField::nullable("m", MapType::new(DataType::INTEGER, DataType::FLOAT, true)),
        ])?),
        &[("delta.columnMapping.mode", "name"), TYPE_WIDENING],
        None,
    )?;
    let batch = batch_from_json(&snapshot, r#"{"id": 1, "m": {"3": 1.5}}"#);
    let snapshot = write_batch_to_table(&snapshot, engine.as_ref(), batch, HashMap::new()).await?;
    let old_schema = snapshot.schema();
    let updated = with_fields([
        field(&old_schema, "id"),
        StructField {
            data_type: MapType::new(DataType::LONG, DataType::DOUBLE, true).into(),
            ..field(&old_schema, "m")
        },
    ]);
    update_schema(&snapshot, engine.as_ref(), updated)?;

    let reloaded = Snapshot::builder_for(&table_path).build(engine.as_ref())?;
    assert_eq!(
        type_changes(&field(&reloaded.schema(), "m")),
        Some(&MetadataValue::Other(serde_json::json!([
            { "fromType": "integer", "toType": "long", "fieldPath": "key" },
            { "fromType": "float", "toType": "double", "fieldPath": "value" }
        ])))
    );
    let batches = read_scan(&reloaded.scan_builder().build()?, engine.clone())?;
    let entries = batches[0].column(1).as_map().value(0);
    assert_eq!(entries.column(0).as_primitive::<Int64Type>().values(), &[3]);
    assert_eq!(
        entries.column(1).as_primitive::<Float64Type>().values(),
        &[1.5]
    );
    Ok(())
}

#[tokio::test]
async fn updated_schema_keeps_type_change_history_omitted_by_the_caller() -> delta_kernel::Result<()>
{
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        widening_schema(),
        &[("delta.columnMapping.mode", "name"), TYPE_WIDENING],
        None,
    )?;
    let widen_i = |schema: &StructType, data_type: DataType, keep_history: bool| {
        let mut i = StructField {
            data_type,
            ..field(schema, "i")
        };
        if !keep_history {
            i.metadata.remove(ColumnMetadataKey::TypeChanges.as_ref());
        }
        let fields = schema.fields().map(|f| {
            if f.name() == "i" {
                i.clone()
            } else {
                f.clone()
            }
        });
        with_fields(fields)
    };
    let first = update_schema(
        &snapshot,
        engine.as_ref(),
        widen_i(&snapshot.schema(), DataType::LONG, true),
    )?;
    let decimal = DataType::decimal(22, 2)?;
    let second = update_schema(
        &first,
        engine.as_ref(),
        widen_i(&first.schema(), decimal, false),
    )?;
    assert_eq!(
        type_changes(&field(&second.schema(), "i")),
        Some(&MetadataValue::Other(serde_json::json!([
            { "fromType": "integer", "toType": "long" },
            { "fromType": "long", "toType": "decimal(22,2)" }
        ])))
    );
    // An update that keeps the type and omits the history keeps the history too.
    let i = field(&second.schema(), "i");
    let third = update_schema(
        &second,
        engine.as_ref(),
        widen_i(&second.schema(), i.data_type().clone(), false),
    )?;
    assert_eq!(field(&third.schema(), "i"), i);
    Ok(())
}

#[rstest]
#[case::not_enabled(
    false,
    |s: &StructType| retype(s, "i", DataType::LONG),
    "Cannot change the type of existing field 'i' from integer to long: type widening is not \
     enabled on the table"
)]
#[case::narrowing(
    true,
    |s: &StructType| retype(s, "i", DataType::SHORT),
    "Cannot change the type of existing field 'i' from integer to short"
)]
#[case::unsupported_element_change(
    true,
    |s: &StructType| retype(s, "scores", ArrayType::new(DataType::STRING, true).into()),
    "Cannot change the type of existing field 'scores' at 'element' from integer to string"
)]
#[case::container_change(
    true,
    |s: &StructType| retype(s, "scores", DataType::INTEGER),
    "Cannot change the type of existing field 'scores' from array<integer> to integer"
)]
#[case::modified_history(
    true,
    |s: &StructType| {
        s.fields()
            .map(|f| match f.name().as_str() {
                "i" => f.clone().add_metadata([(
                    "delta.typeChanges",
                    MetadataValue::Other(serde_json::json!([
                        { "fromType": "short", "toType": "integer" }
                    ])),
                )]),
                _ => f.clone(),
            })
            .collect()
    },
    "Cannot change metadata 'delta.typeChanges' of existing field 'i'"
)]
#[case::history_on_new_field(
    true,
    |s: &StructType| appended(
        s,
        StructField::nullable("new", DataType::LONG).add_metadata([(
            "delta.typeChanges",
            MetadataValue::Other(serde_json::json!([
                { "fromType": "integer", "toType": "long" }
            ])),
        )]),
    ),
    "Cannot add field 'new' under 'new' with `delta.typeChanges` metadata"
)]
#[tokio::test]
async fn updated_schema_rejects_unsupported_type_changes(
    #[case] type_widening: bool,
    #[case] edit: SchemaEdit,
    #[case] expected: &str,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let mut properties = vec![("delta.columnMapping.mode", "name")];
    if type_widening {
        properties.push(TYPE_WIDENING);
    }
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        widening_schema(),
        &properties,
        None,
    )?;
    let updated = with_fields(edit(&snapshot.schema()));
    assert_update_rejected(&snapshot, engine.as_ref(), updated, expected);
    Ok(())
}

/// Returns the fields of `schema` with the type of field `name` replaced by `data_type`.
fn retype(schema: &StructType, name: &str, data_type: DataType) -> Vec<StructField> {
    schema
        .fields()
        .map(|f| match f.name() == name {
            true => StructField {
                data_type: data_type.clone(),
                ..f.clone()
            },
            false => f.clone(),
        })
        .collect()
}

#[rstest]
#[case::integer_to_long(DataType::INTEGER, Scalar::Integer(7), DataType::LONG, Ok("7"))]
#[case::integer_to_double(DataType::INTEGER, Scalar::Integer(7), DataType::DOUBLE, Ok("7.0"))]
#[case::integer_to_decimal_same_scale(
    DataType::INTEGER,
    Scalar::Integer(7),
    DataType::decimal(12, 0).unwrap(),
    Ok("7")
)]
#[case::decimal_precision(
    DataType::decimal(5, 2).unwrap(),
    Scalar::decimal(150, 5, 2).unwrap(),
    DataType::decimal(10, 2).unwrap(),
    Ok("1.50")
)]
#[case::integer_to_decimal(
    DataType::INTEGER,
    Scalar::Integer(7),
    DataType::decimal(12, 2).unwrap(),
    Err("Cannot change the type of partition column 'p' from integer to decimal(12,2)")
)]
#[case::decimal_scale(
    DataType::decimal(5, 2).unwrap(),
    Scalar::decimal(150, 5, 2).unwrap(),
    DataType::decimal(8, 4).unwrap(),
    Err("Cannot change the type of partition column 'p' from decimal(5,2) to decimal(8,4)")
)]
#[case::date_to_timestamp_ntz(
    DataType::DATE,
    Scalar::Date(18000),
    DataType::TIMESTAMP_NTZ,
    Err("Cannot change the type of partition column 'p' from date to timestamp_ntz")
)]
#[tokio::test]
async fn updated_schema_widens_partition_columns_only_when_values_stay_readable(
    #[case] from: DataType,
    #[case] value: Scalar,
    #[case] to: DataType,
    // The partition value read back after the widening, or the rejection message.
    #[case] expected: Result<&str, &str>,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        Arc::new(with_fields([
            StructField::nullable("i", DataType::INTEGER),
            StructField::nullable("p", from),
        ])),
        &[("delta.columnMapping.mode", "name"), TYPE_WIDENING],
        Some(DataLayout::partitioned(["p"])),
    )?;
    let data_schema = schema_ref! { nullable "i": INTEGER };
    let batch = RecordBatch::try_new(
        Arc::new(
            data_schema
                .as_ref()
                .try_into_arrow()
                .map_err(KernelError::from)?,
        ),
        vec![Arc::new(Int32Array::from(vec![1]))],
    )
    .map_err(KernelError::from)?;
    let partition_values = HashMap::from([("p".to_string(), value)]);
    let snapshot =
        write_batch_to_table(&snapshot, engine.as_ref(), batch, partition_values).await?;
    let updated = with_fields(retype(&snapshot.schema(), "p", to.clone()));
    let result = update_schema(&snapshot, engine.as_ref(), updated);
    match expected {
        Err(expected) => {
            assert_result_error_with_message(result, expected);
            assert_nothing_committed(&snapshot, engine.as_ref());
        }
        Ok(expected_value) => {
            let committed = result?;
            let batches = read_scan(&committed.scan_builder().build()?, engine.clone())?;
            let partition = batches[0].column(1);
            assert_eq!(partition.len(), 1);
            assert_eq!(
                partition.data_type(),
                &(&to).try_into_arrow().map_err(KernelError::from)?
            );
            assert_eq!(
                array_value_to_string(partition, 0).map_err(KernelError::from)?,
                expected_value
            );
        }
    }
    Ok(())
}

#[rstest]
#[case::integer_to_long(DataType::LONG, None)]
#[case::integer_to_double(
    DataType::DOUBLE,
    Some("only allow type changes that Iceberg V2 supports")
)]
#[case::integer_to_decimal(
    DataType::decimal(12, 0).unwrap(),
    Some("only allow type changes that Iceberg V2 supports")
)]
#[tokio::test]
async fn updated_schema_restricts_widening_while_iceberg_compat_is_supported(
    #[case] to: DataType,
    #[case] expected_error: Option<&str>,
    #[values("icebergCompatV2", "icebergCompatV3")] compat: &str,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let signal = format!("delta.feature.{compat}");
    let mut properties = vec![
        ("delta.columnMapping.mode", "name"),
        TYPE_WIDENING,
        (signal.as_str(), "supported"),
    ];
    // Kernel requires row tracking whenever icebergCompatV3 is supported.
    if compat == "icebergCompatV3" {
        properties.push(("delta.enableRowTracking", "true"));
    }
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        widening_schema(),
        &properties,
        None,
    )?;
    // The protocol restricts widening while icebergCompatV1 or V2 is supported; the V3 RFC only
    // while icebergCompatV3 is enabled.
    let expected_error = expected_error.filter(|_| compat != "icebergCompatV3");
    let updated = with_fields(retype(&snapshot.schema(), "i", to));
    let result = update_schema(&snapshot, engine.as_ref(), updated);
    match expected_error {
        Some(expected) => {
            assert_result_error_with_message(result, expected);
            assert_nothing_committed(&snapshot, engine.as_ref());
        }
        None => {
            result?;
        }
    }
    Ok(())
}

fn container_schema() -> SchemaRef {
    let tag = StructType::try_new([StructField::nullable("label", DataType::STRING)]).unwrap();
    Arc::new(
        StructType::try_new([
            StructField::nullable("tags", ArrayType::new(tag, true)),
            StructField::nullable("scores", ArrayType::new(DataType::INTEGER, true)),
            StructField::nullable(
                "attrs",
                MapType::new(DataType::STRING, DataType::INTEGER, true),
            ),
        ])
        .unwrap(),
    )
}

#[rstest]
#[case::array_contains_null_tightened(
    |s: &StructType| {
        let DataType::Array(tags) = field(s, "tags").data_type().clone() else { unreachable!() };
        retype(s, "tags", ArrayType::new(tags.element_type().clone(), false).into())
    },
    "Cannot tighten the nullability of existing field 'tags'"
)]
#[case::map_value_contains_null_tightened(
    |s: &StructType| retype(
        s,
        "attrs",
        MapType::new(DataType::STRING, DataType::INTEGER, false).into(),
    ),
    "Cannot tighten the nullability of existing field 'attrs'"
)]
#[case::array_to_primitive(
    |s: &StructType| retype(s, "tags", DataType::STRING),
    "Cannot change the type of existing field 'tags'"
)]
#[case::array_to_map(
    |s: &StructType| retype(
        s,
        "scores",
        MapType::new(DataType::INTEGER, DataType::INTEGER, true).into(),
    ),
    "Cannot change the type of existing field 'scores'"
)]
#[case::array_element_type(
    |s: &StructType| retype(s, "scores", ArrayType::new(DataType::STRING, true).into()),
    "Cannot change the type of existing field 'scores'"
)]
#[case::map_value_type(
    |s: &StructType| retype(
        s,
        "attrs",
        MapType::new(DataType::STRING, DataType::STRING, true).into(),
    ),
    "Cannot change the type of existing field 'attrs'"
)]
#[tokio::test]
async fn updated_schema_rejects_container_and_type_changes(
    #[case] edit: SchemaEdit,
    #[case] expected: &str,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        container_schema(),
        &[
            ("delta.columnMapping.mode", "name"),
            ("delta.enableTypeWidening", "true"),
        ],
        None,
    )?;
    let updated = with_fields(edit(&snapshot.schema()));
    assert_update_rejected(&snapshot, engine.as_ref(), updated, expected);
    Ok(())
}

#[tokio::test]
async fn updated_schema_keeps_type_change_history_through_a_rename() -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let history = MetadataValue::Other(serde_json::json!([
        { "fromType": "integer", "toType": "long" }
    ]));
    let widened = StructField::nullable("i", DataType::LONG)
        .add_metadata([("delta.typeChanges", history.clone())]);
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        Arc::new(with_fields([
            widened,
            StructField::nullable("s", DataType::STRING),
        ])),
        &[
            ("delta.columnMapping.mode", "name"),
            ("delta.enableTypeWidening", "true"),
        ],
        None,
    )?;
    let mut i = field(&snapshot.schema(), "i");
    i.metadata.remove("delta.typeChanges");
    let renamed = with_fields([i, field(&snapshot.schema(), "s").with_name("t")]);
    let committed = update_schema(&snapshot, engine.as_ref(), renamed)?;
    assert_eq!(
        field(&committed.schema(), "i").get_config_value(&ColumnMetadataKey::TypeChanges),
        Some(&history)
    );

    let altered = MetadataValue::Other(serde_json::json!([
        { "fromType": "short", "toType": "long" }
    ]));
    let committed_schema = committed.schema();
    let fields = committed_schema.fields().map(|f| match f.name().as_str() {
        "i" => f
            .clone()
            .add_metadata([("delta.typeChanges", altered.clone())]),
        _ => f.clone(),
    });
    assert_update_rejected(
        &committed,
        engine.as_ref(),
        with_fields(fields),
        "Cannot change metadata 'delta.typeChanges' of existing field 'i'",
    );
    Ok(())
}

// === IcebergCompat ===

/// Table properties that enable `compat` (`icebergCompatV2` or `icebergCompatV3`).
fn iceberg_compat_properties(compat: &str) -> Vec<(&'static str, &'static str)> {
    let mut properties = vec![("delta.columnMapping.mode", "name")];
    match compat {
        "icebergCompatV2" => properties.push(("delta.enableIcebergCompatV2", "true")),
        "icebergCompatV3" => properties.extend([
            ("delta.enableIcebergCompatV3", "true"),
            ("delta.enableRowTracking", "true"),
        ]),
        _ => panic!("unexpected IcebergCompat version {compat}"),
    }
    properties
}

#[rstest]
#[case::inside_the_iceberg_subset(DataType::LONG, None)]
#[case::outside_the_iceberg_subset(
    DataType::DOUBLE,
    Some("only allow type changes that Iceberg V2 supports")
)]
#[tokio::test]
async fn updated_schema_restricts_widening_on_iceberg_compat_enabled_tables(
    #[values("icebergCompatV2", "icebergCompatV3")] compat: &str,
    #[case] element_type: DataType,
    #[case] expected_error: Option<&str>,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let mut properties = iceberg_compat_properties(compat);
    properties.push(TYPE_WIDENING);
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        container_schema(),
        &properties,
        None,
    )?;
    let updated = with_fields(retype(
        &snapshot.schema(),
        "scores",
        ArrayType::new(element_type, true).into(),
    ));
    match expected_error {
        Some(expected) => assert_update_rejected(&snapshot, engine.as_ref(), updated, expected),
        None => {
            let committed = update_schema(&snapshot, engine.as_ref(), updated)?;
            assert_eq!(committed.version(), snapshot.version() + 1);
        }
    }
    Ok(())
}

fn create_iceberg_compat_table(
    table_path: &str,
    engine: &dyn Engine,
    compat: &str,
) -> delta_kernel::Result<SnapshotRef> {
    create(
        table_path,
        engine,
        container_schema(),
        &iceberg_compat_properties(compat),
        None,
    )
}

fn nested_ids(field: &StructField) -> Option<&serde_json::Map<String, serde_json::Value>> {
    match field.get_config_value(&ColumnMetadataKey::ColumnMappingNestedIds)? {
        MetadataValue::Other(serde_json::Value::Object(ids)) => Some(ids),
        other => panic!("unexpected nested ids {other:?}"),
    }
}

/// Returns the nested ids of `field` as (path relative to its physical name, id) pairs.
fn relative_nested_ids(field: &StructField) -> Vec<(String, i64)> {
    let MetadataValue::String(physical_name) = physical_name(field) else {
        panic!("physical name of '{}' is not a string", field.name());
    };
    let mut ids: Vec<_> = nested_ids(field)
        .into_iter()
        .flatten()
        .map(|(path, id)| {
            let path = path.strip_prefix(physical_name.as_str()).unwrap();
            (path.to_string(), id.as_i64().unwrap())
        })
        .collect();
    ids.sort();
    ids
}

/// Returns every column-mapping id and nested id in `schema`.
fn all_column_ids(schema: &StructType) -> Vec<i64> {
    fn collect_in_type(data_type: &DataType, ids: &mut Vec<i64>) {
        match data_type {
            DataType::Struct(inner) => ids.extend(all_column_ids(inner)),
            DataType::Array(array) => collect_in_type(array.element_type(), ids),
            DataType::Map(map) => {
                collect_in_type(map.key_type(), ids);
                collect_in_type(map.value_type(), ids);
            }
            _ => {}
        }
    }
    let mut ids = Vec::new();
    for field in schema.fields() {
        ids.extend(field.column_mapping_id());
        ids.extend(
            nested_ids(field)
                .into_iter()
                .flatten()
                .map(|(_, id)| id.as_i64().unwrap()),
        );
        collect_in_type(field.data_type(), &mut ids);
    }
    ids
}

#[rstest]
#[tokio::test]
async fn updated_schema_assigns_nested_ids_on_iceberg_compat_tables(
    #[values("icebergCompatV2", "icebergCompatV3")] compat: &str,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create_iceberg_compat_table(&table_path, engine.as_ref(), compat)?;
    let old_schema = snapshot.schema();
    let old_max_id = max_column_id(&snapshot);
    assert_eq!(
        all_column_ids(&old_schema).into_iter().max(),
        Some(old_max_id)
    );

    // Add a field to the struct in the existing `tags` array.
    let mut tags = field(&old_schema, "tags");
    let DataType::Array(tags_type) = tags.data_type().clone() else {
        unreachable!()
    };
    let DataType::Struct(tag) = tags_type.element_type().clone() else {
        unreachable!()
    };
    let tag = with_fields(appended(
        &tag,
        StructField::nullable("weight", DataType::INTEGER),
    ));
    tags.data_type = ArrayType::new(tag, true).into();
    let address = with_fields([
        StructField::nullable("city", DataType::STRING),
        StructField::nullable("codes", ArrayType::new(DataType::STRING, true)),
    ]);
    let props = MapType::new(
        DataType::STRING,
        ArrayType::new(DataType::INTEGER, true),
        true,
    );
    let updated = with_fields([
        tags,
        field(&old_schema, "scores"),
        field(&old_schema, "attrs"),
        StructField::nullable("name", DataType::STRING),
        StructField::nullable("address", address),
        StructField::nullable("events", ArrayType::new(DataType::LONG, true)),
        StructField::nullable("props", props),
    ]);
    update_schema(&snapshot, engine.as_ref(), updated)?;

    let reloaded = Snapshot::builder_for(&table_path).build(engine.as_ref())?;
    let schema = reloaded.schema();
    for name in ["tags", "scores", "attrs"] {
        assert_eq!(
            nested_ids(&field(&schema, name)),
            nested_ids(&field(&old_schema, name)),
            "nested ids of existing field '{name}'"
        );
    }
    let new_slot_ids = |field: &StructField| {
        let ids = relative_nested_ids(field);
        assert!(ids.iter().all(|(_, id)| *id > old_max_id), "{ids:?}");
        ids.into_iter().map(|(path, _)| path).collect::<Vec<_>>()
    };
    assert_eq!(new_slot_ids(&field(&schema, "events")), [".element"]);
    assert_eq!(
        new_slot_ids(&field(&schema, "props")),
        [".key", ".value", ".value.element"]
    );
    let DataType::Struct(address) = field(&schema, "address").data_type().clone() else {
        unreachable!()
    };
    assert_eq!(nested_ids(&field(&schema, "address")), None);
    assert_eq!(new_slot_ids(&field(&address, "codes")), [".element"]);
    assert_eq!(nested_ids(&field(&schema, "name")), None);

    let ids = all_column_ids(&schema);
    let unique: std::collections::HashSet<_> = ids.iter().collect();
    assert_eq!(unique.len(), ids.len(), "duplicate ids in {ids:?}");
    assert_eq!(ids.into_iter().max(), Some(max_column_id(&reloaded)));
    assert!(
        reloaded
            .table_configuration()
            .is_feature_enabled(&compat.parse::<TableFeature>().unwrap())
    );
    if compat == "icebergCompatV3" {
        // Row tracking stays enabled with the same materialized column names.
        let (before, after) = (snapshot.table_properties(), reloaded.table_properties());
        assert_eq!(after.enable_row_tracking, Some(true));
        assert!(before.materialized_row_id_column_name.is_some());
        assert_eq!(
            after.materialized_row_id_column_name,
            before.materialized_row_id_column_name
        );
        assert_eq!(
            after.materialized_row_commit_version_column_name,
            before.materialized_row_commit_version_column_name
        );
    }
    // The reloaded table accepts further writes.
    reloaded
        .update_table(UpdateTableOperation::Write, "test-engine", true)
        .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?;
    Ok(())
}

#[rstest]
#[tokio::test]
async fn updated_schema_keeps_nested_ids_through_rename_drop_and_reorder(
    #[values("icebergCompatV2", "icebergCompatV3")] compat: &str,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create_iceberg_compat_table(&table_path, engine.as_ref(), compat)?;
    let old_schema = snapshot.schema();
    let updated = with_fields([
        field(&old_schema, "attrs").with_name("attributes"),
        field(&old_schema, "tags"),
    ]);
    let committed = update_schema(&snapshot, engine.as_ref(), updated)?;

    let schema = committed.schema();
    assert_eq!(
        nested_ids(&field(&schema, "attributes")),
        nested_ids(&field(&old_schema, "attrs"))
    );
    assert_eq!(
        nested_ids(&field(&schema, "tags")),
        nested_ids(&field(&old_schema, "tags"))
    );
    assert_eq!(max_column_id(&committed), max_column_id(&snapshot));
    let reloaded = Snapshot::builder_for(&table_path).build(engine.as_ref())?;
    assert_eq!(reloaded.schema(), schema);
    Ok(())
}

fn with_nested_ids(field: StructField, ids: serde_json::Value) -> StructField {
    field.add_metadata([(
        ColumnMetadataKey::ColumnMappingNestedIds.as_ref(),
        MetadataValue::Other(ids),
    )])
}

#[rstest]
#[case::change_id(|s: &StructType| {
    let attrs = field(s, "attrs");
    let mut ids = nested_ids(&attrs).unwrap().clone();
    let key = ids.keys().next().unwrap().clone();
    ids.insert(key, 1000.into());
    retype_with(s, "attrs", with_nested_ids(attrs, ids.into()))
})]
#[case::remove(|s: &StructType| {
    let mut scores = field(s, "scores");
    scores.metadata.remove(ColumnMetadataKey::ColumnMappingNestedIds.as_ref());
    retype_with(s, "scores", scores)
})]
#[case::add_entry(|s: &StructType| {
    let MetadataValue::String(name) = physical_name(&field(s, "tags")).clone() else {
        unreachable!()
    };
    let tags = field(s, "tags");
    let mut ids = nested_ids(&tags).unwrap().clone();
    ids.insert(format!("{name}.extra"), 1000.into());
    retype_with(s, "tags", with_nested_ids(tags, ids.into()))
})]
#[tokio::test]
async fn updated_schema_rejects_changing_nested_ids(
    #[case] edit: SchemaEdit,
    #[values("icebergCompatV2", "icebergCompatV3")] compat: &str,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create_iceberg_compat_table(&table_path, engine.as_ref(), compat)?;
    let updated = with_fields(edit(&snapshot.schema()));
    assert_update_rejected(
        &snapshot,
        engine.as_ref(),
        updated,
        "Cannot change `delta.columnMapping.nested.ids` of existing field",
    );
    Ok(())
}

/// Returns the fields of `schema` with the field named `name` replaced by `replacement`.
fn retype_with(schema: &StructType, name: &str, replacement: StructField) -> Vec<StructField> {
    schema
        .fields()
        .map(|f| {
            if f.name() == name {
                replacement.clone()
            } else {
                f.clone()
            }
        })
        .collect()
}

#[tokio::test]
async fn updated_schema_rejects_nested_ids_on_a_new_field() -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create_iceberg_compat_table(&table_path, engine.as_ref(), "icebergCompatV2")?;
    let added = with_nested_ids(
        StructField::nullable("events", ArrayType::new(DataType::LONG, true)),
        serde_json::json!({ "events.element": 1000 }),
    );
    assert_update_rejected(
        &snapshot,
        engine.as_ref(),
        with_fields(appended(&snapshot.schema(), added)),
        "pre-populated `delta.columnMapping.nested.ids` metadata",
    );
    Ok(())
}

#[rstest]
#[case::v2_variant("icebergCompatV2", DataType::unshredded_variant())]
#[case::v2_variant_in_array(
    "icebergCompatV2",
    ArrayType::new(DataType::unshredded_variant(), true).into()
)]
#[case::v3_void("icebergCompatV3", DataType::VOID)]
#[tokio::test]
async fn updated_schema_rejects_types_iceberg_compat_does_not_support(
    #[case] compat: &str,
    #[case] data_type: DataType,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create_iceberg_compat_table(&table_path, engine.as_ref(), compat)?;
    let added = StructField::nullable("unsupported", data_type);
    assert_update_rejected(
        &snapshot,
        engine.as_ref(),
        with_fields(appended(&snapshot.schema(), added)),
        &format!("{compat} does not support type at column: unsupported"),
    );
    Ok(())
}

// === Column defaults ===

/// A schema with an existing field of each type family Kernel validates defaults for.
fn defaults_schema() -> SchemaRef {
    let address = StructType::try_new([
        StructField::nullable("city", DataType::STRING),
        StructField::nullable("zip", DataType::INTEGER),
    ])
    .unwrap();
    Arc::new(
        StructType::try_new([
            StructField::nullable("name", DataType::STRING),
            StructField::nullable("address", address),
            StructField::nullable("i", DataType::INTEGER),
            StructField::nullable("d", DataType::DOUBLE),
            StructField::nullable("dec", DataType::decimal(10, 5).unwrap()),
            StructField::nullable("dt", DataType::DATE),
            StructField::nullable("ts", DataType::TIMESTAMP),
            StructField::nullable("ntz", DataType::TIMESTAMP_NTZ),
            StructField::nullable("b", DataType::BYTE),
            StructField::nullable("arr", ArrayType::new(DataType::INTEGER, true)),
        ])
        .unwrap(),
    )
}

/// Creates a column-mapped table with `defaults_schema` and adds the `allowColumnDefaults` feature
/// in a hand-written protocol commit, because `create_table` cannot enable that feature.
fn create_column_defaults_table(
    table_path: &str,
    engine: &dyn Engine,
    iceberg_compat_v3: bool,
) -> delta_kernel::Result<SnapshotRef> {
    let properties = if iceberg_compat_v3 {
        iceberg_compat_properties("icebergCompatV3")
    } else {
        vec![("delta.columnMapping.mode", "name")]
    };
    create(table_path, engine, defaults_schema(), &properties, None)?;
    let table_url = url::Url::from_directory_path(table_path).unwrap();
    let mut protocol = read_actions_from_commit(&table_url, 0, "protocol")?.remove(0);
    protocol["writerFeatures"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!("allowColumnDefaults"));
    std::fs::write(
        std::path::Path::new(table_path).join("_delta_log/00000000000000000001.json"),
        format!("{}\n", serde_json::json!({ "protocol": protocol })),
    )
    .map_err(delta_kernel::KernelError::from)?;
    let snapshot = Snapshot::builder_for(table_path).build(engine)?;
    assert!(
        snapshot
            .table_configuration()
            .is_feature_supported(&TableFeature::AllowColumnDefaults)
    );
    Ok(snapshot)
}

/// Returns `field` with its default set to `value`, or removed when `value` is `None`.
fn with_default(field: StructField, value: Option<&str>) -> StructField {
    let mut field = field;
    let key = ColumnMetadataKey::CurrentDefault.as_ref();
    field.metadata.remove(key);
    match value {
        Some(value) => field.add_metadata([(key, MetadataValue::String(value.into()))]),
        None => field,
    }
}

/// Returns `schema` with the defaults of the top-level `name` and nested `address.zip` fields set
/// to `name_default` and `zip_default`.
fn with_name_and_zip_defaults(
    schema: &StructType,
    name_default: Option<&str>,
    zip_default: Option<&str>,
) -> StructType {
    with_fields(schema.fields().map(|field| match field.name().as_str() {
        "name" => with_default(field.clone(), name_default),
        "address" => {
            let DataType::Struct(address) = field.data_type() else {
                panic!("address must be a struct");
            };
            let address = with_fields(address.fields().map(|nested| {
                if nested.name() == "zip" {
                    with_default(nested.clone(), zip_default)
                } else {
                    nested.clone()
                }
            }));
            StructField {
                data_type: address.into(),
                ..field.clone()
            }
        }
        _ => field.clone(),
    }))
}

fn default_sql(field: &StructField) -> Option<String> {
    field
        .column_default()
        .unwrap()
        .map(|default| default.raw_sql().to_string())
}

/// Returns the defaults of `name` and `address.zip` in the reloaded table.
fn reloaded_name_and_zip_defaults(
    table_path: &str,
    engine: &dyn Engine,
) -> (Option<String>, Option<String>) {
    let schema = Snapshot::builder_for(table_path)
        .build(engine)
        .unwrap()
        .schema();
    let DataType::Struct(address) = field(&schema, "address").data_type().clone() else {
        panic!("address must be a struct");
    };
    (
        default_sql(&field(&schema, "name")),
        default_sql(&field(&address, "zip")),
    )
}

#[tokio::test]
async fn updated_schema_sets_changes_and_removes_top_level_and_nested_defaults()
-> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let mut snapshot = create_column_defaults_table(&table_path, engine.as_ref(), true)?;
    let protocol = snapshot.table_configuration().protocol().clone();
    for (name_default, zip_default) in [
        (Some("'pending'"), Some("10115")),
        (Some("\"done\""), Some("'-7'")),
        (None, None),
    ] {
        let schema = with_name_and_zip_defaults(&snapshot.schema(), name_default, zip_default);
        snapshot = update_schema(&snapshot, engine.as_ref(), schema)?;
        assert_eq!(
            reloaded_name_and_zip_defaults(&table_path, engine.as_ref()),
            (
                name_default.map(str::to_string),
                zip_default.map(str::to_string)
            )
        );
        assert_eq!(snapshot.table_configuration().protocol(), &protocol);
    }
    Ok(())
}

#[rstest]
#[case::string_not_quoted("name", "CURRENT_TIMESTAMP()", "Only literal values")]
#[case::integer_fraction("i", "313.55", "Only literal values")]
#[case::double_text("d", "Good boy", "Only literal values")]
#[case::decimal_precision("dec", "234243243243243234.234", "Only literal values")]
#[case::decimal_scale("dec", "1.234567", "Only literal values")]
#[case::date_format("dt", "2022/01/05", "Only literal values")]
#[case::timestamp_without_time("ts", "2025-01-01", "Only literal values")]
#[case::timestamp_ntz_with_offset("ntz", "2025-01-01T00:00:00+02:00", "Only literal values")]
#[case::non_literal_expression("ts", "current_timestamp()", "Only literal values")]
#[case::null("i", "NULL", "Only literal values")]
#[case::unsupported_byte("b", "1", "does not support default values for data type byte")]
#[case::unsupported_array(
    "arr",
    "array(1)",
    "does not support default values for data type array<integer>"
)]
#[tokio::test]
async fn updated_schema_rejects_invalid_defaults(
    #[case] column: &str,
    #[case] value: &str,
    #[case] expected: &str,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create_column_defaults_table(&table_path, engine.as_ref(), true)?;
    let schema = snapshot.schema();
    let updated = with_fields(schema.fields().map(|field| {
        if field.name() == column {
            with_default(field.clone(), Some(value))
        } else {
            field.clone()
        }
    }));
    assert_update_rejected(&snapshot, engine.as_ref(), updated, expected);
    Ok(())
}

#[tokio::test]
async fn updated_schema_rejects_a_default_on_a_new_field() -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create_column_defaults_table(&table_path, engine.as_ref(), true)?;
    let added = with_default(StructField::nullable("new", DataType::STRING), Some("'x'"));
    assert_update_rejected(
        &snapshot,
        engine.as_ref(),
        with_fields(appended(&snapshot.schema(), added)),
        "Cannot add field 'new' under 'new' with a default value",
    );
    Ok(())
}

/// Defaults do not enable `allowColumnDefaults`, so a table without the feature rejects them.
#[rstest]
#[case::column_mapping(&[("delta.columnMapping.mode", "name")])]
#[case::iceberg_compat_v3(&[
    ("delta.columnMapping.mode", "name"),
    ("delta.enableIcebergCompatV3", "true"),
    ("delta.enableRowTracking", "true"),
])]
#[tokio::test]
async fn updated_schema_rejects_defaults_without_allow_column_defaults(
    #[case] properties: &[(&str, &str)],
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create(
        &table_path,
        engine.as_ref(),
        defaults_schema(),
        properties,
        None,
    )?;
    let updated = with_name_and_zip_defaults(&snapshot.schema(), Some("'x'"), None);
    assert_update_rejected(
        &snapshot,
        engine.as_ref(),
        updated,
        "the table does not support the allowColumnDefaults table feature",
    );
    assert!(
        !snapshot
            .table_configuration()
            .is_feature_supported(&TableFeature::AllowColumnDefaults)
    );
    Ok(())
}

/// Like Java, Kernel rejects any schema update on an `allowColumnDefaults` table without
/// icebergCompatV3, even one that does not touch defaults.
#[rstest]
#[case::set_default(Some("'x'"))]
#[case::rename_only(None)]
#[tokio::test]
async fn updated_schema_requires_iceberg_compat_v3_with_allow_column_defaults(
    #[case] name_default: Option<&str>,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create_column_defaults_table(&table_path, engine.as_ref(), false)?;
    let schema = with_name_and_zip_defaults(&snapshot.schema(), name_default, None);
    let updated = with_fields(schema.fields().map(|field| match field.name().as_str() {
        "i" => field.with_name("renamed"),
        _ => field.clone(),
    }));
    assert_update_rejected(
        &snapshot,
        engine.as_ref(),
        updated,
        "allowColumnDefaults table feature only when icebergCompatV3 is enabled",
    );
    Ok(())
}
