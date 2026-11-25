//! Integration tests for changing clustering columns on an existing table.

use std::sync::Arc;

use delta_kernel::Engine;
use delta_kernel::actions::{MAX_VALUES, MIN_VALUES};
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::expressions::{ColumnName, column_name};
use delta_kernel::schema::{
    ColumnMetadataKey, DataType, MetadataValue, SchemaRef, StructField, StructType, schema_ref,
};
use delta_kernel::snapshot::{Snapshot, SnapshotRef};
use delta_kernel::table_features::TableFeature;
use delta_kernel::transaction::UpdateTableOperation;
use delta_kernel::transaction::create_table::create_table;
use delta_kernel::transaction::data_layout::DataLayout;
use rstest::rstest;
use test_utils::{
    IntoArray, assert_result_error_with_message, generate_batch, read_add_infos, test_table_setup,
    test_table_setup_mt, write_batch_to_table,
};

fn clustering_schema() -> SchemaRef {
    schema_ref! {
        nullable "id": INTEGER,
        nullable "name": STRING,
        nullable "flag": BOOLEAN,
        nullable "address": {
            nullable "city": STRING,
            nullable "zip.code": STRING,
        },
    }
}

fn create(
    table_path: &str,
    engine: &dyn Engine,
    layout: DataLayout,
    column_mapping: &str,
) -> delta_kernel::Result<SnapshotRef> {
    let result = create_table(table_path, clustering_schema(), "test-engine", false)
        .with_data_layout(layout)
        .with_table_properties([("delta.columnMapping.mode", column_mapping)])
        .build(engine, Box::new(FileSystemCommitter::new()))?
        .commit(engine)?;
    Ok(result.unwrap_post_commit_snapshot())
}

fn alter_clustering(
    snapshot: &SnapshotRef,
    engine: &dyn Engine,
    clustering_columns: Vec<ColumnName>,
) -> delta_kernel::Result<SnapshotRef> {
    let result = snapshot
        .clone()
        .update_table(UpdateTableOperation::AlterTable, "test-engine", false)
        .with_clustering_columns(clustering_columns)
        .build(engine, Box::new(FileSystemCommitter::new()))?
        .commit(engine)?;
    Ok(result.unwrap_post_commit_snapshot())
}

fn reload(table_path: &str, engine: &dyn Engine) -> delta_kernel::Result<SnapshotRef> {
    let table_url = delta_kernel::try_parse_uri(table_path)?;
    Snapshot::builder_for(table_url).build(engine)
}

/// Returns the physical path of the logical `path` in `schema`: the field names when column
/// mapping is off, and each field's `delta.columnMapping.physicalName` otherwise.
fn physical_path(schema: &StructType, path: &[&str], column_mapping: &str) -> ColumnName {
    let mut current = schema;
    let mut physical = Vec::with_capacity(path.len());
    for (index, segment) in path.iter().enumerate() {
        let field = current.field(segment).expect("field exists");
        physical.push(match column_mapping {
            "none" => field.name().clone(),
            _ => match field.get_config_value(&ColumnMetadataKey::ColumnMappingPhysicalName) {
                Some(MetadataValue::String(name)) => name.clone(),
                other => panic!("missing physical name on '{segment}': {other:?}"),
            },
        });
        if index + 1 < path.len() {
            let DataType::Struct(inner) = field.data_type() else {
                panic!("'{segment}' is not a struct");
            };
            current = inner;
        }
    }
    ColumnName::new(physical)
}

fn assert_clustering_features(snapshot: &Snapshot) {
    let table_config = snapshot.table_configuration();
    assert!(table_config.is_feature_supported(&TableFeature::ClusteredTable));
    assert!(table_config.is_feature_supported(&TableFeature::DomainMetadata));
}

#[rstest]
#[case::reorder(Some(vec![column_name!("id"), column_name!("name")]), vec![column_name!("name"), column_name!("id")], vec![column_name!("name"), column_name!("id")])]
#[case::replace(Some(vec![column_name!("id")]), vec![column_name!("address.city")], vec![column_name!("address.city")])]
#[case::case_insensitive(Some(vec![column_name!("id")]), vec![column_name!("NAME"), column_name!("Address.City")], vec![column_name!("name"), column_name!("address.city")])]
#[case::enable_on_unclustered(None, vec![column_name!("id"), column_name!("name")], vec![column_name!("id"), column_name!("name")])]
#[tokio::test]
async fn alter_table_replaces_clustering_columns_in_order(
    #[case] initial: Option<Vec<ColumnName>>,
    #[case] requested: Vec<ColumnName>,
    #[case] expected: Vec<ColumnName>,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let initially_clustered = initial.is_some();
    let layout = match initial {
        Some(columns) => DataLayout::Clustered { columns },
        None => DataLayout::None,
    };
    let created = create(&table_path, engine.as_ref(), layout, "none")?;
    assert_eq!(
        created
            .table_configuration()
            .is_feature_supported(&TableFeature::ClusteredTable),
        initially_clustered
    );

    alter_clustering(&created, engine.as_ref(), requested)?;

    let snapshot = reload(&table_path, engine.as_ref())?;
    assert_eq!(snapshot.version(), 1);
    assert_eq!(
        snapshot
            .reader()
            .get_physical_clustering_columns(engine.as_ref())?,
        Some(expected)
    );
    assert_clustering_features(&snapshot);
    // Changing clustering columns never changes the schema.
    assert_eq!(snapshot.schema(), created.schema());
    Ok(())
}

#[rstest]
#[tokio::test]
async fn alter_table_stores_physical_path_for_nested_clustering_column(
    #[values("none", "name", "id")] column_mapping: &str,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let created = create(
        &table_path,
        engine.as_ref(),
        DataLayout::None,
        column_mapping,
    )?;
    let nested = ColumnName::new(["address", "zip.code"]);

    alter_clustering(&created, engine.as_ref(), vec![nested.clone()])?;

    let snapshot = reload(&table_path, engine.as_ref())?;
    let infos = snapshot
        .reader()
        .get_clustering_column_infos(engine.as_ref())?
        .expect("table is clustered");
    assert_eq!(infos.len(), 1);
    assert_eq!(infos[0].logical_column, nested);
    assert_eq!(infos[0].data_type, DataType::STRING);
    assert_eq!(
        infos[0].physical_column,
        physical_path(&snapshot.schema(), &["address", "zip.code"], column_mapping)
    );
    assert_eq!(
        infos[0].physical_column == nested,
        column_mapping == "none",
        "column-mapping tables store physical names: {:?}",
        infos[0].physical_column
    );
    Ok(())
}

#[tokio::test]
async fn reasserting_current_clustering_columns_changes_only_the_domain() -> delta_kernel::Result<()>
{
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let created = create(
        &table_path,
        engine.as_ref(),
        DataLayout::clustered(["id", "name"]),
        "name",
    )?;
    let before = created
        .reader()
        .get_physical_clustering_columns(engine.as_ref())?;

    alter_clustering(
        &created,
        engine.as_ref(),
        vec![column_name!("id"), column_name!("name")],
    )?;

    let snapshot = reload(&table_path, engine.as_ref())?;
    assert_eq!(snapshot.version(), 1);
    assert_eq!(
        snapshot
            .reader()
            .get_physical_clustering_columns(engine.as_ref())?,
        before
    );
    assert_eq!(
        snapshot.table_configuration().protocol(),
        created.table_configuration().protocol()
    );
    assert_eq!(
        snapshot.table_configuration().metadata(),
        created.table_configuration().metadata()
    );
    Ok(())
}

/// A write can change clustering columns and add files in one commit. Only the new file collects
/// statistics for the new key; existing files are not rewritten.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn write_changes_clustering_columns_and_adds_files_in_one_commit()
-> Result<(), Box<dyn std::error::Error>> {
    let (_temp_dir, table_path, engine) = test_table_setup_mt()?;
    let schema = schema_ref! {
        nullable "id": INTEGER,
        nullable "name": STRING,
    };
    let created = create_table(&table_path, schema, "test-engine", true)
        .with_data_layout(DataLayout::clustered(["id"]))
        .with_table_properties([("delta.dataSkippingNumIndexedCols", "1")])
        .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?
        .commit(engine.as_ref())?
        .unwrap_post_commit_snapshot();
    let batch = || {
        generate_batch(vec![
            ("id", vec![1, 2].into_arrow_array()),
            ("name", vec!["a", "b"].into_arrow_array()),
        ])
    };
    let written =
        write_batch_to_table(&created, engine.as_ref(), batch()?, Default::default()).await?;
    let first_file = read_add_infos(&written, engine.as_ref())?;
    let [first_file] = first_file.as_slice() else {
        panic!("expected one file");
    };
    assert!(
        first_file.stats.as_ref().unwrap()[MIN_VALUES]
            .get("name")
            .is_none()
    );

    let mut txn = written
        .clone()
        .update_table(UpdateTableOperation::Write, "test-engine", true)
        .with_clustering_columns(vec![column_name!("name")])
        .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?;
    let write_context = txn.write_state()?.write_context_builder().build()?;
    let add_files = engine
        .write_parquet(&ArrowEngineData::new(batch()?), &write_context)
        .await?;
    txn.add_files(add_files);
    let snapshot = txn.commit(engine.as_ref())?.unwrap_post_commit_snapshot();

    assert_eq!(snapshot.version(), 2);
    assert_eq!(
        snapshot
            .reader()
            .get_physical_clustering_columns(engine.as_ref())?,
        Some(vec![column_name!("name")])
    );
    let add_infos = read_add_infos(&snapshot, engine.as_ref())?;
    assert_eq!(add_infos.len(), 2);
    let unchanged = add_infos
        .iter()
        .find(|info| info.path == first_file.path)
        .expect("first file is still in the table");
    assert_eq!(unchanged.stats, first_file.stats);
    let new_file = add_infos
        .iter()
        .find(|info| info.path != first_file.path)
        .expect("new file");
    let stats = new_file.stats.as_ref().expect("new file has stats");
    for key in [MIN_VALUES, MAX_VALUES] {
        assert!(stats[key].get("name").is_some(), "{key} lacks the new key");
    }
    Ok(())
}

#[rstest]
#[tokio::test]
async fn empty_clustering_columns_clear_keys_and_keep_features(
    #[values(true, false)] initially_clustered: bool,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let layout = if initially_clustered {
        DataLayout::clustered(["id"])
    } else {
        DataLayout::None
    };
    let created = create(&table_path, engine.as_ref(), layout, "none")?;

    alter_clustering(&created, engine.as_ref(), vec![])?;

    let snapshot = reload(&table_path, engine.as_ref())?;
    assert_eq!(
        snapshot
            .reader()
            .get_physical_clustering_columns(engine.as_ref())?,
        Some(vec![])
    );
    assert_clustering_features(&snapshot);
    Ok(())
}

#[tokio::test]
async fn omitted_clustering_columns_keep_current_keys() -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let created = create(
        &table_path,
        engine.as_ref(),
        DataLayout::clustered(["name", "id"]),
        "name",
    )?;

    created
        .update_table(UpdateTableOperation::AlterTable, "test-engine", false)
        .add_column(StructField::nullable("country", DataType::STRING))
        .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?
        .commit(engine.as_ref())?
        .unwrap_committed();

    let snapshot = reload(&table_path, engine.as_ref())?;
    let logical: Vec<_> = snapshot
        .reader()
        .get_clustering_column_infos(engine.as_ref())?
        .expect("table is clustered")
        .into_iter()
        .map(|info| info.logical_column)
        .collect();
    assert_eq!(logical, vec![column_name!("name"), column_name!("id")]);
    Ok(())
}

#[rstest]
#[tokio::test]
async fn clustering_columns_resolve_against_schema_changes_in_same_transaction(
    #[values("none", "name")] column_mapping: &str,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let created = create(
        &table_path,
        engine.as_ref(),
        DataLayout::clustered(["id"]),
        column_mapping,
    )?;

    let txn = created
        .update_table(UpdateTableOperation::AlterTable, "test-engine", false)
        .add_column(StructField::nullable("country", DataType::STRING))
        .with_clustering_columns(vec![column_name!("country"), column_name!("id")])
        .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?;
    let physical_columns = txn.stats_columns();
    txn.commit(engine.as_ref())?.unwrap_committed();

    let snapshot = reload(&table_path, engine.as_ref())?;
    let infos = snapshot
        .reader()
        .get_clustering_column_infos(engine.as_ref())?
        .expect("table is clustered");
    let logical: Vec<_> = infos
        .iter()
        .map(|info| info.logical_column.clone())
        .collect();
    assert_eq!(logical, vec![column_name!("country"), column_name!("id")]);
    // Writes in the same transaction collect statistics for the new clustering columns.
    for info in &infos {
        assert!(physical_columns.contains(&info.physical_column));
    }
    Ok(())
}

#[rstest]
#[case::unresolved(vec![column_name!("missing")], "not found")]
#[case::unresolved_nested(vec![column_name!("address.missing")], "not found")]
#[case::duplicate(vec![column_name!("id"), column_name!("name"), column_name!("id")], "Duplicate clustering column")]
#[case::duplicate_case_insensitive(vec![column_name!("id"), column_name!("ID")], "Duplicate clustering column")]
#[case::boolean(vec![column_name!("flag")], "unsupported type")]
#[case::struct_column(vec![column_name!("address")], "unsupported type")]
#[tokio::test]
async fn invalid_clustering_columns_are_rejected(
    #[case] requested: Vec<ColumnName>,
    #[case] expected_error: &str,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let created = create(
        &table_path,
        engine.as_ref(),
        DataLayout::clustered(["id"]),
        "none",
    )?;

    let result = created
        .clone()
        .update_table(UpdateTableOperation::AlterTable, "test-engine", false)
        .with_clustering_columns(requested)
        .build(engine.as_ref(), Box::new(FileSystemCommitter::new()));
    assert_result_error_with_message(result, expected_error);

    // Clustering never adds a missing column to the schema.
    let snapshot = reload(&table_path, engine.as_ref())?;
    assert_eq!(snapshot.version(), 0);
    assert_eq!(snapshot.schema(), created.schema());
    Ok(())
}

#[rstest]
#[case::non_empty(vec![column_name!("name")])]
#[case::empty(vec![])]
#[tokio::test]
async fn clustering_a_partitioned_table_is_rejected(
    #[case] requested: Vec<ColumnName>,
    #[values(
        (UpdateTableOperation::AlterTable, false),
        (UpdateTableOperation::Write, true)
    )]
    operation: (UpdateTableOperation, bool),
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let created = create(
        &table_path,
        engine.as_ref(),
        DataLayout::partitioned(["id"]),
        "none",
    )?;

    let (operation, data_change) = operation;
    let result = Arc::clone(&created)
        .update_table(operation, "test-engine", data_change)
        .with_clustering_columns(requested)
        .build(engine.as_ref(), Box::new(FileSystemCommitter::new()));
    assert_result_error_with_message(result, "Cannot enable clustering on a partitioned table");

    let snapshot = reload(&table_path, engine.as_ref())?;
    assert_eq!(snapshot.version(), 0);
    assert!(
        !snapshot
            .table_configuration()
            .is_feature_supported(&TableFeature::ClusteredTable)
    );
    assert_eq!(
        snapshot.table_configuration().protocol(),
        created.table_configuration().protocol()
    );
    Ok(())
}
