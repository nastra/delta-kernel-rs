//! Integration tests for setting table properties on existing tables.

use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::schema::{SchemaRef, schema_ref};
use delta_kernel::snapshot::Snapshot;
use delta_kernel::table_features::TableFeature;
use delta_kernel::transaction::UpdateTableOperation;
use delta_kernel::{Engine, SnapshotRef};
use rstest::rstest;
use test_utils::{
    assert_result_error_with_message, create_table_and_load_snapshot, test_table_setup,
};

const CHECKPOINT_INTERVAL: &str = "delta.checkpointInterval";
const CHECKPOINT_POLICY: &str = "delta.checkpointPolicy";
const ENABLE_ROW_TRACKING: &str = "delta.enableRowTracking";
const ENABLE_TYPE_WIDENING: &str = "delta.enableTypeWidening";

fn simple_schema() -> SchemaRef {
    schema_ref! {
        nullable "id": INTEGER,
        nullable "name": STRING,
    }
}

/// Commits an ALTER TABLE that sets `properties` and returns the reloaded snapshot.
fn set_properties(
    snapshot: SnapshotRef,
    table_path: &str,
    engine: &dyn Engine,
    properties: &[(&str, &str)],
) -> delta_kernel::Result<SnapshotRef> {
    snapshot
        .update_table(UpdateTableOperation::AlterTable, "test-engine", false)
        .with_table_properties_added(properties.to_vec())
        .build(engine, Box::new(FileSystemCommitter::new()))?
        .commit(engine)?
        .unwrap_committed();
    Snapshot::builder_for(table_path).build(engine)
}

fn configuration_value<'a>(snapshot: &'a Snapshot, key: &str) -> Option<&'a str> {
    snapshot
        .table_configuration()
        .metadata()
        .configuration()
        .get(key)
        .map(String::as_str)
}

#[tokio::test]
async fn set_properties_patches_configuration_and_canonicalizes_delta_keys()
-> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create_table_and_load_snapshot(
        &table_path,
        simple_schema(),
        engine.as_ref(),
        &[("app.keep", "1"), ("app.overwrite", "old")],
    )?;
    let read_protocol = snapshot.table_configuration().protocol().clone();

    let reloaded = set_properties(
        snapshot,
        &table_path,
        engine.as_ref(),
        &[
            ("app.overwrite", "new"),
            ("App.MixedCase", "kept"),
            ("format-version", "2"),
            ("DELTA.CHECKPOINTINTERVAL", "5"),
        ],
    )?;

    assert_eq!(reloaded.version(), 1);
    assert_eq!(configuration_value(&reloaded, "app.keep"), Some("1"));
    assert_eq!(configuration_value(&reloaded, "app.overwrite"), Some("new"));
    assert_eq!(
        configuration_value(&reloaded, "App.MixedCase"),
        Some("kept")
    );
    assert_eq!(configuration_value(&reloaded, "format-version"), Some("2"));
    assert_eq!(
        configuration_value(&reloaded, CHECKPOINT_INTERVAL),
        Some("5")
    );
    assert_eq!(
        configuration_value(&reloaded, "DELTA.CHECKPOINTINTERVAL"),
        None
    );
    assert_eq!(reloaded.table_configuration().protocol(), &read_protocol);
    Ok(())
}

#[rstest]
#[case::malformed_boolean(&[], &[("delta.enableChangeDataFeed", "yes")], "Invalid value")]
#[case::malformed_integer(&[], &[(CHECKPOINT_INTERVAL, "0")], "Invalid value")]
#[case::checkpoint_interval_beyond_java_int(
    &[],
    &[(CHECKPOINT_INTERVAL, "2147483648")],
    "must be between 1 and 2147483647"
)]
#[case::num_indexed_cols_beyond_java_int(
    &[],
    &[("delta.dataSkippingNumIndexedCols", "2147483648")],
    "must be between -1 and 2147483647"
)]
#[case::malformed_enum(&[], &[(CHECKPOINT_POLICY, "v3")], "Invalid value")]
#[case::unknown_delta_key(&[], &[("delta.unknownProperty", "x")], "not supported")]
#[case::unknown_delta_key_case_variant(&[], &[("Delta.unknownProperty", "x")], "not supported")]
#[case::unknown_feature_in_other_case(
    &[],
    &[("DELTA.FEATURE.noSuchFeature", "supported")],
    "Enabling feature 'noSuchFeature'"
)]
#[case::kernel_managed_key(&[], &[("delta.columnMapping.maxColumnId", "100")], "not supported")]
#[case::not_editable_in_java(
    &[],
    &[("delta.setTransactionRetentionDuration", "interval 2 weeks")],
    "not supported"
)]
#[case::unknown_to_java(&[], &[("delta.parquet.format.version", "2.12.0")], "not supported")]
#[case::duplicate_after_canonicalization(
    &[],
    &[("delta.appendOnly", "true"), ("DELTA.APPENDONLY", "false")],
    "more than once"
)]
#[case::enable_row_tracking(&[], &[(ENABLE_ROW_TRACKING, "true")], "row tracking")]
#[case::change_column_mapping(
    &[("delta.columnMapping.mode", "name")],
    &[("delta.columnMapping.mode", "id")],
    "column mapping mode"
)]
#[tokio::test]
async fn set_properties_rejects_invalid_requests(
    #[case] create_properties: &[(&str, &str)],
    #[case] properties: &[(&str, &str)],
    #[case] error_contains: &str,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create_table_and_load_snapshot(
        &table_path,
        simple_schema(),
        engine.as_ref(),
        create_properties,
    )?;

    let result = snapshot
        .clone()
        .update_table(UpdateTableOperation::AlterTable, "test-engine", false)
        .with_table_properties_added(properties.to_vec())
        .build(engine.as_ref(), Box::new(FileSystemCommitter::new()));
    assert_result_error_with_message(result, error_contains);

    let reloaded = Snapshot::builder_for(&table_path).build(engine.as_ref())?;
    assert_eq!(reloaded.version(), 0);
    assert_eq!(
        reloaded.table_configuration().metadata(),
        snapshot.table_configuration().metadata()
    );
    assert_eq!(
        reloaded.table_configuration().protocol(),
        snapshot.table_configuration().protocol()
    );
    Ok(())
}

#[rstest]
#[case::exact("delta.feature.domainMetadata")]
#[case::name_case("delta.feature.DOMAINMETADATA")]
#[case::prefix_case("DELTA.Feature.domainMetadata")]
#[tokio::test]
async fn feature_signals_match_prefix_and_feature_name_in_any_case(
    #[case] key: &str,
) -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot =
        create_table_and_load_snapshot(&table_path, simple_schema(), engine.as_ref(), &[])?;
    assert!(
        !snapshot
            .table_configuration()
            .is_feature_supported(&TableFeature::DomainMetadata)
    );

    let reloaded = set_properties(
        snapshot,
        &table_path,
        engine.as_ref(),
        &[(key, "supported")],
    )?;

    assert_eq!(reloaded.version(), 1);
    let config = reloaded.table_configuration();
    assert!(config.is_feature_supported(&TableFeature::DomainMetadata));
    assert!(
        !config
            .metadata()
            .configuration()
            .keys()
            .any(|key| key.to_ascii_lowercase().starts_with("delta.feature."))
    );
    Ok(())
}

#[tokio::test]
async fn reasserting_row_tracking_keeps_materialized_column_names() -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create_table_and_load_snapshot(
        &table_path,
        simple_schema(),
        engine.as_ref(),
        &[(ENABLE_ROW_TRACKING, "true")],
    )?;
    let names = |snapshot: &Snapshot| {
        [
            "delta.rowTracking.materializedRowIdColumnName",
            "delta.rowTracking.materializedRowCommitVersionColumnName",
        ]
        .map(|key| configuration_value(snapshot, key).map(str::to_string))
    };
    let read_names = names(&snapshot);
    assert!(read_names.iter().all(Option::is_some));

    let reloaded = set_properties(
        snapshot,
        &table_path,
        engine.as_ref(),
        &[(ENABLE_ROW_TRACKING, "true"), ("app.key", "value")],
    )?;

    assert_eq!(reloaded.version(), 1);
    assert_eq!(names(&reloaded), read_names);
    assert!(
        reloaded
            .table_configuration()
            .is_feature_enabled(&TableFeature::RowTracking)
    );
    Ok(())
}

#[tokio::test]
async fn disabling_type_widening_keeps_the_feature() -> delta_kernel::Result<()> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot = create_table_and_load_snapshot(
        &table_path,
        simple_schema(),
        engine.as_ref(),
        &[(ENABLE_TYPE_WIDENING, "true")],
    )?;

    let reloaded = set_properties(
        snapshot,
        &table_path,
        engine.as_ref(),
        &[(ENABLE_TYPE_WIDENING, "false")],
    )?;

    let config = reloaded.table_configuration();
    assert_eq!(
        configuration_value(&reloaded, ENABLE_TYPE_WIDENING),
        Some("false")
    );
    assert!(config.is_feature_supported(&TableFeature::TypeWidening));
    assert!(!config.is_feature_enabled(&TableFeature::TypeWidening));
    Ok(())
}

#[tokio::test]
async fn v2_checkpoint_policy_adds_feature_without_writing_a_checkpoint() -> delta_kernel::Result<()>
{
    let (temp_dir, table_path, engine) = test_table_setup()?;
    let snapshot =
        create_table_and_load_snapshot(&table_path, simple_schema(), engine.as_ref(), &[])?;
    assert!(
        !snapshot
            .table_configuration()
            .is_feature_supported(&TableFeature::V2Checkpoint)
    );

    let reloaded = set_properties(
        snapshot,
        &table_path,
        engine.as_ref(),
        &[(CHECKPOINT_POLICY, "v2")],
    )?;

    assert_eq!(
        configuration_value(&reloaded, CHECKPOINT_POLICY),
        Some("v2")
    );
    assert!(
        reloaded
            .table_configuration()
            .is_feature_supported(&TableFeature::V2Checkpoint)
    );
    let log_files: Vec<_> = std::fs::read_dir(temp_dir.path().join("_delta_log"))
        .expect("read _delta_log")
        .map(|entry| entry.expect("log entry").file_name().into_string().unwrap())
        .collect();
    assert!(
        log_files
            .iter()
            .all(|name| !name.contains("checkpoint") && !name.ends_with(".parquet")),
        "{log_files:?}"
    );
    Ok(())
}
