//! Builder for transactions against an existing table.

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use delta_kernel_derive::internal_api;

use super::TransactionBuilder;
use super::state::TransactionBuilderState;
use super::update_metadata::{
    PreparedUpdateTableMetadata, UpdateTableMetadataRequest, prepare_update_table_metadata,
};
use crate::committer::Committer;
use crate::expressions::ColumnName;
use crate::schema::{SchemaRef, StructField};
use crate::snapshot::SnapshotRef;
use crate::table_features::{
    IcebergCompatValidationContext, Operation, V2_VALIDATOR, V3_VALIDATOR,
    validate_iceberg_compat_if_needed,
};
use crate::transaction::domain_metadata::validate_unique_domains;
use crate::transaction::schema_evolution::SchemaOperation;
use crate::transaction::schema_update::validate_clustering_columns_retained;
use crate::transaction::{
    ExecutionMode, Imperative, Transaction, UpdateTable, UpdateTableOperation,
};
use crate::utils::{current_time_ms, require};
use crate::{Engine, KernelError, KernelResult};

/// Configures DML and schema-changing DDL transactions against an existing table.
///
/// This builder does not create or replace tables. Construct it with
/// [`Snapshot::update_table`](crate::Snapshot::update_table); operation, engine info, and
/// `data_change` cannot be changed afterward.
/// [`build`](UpdateTableTransactionBuilder::build) validates the accumulated intent and constructs
/// the transaction.
///
/// # Schema changes
///
/// Use **ALTER TABLE** for a standalone schema change without changing table data, such as adding
/// a nullable column without writing data files. Select [`UpdateTableOperation::AlterTable`].
///
/// Use **schema evolution during a write** to change the schema while writing or rewriting data
/// files in the same transaction. Select the data operation, such as
/// [`UpdateTableOperation::Write`], and configure the schema changes on the same builder.
///
/// In both workflows, configure schema operations before `build()`, which validates and applies
/// them. For schema evolution, stage data-file changes on the returned [`Transaction`], using the
/// evolved schema provided by [`Transaction::write_state`].
///
/// When the evolved schema contains a type or annotation whose table feature the protocol does not
/// support, such as `timestampNtz`, `variantType`, or `collations`, `build()` adds the feature and
/// the commit includes a Protocol action. Existing protocol features are always retained.
///
/// Set `data_change` when constructing the builder: `true` for logical data changes, or `false`
/// for metadata-only changes and data-preserving rewrites. `ALTER TABLE` requires `false`, while
/// blind append requires `true`. [`build`](UpdateTableTransactionBuilder::build) rejects
/// incompatible intent; commit validates the staged file actions and rejects any staged
/// data-file batches or manifest writes for `ALTER TABLE`.
///
/// cbindgen:ignore
pub type UpdateTableTransactionBuilder<MODE = Imperative> = TransactionBuilder<UpdateTable, MODE>;

/// Opaque state collected by update-table-specific builder setters.
#[doc(hidden)]
pub struct UpdateTableBuilderState {
    snapshot: SnapshotRef,
    operation: UpdateTableOperation,
    schema_changes: Option<Vec<SchemaOperation>>,
    updated_schema: Option<SchemaRef>,
    // Configuration overrides for the shared update preparation path. No public setter populates
    // these yet.
    properties_added: HashMap<String, String>,
    property_keys_removed: HashSet<String>,
    domain_metadata_removals: Vec<String>,
    is_blind_append: bool,
}

impl<MODE: ExecutionMode> std::fmt::Debug for TransactionBuilder<UpdateTable, MODE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UpdateTableTransactionBuilder")
            .field("snapshot_version", &self.local_state.snapshot.version())
            .field("operation", &self.local_state.operation)
            .field(
                "schema_change_count",
                &self.local_state.schema_changes.as_ref().map_or(0, Vec::len),
            )
            .field(
                "has_updated_schema",
                &self.local_state.updated_schema.is_some(),
            )
            .finish()
    }
}

impl TransactionBuilder<UpdateTable> {
    pub(crate) fn new(
        snapshot: SnapshotRef,
        operation: UpdateTableOperation,
        engine_info: String,
        data_change: bool,
    ) -> Self {
        Self {
            common_state: TransactionBuilderState::new(engine_info, data_change),
            local_state: UpdateTableBuilderState {
                snapshot,
                operation,
                schema_changes: None,
                updated_schema: None,
                properties_added: HashMap::new(),
                property_keys_removed: HashSet::new(),
                domain_metadata_removals: Vec::new(),
                is_blind_append: false,
            },
            _mode: PhantomData,
        }
    }
}

impl<MODE: ExecutionMode> TransactionBuilder<UpdateTable, MODE> {
    /// Marks the transaction as a blind append assertion.
    ///
    /// Blind appends add new files without depending on existing table state. Commit validation
    /// requires staged Add metadata and rejects staged Remove metadata or deletion-vector batches.
    /// Blind append is also invalid for `ALTER TABLE`, schema changes, or `dataChange = false`.
    pub fn with_blind_append(mut self) -> Self {
        self.local_state.is_blind_append = true;
        self
    }

    /// Adds schema changes to validate and apply during [`build`](Self::build).
    ///
    /// Appends `changes` in call order and returns the updated builder. Each change observes the
    /// result of prior changes. Empty input is rejected immediately.
    ///
    /// # Errors
    ///
    /// Returns an error if `changes` is empty.
    #[internal_api]
    pub(crate) fn with_schema_changes(
        mut self,
        changes: impl IntoIterator<Item = SchemaOperation>,
    ) -> KernelResult<Self> {
        let mut changes = changes.into_iter().peekable();
        require!(
            changes.peek().is_some(),
            KernelError::generic("with_schema_changes requires at least one schema operation")
        );
        self.local_state
            .schema_changes
            .get_or_insert_with(Vec::new)
            .extend(changes);
        Ok(self)
    }

    /// Adds a nullable top-level column to the table schema.
    ///
    /// The field must not already exist in the schema, using a case-insensitive comparison, and
    /// must be nullable because existing data files do not contain it. On column-mapping tables,
    /// Kernel assigns or preserves column-mapping IDs and physical names.
    ///
    /// Schema changes are applied in call order and validated by [`build`](Self::build).
    pub fn add_column(mut self, field: StructField) -> Self {
        self.local_state
            .schema_changes
            .get_or_insert_with(Vec::new)
            .push(SchemaOperation::add_column(None, field));
        self
    }

    /// Adds a nullable field under `parent` in the table schema.
    ///
    /// An empty `parent` targets the root schema. Path segments may traverse nested structs, array
    /// elements, map keys, and map values, but the resolved parent must be a struct. The field must
    /// be nullable, must not be a metadata column, and must not collide case-insensitively with a
    /// sibling. On column-mapping tables, Kernel assigns or preserves column-mapping IDs and
    /// physical names.
    ///
    /// Schema changes are applied in call order and validated by [`build`](Self::build).
    pub fn add_column_at(mut self, parent: ColumnName, field: StructField) -> Self {
        self.local_state
            .schema_changes
            .get_or_insert_with(Vec::new)
            .push(SchemaOperation::add_column(parent, field));
        self
    }

    /// Changes a possibly nested column from non-nullable to nullable.
    ///
    /// The column path is resolved after all preceding schema changes. An already-nullable column
    /// is unchanged, but the transaction still emits its metadata action.
    pub fn set_nullable(mut self, column: ColumnName) -> Self {
        self.local_state
            .schema_changes
            .get_or_insert_with(Vec::new)
            .push(SchemaOperation::SetNullable { column });
        self
    }

    /// Replaces the table schema with `schema`, the complete desired schema.
    ///
    /// Start from [`Snapshot::schema`](crate::Snapshot::schema) and modify it: existing fields are
    /// identified by their column-mapping ID, so keep each retained field's column-mapping
    /// metadata. A field without column-mapping metadata is new and receives a fresh ID and
    /// physical name; new fields must not carry column-mapping metadata of their own. Omitting an
    /// existing field drops it. A retained field that omits its `delta.typeChanges` history keeps
    /// the current history. Repeated calls replace the previous schema.
    ///
    /// Supported changes are adding nullable fields, dropping fields, renaming fields, reordering
    /// fields within their struct, changing comments and other non-protocol field metadata, and
    /// relaxing nullability. Existing fields keep their physical names, so data files are not
    /// rewritten, and `delta.columnMapping.maxColumnId` never decreases. As with
    /// [`add_column`](Self::add_column), `build()` adds any table feature the new schema requires.
    ///
    /// When `delta.enableTypeWidening` is `true`, the type of a field, array element, or map key
    /// or value may also change to a wider type the Delta protocol supports, such as `integer` to
    /// `long`. Kernel appends each change to the field's `delta.typeChanges` metadata, keeping the
    /// existing history when the supplied field omits it. A partition column cannot widen from
    /// `date` to `timestamp_ntz`, change its decimal scale, or widen from an integer to a decimal
    /// with a non-zero scale, because Kernel cannot parse its existing partition values as the new
    /// type.
    ///
    /// [`build`](Self::build) rejects the update if the table does not use column mapping, or if
    /// the schema has no columns, tightens nullability, makes any other type change or alters
    /// `delta.typeChanges`, moves a field to a different parent struct, drops or renames a
    /// partition column or a column listed in
    /// `delta.dataSkippingStatsColumns`, drops a clustering column, renames a collated field, adds
    /// a non-nullable field or a field with a default value, or changes protocol-defined field
    /// metadata such as generation expressions, invariants, or defaults. It cannot be combined with
    /// [`add_column`](Self::add_column), [`add_column_at`](Self::add_column_at), or
    /// [`set_nullable`](Self::set_nullable) in the same transaction.
    ///
    /// Unlike Delta Kernel Java, `build()` also rejects any schema update on a table with
    /// `icebergCompatV2` or `icebergCompatV3` enabled, with `allowColumnDefaults` enabled, or with
    /// generated columns. Kernel cannot yet maintain the nested Iceberg field IDs, default values,
    /// or generation expressions of those tables when their schema changes.
    pub fn with_updated_schema(mut self, schema: SchemaRef) -> Self {
        self.local_state.updated_schema = Some(schema);
        self
    }

    /// Adds a user-controlled domain metadata removal to the transaction.
    ///
    /// If the domain exists, commit emits a tombstone that preserves its previous configuration.
    /// Removing a domain that does not exist is a no-op.
    ///
    /// Each domain may occur only once across additions and removals. Conflicts are rejected by
    /// [`build`](Self::build).
    pub fn with_domain_metadata_removed(mut self, domain: impl Into<String>) -> Self {
        self.local_state
            .domain_metadata_removals
            .push(domain.into());
        self
    }

    /// Validates the accumulated intent and builds a transaction.
    ///
    /// The caller-supplied `data_change` is preserved, including when no files are staged.
    /// `ALTER TABLE` requires `false` and rejects staged data-file batches or manifest writes at
    /// commit; blind append requires `true`. Commit validates file actions staged after building
    /// without inferring or overriding the configured value.
    ///
    /// # Parameters
    ///
    /// - `engine`: Provides table-state reads needed during validation.
    /// - `committer`: Executes the eventual atomic commit.
    ///
    /// For clustered tables, building performs log replay to load clustering columns from domain
    /// metadata and may therefore incur additional I/O.
    ///
    /// # Errors
    ///
    /// Returns an error for:
    /// - Invalid operation intent, including incompatible blind-append or schema intent.
    /// - Unsupported table writes, schema evolution, or CDF validation failures.
    /// - Conflicting application ids or domain metadata, or reserved or unsupported domains.
    /// - Connector commit information that does not contain exactly one row.
    ///
    /// Schema changes are rejected when `icebergCompatV2`, `icebergCompatV3`, or
    /// `allowColumnDefaults` is enabled, or when the table has generated columns. This applies to
    /// standalone ALTER TABLE transactions and schema evolution during data-changing writes.
    pub fn build(
        self,
        engine: &dyn Engine,
        committer: Box<dyn Committer>,
    ) -> KernelResult<Transaction<UpdateTable, MODE>> {
        self.validate()?;
        let transaction = try_new_update_table(self, committer, engine)?;
        transaction.validate_domain_metadata_operations()?;
        Ok(transaction)
    }

    fn validate(&self) -> KernelResult<()> {
        let operation = &self.local_state.operation;
        operation
            .validate()
            .map_err(KernelError::invalid_transaction_state)?;
        if operation == &UpdateTableOperation::AlterTable {
            if self.local_state.schema_changes.is_none()
                && self.local_state.updated_schema.is_none()
                && self.local_state.properties_added.is_empty()
                && self.local_state.property_keys_removed.is_empty()
            {
                return Err(KernelError::invalid_transaction_state(
                    "ALTER TABLE requires at least one metadata change",
                ));
            }
            if self.local_state.is_blind_append {
                return Err(KernelError::invalid_transaction_state(
                    "ALTER TABLE cannot be marked as a blind append",
                ));
            }
            if self.common_state.data_change {
                return Err(KernelError::invalid_transaction_state(
                    "ALTER TABLE requires data_change to be false",
                ));
            }
        }
        if self.local_state.is_blind_append && !self.common_state.data_change {
            return Err(KernelError::invalid_transaction_state(
                "blind append requires data_change to be true",
            ));
        }
        let has_updated_schema = self.local_state.updated_schema.is_some();
        if self.local_state.is_blind_append
            && (self.local_state.schema_changes.is_some() || has_updated_schema)
        {
            return Err(KernelError::invalid_transaction_state(
                "blind append cannot include schema changes",
            ));
        }
        // Incremental operations would have to be resolved against either the snapshot schema or
        // the replacement schema, so the two forms are kept exclusive.
        if has_updated_schema && self.local_state.schema_changes.is_some() {
            return Err(KernelError::invalid_transaction_state(
                "with_updated_schema cannot be combined with add_column, add_column_at, or \
                 set_nullable in one transaction",
            ));
        }

        let removals = validate_unique_domains(
            self.local_state
                .domain_metadata_removals
                .iter()
                .map(String::as_str),
            |domain| {
                KernelError::invalid_transaction_state(format!(
                    "domain metadata '{domain}' is removed more than once"
                ))
            },
        )?;
        for metadata in &self.common_state.domain_metadata_additions {
            let domain = metadata.domain();
            if removals.contains(domain) {
                return Err(KernelError::invalid_transaction_state(format!(
                    "domain metadata '{domain}' cannot be added and removed in one transaction"
                )));
            }
        }
        // Duplicate additions are validated by the shared builder state.
        self.common_state.validate()?;
        Ok(())
    }
}

fn try_new_update_table<MODE: ExecutionMode>(
    builder: UpdateTableTransactionBuilder<MODE>,
    committer: Box<dyn Committer>,
    engine: &dyn Engine,
) -> KernelResult<Transaction<UpdateTable, MODE>> {
    let TransactionBuilder {
        common_state,
        local_state,
        _mode: _,
    } = builder;
    let UpdateTableBuilderState {
        snapshot: read_snapshot,
        operation,
        schema_changes,
        updated_schema,
        properties_added,
        property_keys_removed,
        domain_metadata_removals,
        is_blind_append,
    } = local_state;

    // important! before writing to the table we must check it is supported
    read_snapshot
        .table_configuration()
        .ensure_operation_supported(Operation::Write)?;

    // TODO(#3223): Strip generation expressions on metadata updates when the pre-update protocol
    // did not support generated columns, so a protocol upgrade cannot activate previously ignored
    // formulas.

    // TODO(#3240): Validate that delta.enableRowTracking=true has the required protocol support
    // and that the materialized column names are distinct.

    // Read clustering columns from snapshot (returns None if clustering not enabled)
    let clustering_columns = read_snapshot
        .reader()
        .get_physical_clustering_columns(engine)?;

    let commit_timestamp = current_time_ms()?;

    let span = tracing::info_span!(
        "txn",
        path = %read_snapshot.table_root(),
        read_version = read_snapshot.version(),
    );

    let read_table_config = read_snapshot.table_configuration();
    validate_iceberg_compat_if_needed(
        read_table_config,
        &V2_VALIDATOR,
        IcebergCompatValidationContext::Write,
    )?;
    validate_iceberg_compat_if_needed(
        read_table_config,
        &V3_VALIDATOR,
        IcebergCompatValidationContext::Write,
    )?;

    let has_updated_schema = updated_schema.is_some();
    let request = UpdateTableMetadataRequest {
        schema_changes,
        updated_schema,
        properties_added,
        property_keys_removed,
    };
    let has_metadata_request = !request.is_empty();
    let PreparedUpdateTableMetadata {
        table_config: effective_table_config,
        should_emit_metadata,
        should_emit_protocol,
    } = prepare_update_table_metadata(&read_snapshot, request, commit_timestamp, engine)?;
    // A schema update keeps the snapshot's clustering columns, so each must still resolve.
    // Replacing or clearing them in the same transaction requires a clustering-column request,
    // which would supply the columns validated here.
    if let Some(columns) = clustering_columns.as_deref().filter(|_| has_updated_schema) {
        validate_clustering_columns_retained(read_table_config, &effective_table_config, columns)?;
    }

    Ok(Transaction {
        read_snapshot_opt: Some(read_snapshot),
        should_emit_metadata,
        should_emit_protocol,
        has_metadata_request,
        user_domain_removals: domain_metadata_removals,
        is_blind_append,
        physical_clustering_columns: clustering_columns,
        ..Transaction::new(
            common_state,
            effective_table_config,
            committer,
            operation.into(),
            span,
            commit_timestamp,
        )
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use rstest_reuse::{apply, template};
    use tempfile::TempDir;

    use crate::arrow::array::{ArrayRef, Int32Array, StringArray};
    use crate::arrow::datatypes::Schema as ArrowSchema;
    use crate::arrow::record_batch::RecordBatch;
    use crate::committer::FileSystemCommitter;
    use crate::engine::arrow_conversion::TryIntoArrow;
    use crate::engine::arrow_data::ArrowEngineData;
    use crate::engine::sync::SyncEngine;
    #[cfg(feature = "adaptive-metadata-in-dev")]
    use crate::expressions::ColumnName;
    use crate::expressions::column_name;
    use crate::object_store::memory::InMemory;
    use crate::schema::{
        ArrayType, ColumnMetadataKey, DataType, MapType, MetadataValue, StructField, StructType,
        schema_ref,
    };
    use crate::snapshot::Snapshot;
    use crate::table_configuration::InCommitTimestampEnablement;
    use crate::table_features::{ColumnMappingMode, TableFeature};
    use crate::table_properties::{
        COLUMN_MAPPING_MODE, ENABLE_ICEBERG_COMPAT_V2, ENABLE_ICEBERG_COMPAT_V3,
        ENABLE_IN_COMMIT_TIMESTAMPS, ENABLE_ROW_TRACKING,
    };
    use crate::transaction::create_table::create_table;
    use crate::transaction::data_layout::DataLayout;
    use crate::transaction::{SchemaOperation, UpdateTableOperation};
    use crate::unit_test_utils::{copy_test_table, load_test_table, test_schema_flat};
    use crate::{Engine, KernelError, Result, SnapshotRef};

    #[rstest]
    #[case::write(UpdateTableOperation::Write)]
    #[case::delete(UpdateTableOperation::Delete)]
    #[case::custom(UpdateTableOperation::Custom("vendor.write".to_string()))]
    fn builder_collects_existing_table_transaction_intent(
        #[case] operation: UpdateTableOperation,
    ) -> Result<()> {
        let (engine, snapshot, _tempdir) = load_test_table("table-without-dv-small")?;
        let transaction = snapshot
            .update_table(
                operation.clone(),
                "test-engine",
                operation != UpdateTableOperation::AlterTable,
            )
            .with_operation_parameters([("mode", Some("Append"))])
            .with_operation_metrics([("numFiles", Some("3"))])
            .with_transaction_id("app", 7)
            .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?;

        assert_eq!(transaction.operation, operation.into());
        assert_eq!(transaction.engine_info, "test-engine");
        assert_eq!(
            transaction.operation_parameters.as_ref().unwrap()["mode"].as_deref(),
            Some("Append")
        );
        assert_eq!(
            transaction.operation_metrics.as_ref().unwrap()["numFiles"].as_deref(),
            Some("3")
        );
        assert_eq!(transaction.set_transactions[0].app_id, "app");
        assert_eq!(transaction.set_transactions[0].version, 7);
        assert_eq!(
            transaction.set_transactions[0].last_updated,
            Some(transaction.commit_timestamp)
        );
        assert!(transaction.data_change);
        Ok(())
    }

    #[test]
    fn replacement_setters_use_the_last_value_and_empty_correlation_clears() -> Result<()> {
        let (engine, snapshot, _tempdir) = load_test_table("table-without-dv-small")?;
        let transaction = snapshot
            .update_table(UpdateTableOperation::Delete, "second-engine", true)
            .with_correlation_id("first-correlation")
            .with_correlation_id("")
            .with_operation_parameters([("first", Some("value"))])
            .with_operation_parameters([("second", Some("value"))])
            .with_operation_metrics([("first", Some("value"))])
            .with_operation_metrics([("second", Some("value"))])
            .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?;

        assert_eq!(transaction.operation, UpdateTableOperation::Delete.into());
        assert_eq!(transaction.engine_info, "second-engine");
        assert!(transaction.correlation_id.is_none());
        let parameters = transaction.operation_parameters.as_ref().unwrap();
        assert_eq!(parameters.len(), 1);
        assert_eq!(parameters["second"].as_deref(), Some("value"));
        let metrics = transaction.operation_metrics.as_ref().unwrap();
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics["second"].as_deref(), Some("value"));
        assert!(transaction.data_change);
        Ok(())
    }

    #[test]
    fn builder_debug_summarizes_intent() -> Result<()> {
        let (_engine, snapshot, _tempdir) = load_test_table("table-without-dv-small")?;
        let debug = format!(
            "{:?}",
            snapshot
                .update_table(UpdateTableOperation::Write, "test-engine", true)
                .add_column(StructField::nullable("new_column", DataType::STRING))
        );

        assert!(debug.contains("UpdateTableTransactionBuilder"));
        assert!(debug.contains("operation: Write"));
        assert!(debug.contains("schema_change_count: 1"));
        Ok(())
    }

    #[test]
    fn builder_applies_optional_transaction_intent() -> Result<()> {
        let (engine, snapshot, _tempdir) = load_test_table("table-without-dv-small")?;
        let commit_info_schema = schema_ref! { nullable "tag": STRING };
        let arrow_schema: ArrowSchema = commit_info_schema
            .as_ref()
            .try_into_arrow()
            .map_err(KernelError::from)?;
        let commit_info = Box::new(ArrowEngineData::new(
            RecordBatch::try_new(
                Arc::new(arrow_schema),
                vec![Arc::new(StringArray::from(vec![Some("value")]))],
            )
            .map_err(KernelError::from)?,
        ));
        let mut transaction = snapshot
            .update_table(UpdateTableOperation::Write, "test-engine", false)
            .with_commit_info(commit_info, commit_info_schema)
            .with_correlation_id("correlation-id")
            .with_schema_changes([SchemaOperation::add_column(
                None,
                StructField::nullable("new_column", DataType::STRING),
            )])
            .expect("non-empty schema changes must be accepted by the setter")
            .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?;
        transaction.ack_column_defaults();
        transaction.ack_row_tracking_preservation();
        transaction = transaction.with_row_tracking_high_water_mark(17)?;

        assert_eq!(
            transaction.correlation_id.as_deref(),
            Some("correlation-id")
        );
        assert!(!transaction.data_change);
        assert!(transaction.column_defaults_acknowledged);
        assert!(transaction.row_tracking_preservation_acknowledged);
        assert_eq!(transaction.provided_row_tracking_high_water_mark, Some(17));
        assert!(transaction.engine_commit_info.is_some());
        assert!(transaction.user_domain_removals.is_empty());
        Ok(())
    }

    #[rstest]
    #[case(0)]
    #[case(2)]
    fn builder_rejects_non_single_row_commit_info(#[case] row_count: usize) -> Result<()> {
        let (engine, snapshot, _tempdir) = load_test_table("table-without-dv-small")?;
        let commit_info_schema = schema_ref! { nullable "tag": STRING };
        let arrow_schema: ArrowSchema = commit_info_schema
            .as_ref()
            .try_into_arrow()
            .map_err(KernelError::from)?;
        let commit_info = Box::new(ArrowEngineData::new(
            RecordBatch::try_new(
                Arc::new(arrow_schema),
                vec![Arc::new(StringArray::from(vec![Some("value"); row_count]))],
            )
            .map_err(KernelError::from)?,
        ));

        let error = snapshot
            .update_table(UpdateTableOperation::Write, "test-engine", true)
            .with_commit_info(commit_info, commit_info_schema)
            .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Connector commit info must contain exactly one row"),
            "{error}"
        );
        Ok(())
    }

    #[cfg(feature = "adaptive-metadata-in-dev")]
    #[test]
    fn builder_collects_nested_schema_intent() -> Result<()> {
        let (_engine, snapshot, _tempdir) = load_test_table("table-without-dv-small")?;
        let builder = snapshot
            .update_table(UpdateTableOperation::AlterTable, "test-engine", false)
            .add_column_at(
                ColumnName::new(["parent"]),
                StructField::nullable("child", DataType::STRING),
            );

        assert_eq!(
            builder.local_state.schema_changes.as_ref().unwrap().len(),
            1
        );
        Ok(())
    }

    #[test]
    fn alter_table_operation_uses_unified_builder() -> Result<()> {
        let (engine, snapshot, _tempdir) = load_test_table("table-without-dv-small")?;
        let transaction = snapshot
            .update_table(UpdateTableOperation::AlterTable, "test-engine", false)
            .add_column(StructField::nullable("new_column", DataType::STRING))
            .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?;

        assert_eq!(
            transaction.operation,
            UpdateTableOperation::AlterTable.into()
        );
        assert!(transaction.should_emit_metadata);
        assert!(
            transaction
                .effective_table_config
                .logical_schema()
                .contains("new_column")
        );
        Ok(())
    }

    #[test]
    fn custom_operation_with_schema_change_uses_behavioral_validation() -> Result<()> {
        let (engine, snapshot, _tempdir) = load_test_table("table-without-dv-small")?;
        let transaction = snapshot
            .update_table(
                UpdateTableOperation::Custom("vendor schema write".to_string()),
                "test-engine",
                true,
            )
            .add_column(StructField::nullable("new_column", DataType::STRING))
            .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?;

        assert!(transaction.should_emit_metadata);
        Ok(())
    }

    #[rstest]
    fn custom_operation_rejects_reserved_known_name(
        #[values("ALTER TABLE", "REORG", "LIQUID TAGGING", "Deletion Vector Tombstones")]
        operation: &str,
        #[values(false, true)] data_change: bool,
    ) -> Result<()> {
        let (engine, snapshot, _tempdir) = load_test_table("table-without-dv-small")?;
        let error = snapshot
            .update_table(
                UpdateTableOperation::Custom(operation.to_string()),
                "test-engine",
                data_change,
            )
            .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))
            .unwrap_err();

        assert!(
            matches!(&error, KernelError::InvalidTransactionState(message)
            if message == &format!("custom operation name '{operation}' is reserved"))
        );
        Ok(())
    }

    #[test]
    fn alter_table_requires_schema_changes() -> Result<()> {
        let (engine, snapshot, _tempdir) = load_test_table("table-without-dv-small")?;
        let error = snapshot
            .update_table(UpdateTableOperation::AlterTable, "test-engine", false)
            .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("requires at least one metadata change")
        );
        Ok(())
    }

    #[test]
    fn empty_schema_changes_are_rejected() {
        let (_engine, snapshot, _tempdir) =
            load_test_table("table-without-dv-small").expect("load test table");
        let error = snapshot
            .update_table(UpdateTableOperation::Write, "test-engine", true)
            .with_schema_changes(std::iter::from_fn(|| None::<SchemaOperation>))
            .unwrap_err();

        assert!(matches!(&error, KernelError::Generic(message)
            if message == "with_schema_changes requires at least one schema operation"));
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum NonemptySchemaInput {
        AddColumn,
        AlreadyNullable,
    }

    impl NonemptySchemaInput {
        fn changes(self) -> Vec<SchemaOperation> {
            match self {
                Self::AddColumn => vec![SchemaOperation::add_column(
                    None,
                    StructField::nullable("added", DataType::STRING),
                )],
                Self::AlreadyNullable => vec![SchemaOperation::SetNullable {
                    column: column_name!("id"),
                }],
            }
        }
    }

    type SchemaValidationTable = (Arc<dyn Engine>, SnapshotRef, Option<TempDir>);

    #[template]
    #[rstest]
    #[case::plain(None, false)]
    #[case::v2_compatibility_property_true(Some(TableFeature::IcebergCompatV2), true)]
    #[case::v2_compatibility_property_false(Some(TableFeature::IcebergCompatV2), false)]
    #[case::v3_compatibility_property_true(Some(TableFeature::IcebergCompatV3), true)]
    #[case::v3_compatibility_property_false(Some(TableFeature::IcebergCompatV3), false)]
    #[case::column_defaults(Some(TableFeature::AllowColumnDefaults), true)]
    fn schema_validation_tables(
        #[case] feature: Option<TableFeature>,
        #[case] feature_enabled: bool,
    ) {
    }

    #[template]
    #[apply(schema_validation_tables)]
    fn schema_validation_tables_and_operations(
        feature: Option<TableFeature>,
        feature_enabled: bool,
        #[values(
            UpdateTableOperation::Write,
            UpdateTableOperation::StreamingUpdate,
            UpdateTableOperation::AlterTable,
            UpdateTableOperation::Delete,
            UpdateTableOperation::Update,
            UpdateTableOperation::Merge,
            UpdateTableOperation::Optimize,
            UpdateTableOperation::Reorg,
            UpdateTableOperation::Custom("UNKNOWN".to_string())
        )]
        operation: UpdateTableOperation,
    ) {
    }

    #[apply(schema_validation_tables_and_operations)]
    fn empty_schema_changes_are_rejected_for_all_table_operations(
        feature: Option<TableFeature>,
        feature_enabled: bool,
        operation: UpdateTableOperation,
    ) {
        let (_engine, snapshot, _tempdir) =
            create_schema_validation_table(feature.as_ref(), feature_enabled);
        let error = snapshot
            .update_table(operation, "test-engine", true)
            .with_schema_changes([])
            .unwrap_err();
        assert!(matches!(&error, KernelError::Generic(message)
            if message == "with_schema_changes requires at least one schema operation"));
    }

    #[apply(schema_validation_tables_and_operations)]
    fn omitted_schema_changes_require_a_non_alter_operation(
        feature: Option<TableFeature>,
        feature_enabled: bool,
        operation: UpdateTableOperation,
    ) {
        let (engine, snapshot, _tempdir) =
            create_schema_validation_table(feature.as_ref(), feature_enabled);
        let builder = snapshot
            .clone()
            .update_table(operation.clone(), "test-engine", true);
        assert!(builder.local_state.schema_changes.is_none());
        let result = builder.build(engine.as_ref(), Box::new(FileSystemCommitter::new()));
        if operation == UpdateTableOperation::AlterTable {
            let error = result.unwrap_err();
            assert!(
                matches!(&error, KernelError::InvalidTransactionState(message)
                if message == "ALTER TABLE requires at least one metadata change")
            );
        } else {
            let transaction = result.expect("non-ALTER transactions do not require schema changes");
            assert_eq!(transaction.operation, operation.into());
            assert!(!transaction.should_emit_metadata);
            assert!(!transaction.should_emit_protocol);
            assert_eq!(
                transaction.effective_table_config.logical_schema(),
                snapshot.schema()
            );
        }
    }

    #[apply(schema_validation_tables_and_operations)]
    fn nonempty_schema_changes_respect_table_feature_restrictions(
        feature: Option<TableFeature>,
        feature_enabled: bool,
        operation: UpdateTableOperation,
        #[values(NonemptySchemaInput::AddColumn, NonemptySchemaInput::AlreadyNullable)]
        input: NonemptySchemaInput,
    ) {
        let (engine, snapshot, _tempdir) =
            create_schema_validation_table(feature.as_ref(), feature_enabled);
        let builder = snapshot
            .clone()
            .update_table(
                operation.clone(),
                "test-engine",
                operation != UpdateTableOperation::AlterTable,
            )
            .with_schema_changes(input.changes())
            .expect("non-empty schema changes must be accepted by the setter");
        assert_eq!(
            builder.local_state.schema_changes.as_ref().unwrap().len(),
            1
        );
        let result = builder.build(engine.as_ref(), Box::new(FileSystemCommitter::new()));
        if let Some(feature) = feature.filter(|_| feature_enabled) {
            assert!(
                snapshot
                    .table_configuration()
                    .is_feature_supported(&feature)
            );
            assert!(snapshot.table_configuration().is_feature_enabled(&feature));
            let error = result.unwrap_err();
            let expected =
                format!("Schema changes are not yet supported on tables with {feature} enabled");
            assert!(matches!(&error, KernelError::Unsupported(message) if message == &expected));
        } else {
            let transaction = result.expect("supported schema changes must build successfully");
            assert_eq!(transaction.operation, operation.into());
            assert!(transaction.should_emit_metadata);
            assert!(!transaction.should_emit_protocol);
            let schema = transaction.effective_table_config.logical_schema();
            if input == NonemptySchemaInput::AddColumn {
                assert_eq!(
                    schema.field("added").unwrap().data_type(),
                    &DataType::STRING
                );
                assert!(schema.field("added").unwrap().is_nullable());
                assert_eq!(schema.num_fields(), snapshot.schema().num_fields() + 1);
            } else {
                assert_eq!(schema, snapshot.schema());
            }
        }
    }

    #[apply(schema_validation_tables)]
    fn empty_schema_changes_are_rejected_after_queued_changes(
        feature: Option<TableFeature>,
        feature_enabled: bool,
        #[values(false, true)] individual_setter: bool,
        #[values(NonemptySchemaInput::AddColumn, NonemptySchemaInput::AlreadyNullable)]
        input: NonemptySchemaInput,
    ) {
        let (_engine, snapshot, _tempdir) =
            create_schema_validation_table(feature.as_ref(), feature_enabled);
        let builder = snapshot.update_table(UpdateTableOperation::Write, "test-engine", true);
        let builder = match (individual_setter, input) {
            (true, NonemptySchemaInput::AddColumn) => {
                builder.add_column(StructField::nullable("added", DataType::STRING))
            }
            (true, NonemptySchemaInput::AlreadyNullable) => {
                builder.set_nullable(column_name!("id"))
            }
            _ => builder
                .with_schema_changes(input.changes())
                .expect("non-empty schema changes must be accepted by the setter"),
        };
        assert_eq!(
            builder.local_state.schema_changes.as_ref().unwrap().len(),
            1
        );
        let error = builder.with_schema_changes([]).unwrap_err();
        assert!(matches!(&error, KernelError::Generic(message)
            if message == "with_schema_changes requires at least one schema operation"));
    }

    #[test]
    fn nonempty_schema_change_calls_preserve_iterator_items_and_call_order() {
        let (engine, snapshot, _tempdir) =
            load_test_table("table-without-dv-small").expect("load test table");
        let mut changes = [SchemaOperation::add_column(
            None,
            StructField::nullable("new_column", DataType::STRING),
        )]
        .into_iter();
        let builder = snapshot
            .update_table(UpdateTableOperation::Write, "test-engine", true)
            .with_schema_changes(std::iter::from_fn(move || changes.next()))
            .expect("non-empty iterator must be accepted")
            .with_schema_changes([SchemaOperation::SetNullable {
                column: column_name!("new_column"),
            }])
            .expect("subsequent non-empty schema changes must be accepted")
            .add_column(StructField::nullable("another_column", DataType::INTEGER));

        let queued_changes = builder.local_state.schema_changes.as_deref().unwrap();
        assert!(matches!(
            queued_changes,
            [
                SchemaOperation::AddColumn { parent: None, field: first_column },
                SchemaOperation::SetNullable { column },
                SchemaOperation::AddColumn { parent: None, field: second_column },
            ] if first_column == &StructField::nullable("new_column", DataType::STRING)
                && column == &column_name!("new_column")
                && second_column == &StructField::nullable("another_column", DataType::INTEGER)
        ));
        let transaction = builder
            .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))
            .expect("schema changes must build in call order");
        assert!(transaction.should_emit_metadata);
        let schema = transaction.effective_table_config.logical_schema();
        assert!(schema.contains("new_column"));
        assert!(schema.contains("another_column"));
    }

    #[derive(Debug)]
    enum InvalidIntent {
        AlterTableBlindAppend,
        BlindAppendWithoutDataChange,
        BlindAppendWithSchemaChange,
        BlindAppendWithUpdatedSchema,
        AlterTableWithDataChange,
    }

    #[rstest]
    #[case::alter_blind_append(
        InvalidIntent::AlterTableBlindAppend,
        "cannot be marked as a blind append"
    )]
    #[case::blind_append_without_data_change(
        InvalidIntent::BlindAppendWithoutDataChange,
        "requires data_change to be true"
    )]
    #[case::blind_append_with_schema_change(
        InvalidIntent::BlindAppendWithSchemaChange,
        "cannot include schema changes"
    )]
    #[case::blind_append_with_updated_schema(
        InvalidIntent::BlindAppendWithUpdatedSchema,
        "cannot include schema changes"
    )]
    #[case::alter_with_data_change(
        InvalidIntent::AlterTableWithDataChange,
        "ALTER TABLE requires data_change to be false"
    )]
    fn invalid_intent_is_rejected(
        #[case] intent: InvalidIntent,
        #[case] expected: &str,
    ) -> Result<()> {
        let (engine, snapshot, _tempdir) = load_test_table("table-without-dv-small")?;
        let operation = match intent {
            InvalidIntent::AlterTableBlindAppend | InvalidIntent::AlterTableWithDataChange => {
                UpdateTableOperation::AlterTable
            }
            _ => UpdateTableOperation::Write,
        };
        let data_change = !matches!(intent, InvalidIntent::BlindAppendWithoutDataChange);
        let snapshot_schema = snapshot.schema();
        let builder = snapshot.update_table(operation, "test-engine", data_change);
        let builder = match intent {
            InvalidIntent::AlterTableBlindAppend => builder
                .add_column(StructField::nullable("new_column", DataType::STRING))
                .with_blind_append(),
            InvalidIntent::BlindAppendWithoutDataChange => builder.with_blind_append(),
            InvalidIntent::BlindAppendWithSchemaChange => builder
                .add_column(StructField::nullable("new_column", DataType::STRING))
                .with_blind_append(),
            InvalidIntent::BlindAppendWithUpdatedSchema => builder
                .with_updated_schema(snapshot_schema.clone())
                .with_blind_append(),
            InvalidIntent::AlterTableWithDataChange => {
                builder.add_column(StructField::nullable("new_column", DataType::STRING))
            }
        };

        let error = builder
            .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))
            .unwrap_err();
        assert!(matches!(&error, KernelError::InvalidTransactionState(_)));
        assert!(error.to_string().contains(expected), "{error}");
        Ok(())
    }

    #[derive(Clone, Copy, Debug)]
    enum StagedFileAction {
        Add,
        Remove,
        DeletionVectorUpdate,
    }

    #[rstest]
    #[case::add(StagedFileAction::Add, &[true])]
    #[case::empty_add(StagedFileAction::Add, &[])]
    #[case::remove(StagedFileAction::Remove, &[true])]
    #[case::empty_remove(StagedFileAction::Remove, &[])]
    #[case::unselected_remove(StagedFileAction::Remove, &[false])]
    #[case::dv_update(StagedFileAction::DeletionVectorUpdate, &[true])]
    #[case::empty_dv_update(StagedFileAction::DeletionVectorUpdate, &[])]
    #[case::unselected_dv_update(StagedFileAction::DeletionVectorUpdate, &[false])]
    fn alter_table_commit_rejects_staged_file_batches(
        #[case] action: StagedFileAction,
        #[case] selection_vector: &[bool],
    ) -> Result<()> {
        let (engine, snapshot, _tempdir) = load_test_table("table-without-dv-small")?;
        let mut transaction = snapshot
            .update_table(UpdateTableOperation::AlterTable, "test-engine", false)
            .add_column(StructField::nullable("new_column", DataType::STRING))
            .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?;

        assert!(!transaction.data_change);

        let data = || {
            let values = Arc::new(Int32Array::from(vec![1; selection_vector.len()])) as ArrayRef;
            Box::new(ArrowEngineData::new(
                RecordBatch::try_from_iter([("value", values)]).unwrap(),
            ))
        };
        match action {
            StagedFileAction::Add => transaction.add_files(data()),
            StagedFileAction::Remove | StagedFileAction::DeletionVectorUpdate => {
                let data = crate::FilteredEngineData::try_new(data(), selection_vector.to_vec())?;
                match action {
                    StagedFileAction::Remove => transaction.remove_files(data),
                    StagedFileAction::DeletionVectorUpdate => {
                        transaction.staged_data_changes.dv_matched_files.push(data);
                        transaction.staged_data_changes.num_dv_updates = selection_vector
                            .iter()
                            .filter(|&&selected| selected)
                            .count();
                    }
                    StagedFileAction::Add => unreachable!(),
                }
            }
        }
        assert!(!transaction.data_change);
        let error = transaction.commit(engine.as_ref()).unwrap_err();
        assert!(
            matches!(&error, KernelError::InvalidTransactionState(message)
            if message == "ALTER TABLE cannot include data-file actions")
        );
        Ok(())
    }

    #[rstest]
    fn alter_table_build_requires_data_change_false(
        #[values(false, true)] data_change: bool,
    ) -> Result<()> {
        let (url, _tempdir) = copy_test_table("table-without-dv-small")?;
        let engine = SyncEngine::new();
        let snapshot = Snapshot::builder_for(url).build(&engine)?;
        let version = snapshot.version();
        let builder = snapshot
            .update_table(UpdateTableOperation::AlterTable, "test-engine", data_change)
            .add_column(StructField::nullable("new_column", DataType::STRING));
        let result = builder.build(&engine, Box::new(FileSystemCommitter::new()));
        if data_change {
            let error = result.unwrap_err();
            assert!(
                matches!(&error, KernelError::InvalidTransactionState(message)
                if message == "ALTER TABLE requires data_change to be false")
            );
        } else {
            let transaction = result?;
            assert!(!transaction.data_change);
            let snapshot = transaction.commit(&engine)?.unwrap_post_commit_snapshot();
            assert_eq!(snapshot.version(), version + 1);
            assert!(snapshot.schema().contains("new_column"));
        }
        Ok(())
    }

    #[test]
    fn alter_table_commit_rejects_invalid_internal_data_change() -> Result<()> {
        let (engine, snapshot, _tempdir) = load_test_table("table-without-dv-small")?;
        let mut transaction = snapshot
            .update_table(UpdateTableOperation::AlterTable, "test-engine", false)
            .add_column(StructField::nullable("new_column", DataType::STRING))
            .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?;
        transaction.data_change = true;
        let error = transaction.commit(engine.as_ref()).unwrap_err();
        assert!(
            matches!(&error, KernelError::InvalidTransactionState(message)
            if message == "ALTER TABLE requires data_change to be false")
        );
        Ok(())
    }

    #[rstest]
    fn custom_schema_changes_preserve_explicit_data_change(
        #[values(false, true)] data_change: bool,
    ) -> Result<()> {
        let (engine, snapshot, _tempdir) = load_test_table("table-without-dv-small")?;
        let mut transaction = snapshot
            .update_table(
                UpdateTableOperation::Custom("CUSTOM".to_string()),
                "test-engine",
                data_change,
            )
            .add_column(StructField::nullable("new_column", DataType::STRING))
            .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?;

        assert_eq!(transaction.data_change, data_change);

        let values = Arc::new(Int32Array::from_iter_values([1])) as ArrayRef;
        transaction
            .staged_data_changes
            .add_files_metadata
            .push(Box::new(ArrowEngineData::new(
                RecordBatch::try_from_iter([("value", values)]).map_err(KernelError::from)?,
            )));
        assert_eq!(transaction.data_change, data_change);
        Ok(())
    }

    #[rstest]
    #[case::duplicate_app_id(&[], &[], &["app", "app"], "app_id app appears more than once")]
    #[case::duplicate_domain(
        &["domain", "domain"], &[], &[], "domain metadata 'domain' appears more than once"
    )]
    #[case::added_and_removed_domain(
        &["domain"], &["domain"], &[], "cannot be added and removed"
    )]
    #[case::duplicate_removals(
        &[], &["domain", "domain"], &[], "removed more than once"
    )]
    #[case::duplicate_removal_precedes_add_remove_conflict(
        &["domain"], &["domain", "domain"], &[], "removed more than once"
    )]
    #[case::add_remove_conflict_precedes_duplicate_addition(
        &["domain", "domain"], &["domain"], &[], "cannot be added and removed"
    )]
    #[case::duplicate_app_id_precedes_duplicate_addition(
        &["domain", "domain"], &[], &["app", "app"], "app_id app appears more than once"
    )]
    fn duplicate_builder_values_are_rejected(
        #[case] additions: &[&str],
        #[case] removals: &[&str],
        #[case] app_ids: &[&str],
        #[case] expected: &str,
    ) -> Result<()> {
        let (engine, snapshot, _tempdir) = load_test_table("table-without-dv-small")?;
        let mut builder = snapshot.update_table(UpdateTableOperation::Write, "test-engine", true);
        for domain in additions {
            builder = builder.with_domain_metadata(*domain, "value");
        }
        for domain in removals {
            builder = builder.with_domain_metadata_removed(*domain);
        }
        for app_id in app_ids {
            builder = builder.with_transaction_id(*app_id, 1);
        }

        let error = builder
            .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))
            .unwrap_err();
        assert!(matches!(&error, KernelError::InvalidTransactionState(_)));
        assert!(error.to_string().contains(expected), "{error}");
        Ok(())
    }

    #[test]
    fn domain_metadata_without_table_feature_is_rejected_during_build() -> Result<()> {
        let (engine, snapshot, _tempdir) = load_test_table("table-without-dv-small")?;
        let error = snapshot
            .update_table(UpdateTableOperation::Write, "test-engine", true)
            .with_domain_metadata("app.config", "{}")
            .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))
            .unwrap_err();

        assert!(error.to_string().contains("domainMetadata"), "{error}");
        Ok(())
    }

    #[test]
    fn schema_only_change_derives_protocol_features() -> Result<()> {
        let (engine, snapshot, _tempdir) = load_test_table("table-without-dv-small")?;
        let transaction = snapshot
            .update_table(UpdateTableOperation::AlterTable, "test-engine", false)
            .add_column(StructField::nullable("ts", DataType::TIMESTAMP_NTZ))
            .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?;

        assert!(transaction.should_emit_metadata);
        assert!(transaction.should_emit_protocol);
        let config = &transaction.effective_table_config;
        assert!(config.is_feature_supported(&TableFeature::TimestampWithoutTimezone));
        assert!(config.is_feature_supported(&TableFeature::AppendOnly));
        assert!(config.is_feature_supported(&TableFeature::Invariants));
        Ok(())
    }

    #[test]
    fn enabling_in_commit_timestamps_commits_recorded_enablement() -> Result<()> {
        let engine = SyncEngine::new_with_store(Arc::new(InMemory::new()));
        let snapshot = create_table("memory:///", test_schema_flat(), "test", true)
            .build(&engine, Box::new(FileSystemCommitter::new()))?
            .commit(&engine)?
            .unwrap_post_commit_snapshot();
        let mut builder =
            snapshot.update_table(UpdateTableOperation::AlterTable, "test-engine", false);
        builder
            .local_state
            .properties_added
            .insert(ENABLE_IN_COMMIT_TIMESTAMPS.to_string(), "true".to_string());
        let transaction = builder.build(&engine, Box::new(FileSystemCommitter::new()))?;
        assert!(transaction.should_emit_protocol);
        let committed = transaction.commit(&engine)?.unwrap_post_commit_snapshot();

        let reloaded = Snapshot::builder_for("memory:///").build(&engine)?;
        assert_eq!(reloaded.version(), committed.version());
        let InCommitTimestampEnablement::Enabled {
            enablement: Some((version, timestamp)),
        } = reloaded
            .table_configuration()
            .in_commit_timestamp_enablement()?
        else {
            panic!("expected recorded enablement metadata");
        };
        assert_eq!(version, reloaded.version());
        assert_eq!(
            reloaded.reader().get_in_commit_timestamp(&engine)?,
            Some(timestamp)
        );
        Ok(())
    }

    /// Creates an in-memory column-mapping table clustered by `name`.
    fn create_clustered_column_mapping_table() -> Result<(SyncEngine, SnapshotRef)> {
        let engine = SyncEngine::new_with_store(Arc::new(InMemory::new()));
        let snapshot = create_table("memory:///", test_schema_flat(), "test", true)
            .with_table_properties([("delta.columnMapping.mode", "name")])
            .with_data_layout(DataLayout::clustered(["name"]))
            .build(&engine, Box::new(FileSystemCommitter::new()))?
            .commit(&engine)?
            .unwrap_post_commit_snapshot();
        Ok((engine, snapshot))
    }

    #[test]
    fn updated_schema_rejects_dropping_a_clustering_column() -> Result<()> {
        let (engine, snapshot) = create_clustered_column_mapping_table()?;
        let schema = snapshot.schema();
        let without_name = StructType::try_new([schema.field("id").unwrap().clone()])?;
        let error = snapshot
            .clone()
            .update_table(UpdateTableOperation::AlterTable, "test-engine", false)
            .with_updated_schema(Arc::new(without_name))
            .build(&engine, Box::new(FileSystemCommitter::new()))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Cannot drop clustering column 'name'"),
            "{error}"
        );
        let reloaded = Snapshot::builder_for("memory:///").build(&engine)?;
        assert_eq!(reloaded.version(), snapshot.version());
        Ok(())
    }

    #[rstest]
    fn updated_schema_handles_a_nested_clustering_column(
        #[values(false, true)] drop: bool,
    ) -> Result<()> {
        let engine = SyncEngine::new_with_store(Arc::new(InMemory::new()));
        let address = StructType::try_new([
            StructField::nullable("city", DataType::STRING),
            StructField::nullable("zip", DataType::STRING),
        ])?;
        let schema = Arc::new(StructType::try_new([
            StructField::nullable("id", DataType::INTEGER),
            StructField::nullable("address", address),
        ])?);
        let snapshot = create_table("memory:///", schema, "test", true)
            .with_table_properties([(COLUMN_MAPPING_MODE, "name")])
            .with_data_layout(DataLayout::Clustered {
                columns: vec![column_name!("address.city")],
            })
            .build(&engine, Box::new(FileSystemCommitter::new()))?
            .commit(&engine)?
            .unwrap_post_commit_snapshot();
        let schema = snapshot.schema();
        let DataType::Struct(address) = schema.field("address").unwrap().data_type() else {
            panic!("address is a struct");
        };
        let city = address.field("city").unwrap();
        let zip = address.field("zip").unwrap().clone();
        let address_fields = if drop {
            vec![zip]
        } else {
            vec![city.with_name("town"), zip]
        };
        let mut new_address = schema.field("address").unwrap().clone();
        new_address.data_type = StructType::try_new(address_fields)?.into();
        let updated = StructType::try_new([schema.field("id").unwrap().clone(), new_address])?;

        let result = snapshot
            .clone()
            .update_table(UpdateTableOperation::AlterTable, "test-engine", false)
            .with_updated_schema(Arc::new(updated))
            .build(&engine, Box::new(FileSystemCommitter::new()));
        if drop {
            let error = result.unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("Cannot drop clustering column 'address.city'"),
                "{error}"
            );
            let reloaded = Snapshot::builder_for("memory:///").build(&engine)?;
            assert_eq!(reloaded.version(), snapshot.version());
            return Ok(());
        }
        let committed = result?.commit(&engine)?.unwrap_post_commit_snapshot();
        let reader = committed.reader();
        assert_eq!(
            reader.get_physical_clustering_columns(&engine)?,
            snapshot.reader().get_physical_clustering_columns(&engine)?
        );
        let infos = reader.get_clustering_column_infos(&engine)?.unwrap();
        assert_eq!(infos[0].logical_column, column_name!("address.town"));
        Ok(())
    }

    #[rstest]
    fn type_widening_enabled_in_the_same_request_allows_widening(
        #[values(false, true)] enable_in_request: bool,
    ) -> Result<()> {
        let engine = SyncEngine::new_with_store(Arc::new(InMemory::new()));
        let schema = Arc::new(StructType::try_new([StructField::nullable(
            "i",
            DataType::INTEGER,
        )])?);
        let snapshot = create_table("memory:///", schema, "test", true)
            .with_table_properties([(COLUMN_MAPPING_MODE, "name")])
            .build(&engine, Box::new(FileSystemCommitter::new()))?
            .commit(&engine)?
            .unwrap_post_commit_snapshot();
        let mut widened = snapshot.schema().field("i").unwrap().clone();
        widened.data_type = DataType::LONG;
        let mut builder = snapshot
            .clone()
            .update_table(UpdateTableOperation::AlterTable, "test-engine", false)
            .with_updated_schema(Arc::new(StructType::try_new([widened])?));
        if enable_in_request {
            builder
                .local_state
                .properties_added
                .insert("delta.enableTypeWidening".to_string(), "true".to_string());
        }
        let result = builder.build(&engine, Box::new(FileSystemCommitter::new()));
        if !enable_in_request {
            let error = result.unwrap_err();
            assert!(
                error.to_string().contains("type widening is not enabled"),
                "{error}"
            );
            let reloaded = Snapshot::builder_for("memory:///").build(&engine)?;
            assert_eq!(reloaded.version(), snapshot.version());
            assert_eq!(reloaded.schema(), snapshot.schema());
            return Ok(());
        }
        result?.commit(&engine)?.unwrap_committed();

        let reloaded = Snapshot::builder_for("memory:///").build(&engine)?;
        let config = reloaded.table_configuration();
        assert!(config.is_feature_enabled(&TableFeature::TypeWidening));
        let field = reloaded.schema().field("i").unwrap().clone();
        assert_eq!(field.data_type(), &DataType::LONG);
        assert!(
            field
                .get_config_value(&ColumnMetadataKey::TypeChanges)
                .is_some()
        );
        Ok(())
    }

    #[test]
    fn unchanged_updated_schema_counts_as_a_request_and_commits() -> Result<()> {
        let (engine, snapshot) = create_clustered_column_mapping_table()?;
        let transaction = snapshot
            .clone()
            .update_table(UpdateTableOperation::AlterTable, "test-engine", false)
            .with_updated_schema(snapshot.schema())
            .build(&engine, Box::new(FileSystemCommitter::new()))?;
        assert!(transaction.has_metadata_request);
        transaction.commit(&engine)?.unwrap_committed();

        let reloaded = Snapshot::builder_for("memory:///").build(&engine)?;
        assert_eq!(reloaded.version(), snapshot.version() + 1);
        assert_eq!(reloaded.schema(), snapshot.schema());
        Ok(())
    }

    #[test]
    fn updated_schema_with_column_mapping_enablement_is_rejected() -> Result<()> {
        let engine = SyncEngine::new_with_store(Arc::new(InMemory::new()));
        let snapshot = create_table("memory:///", test_schema_flat(), "test", true)
            .build(&engine, Box::new(FileSystemCommitter::new()))?
            .commit(&engine)?
            .unwrap_post_commit_snapshot();
        let mut builder = snapshot
            .clone()
            .update_table(UpdateTableOperation::AlterTable, "test-engine", false)
            .with_updated_schema(snapshot.schema());
        builder
            .local_state
            .properties_added
            .insert(COLUMN_MAPPING_MODE.to_string(), "name".to_string());
        let error = builder
            .build(&engine, Box::new(FileSystemCommitter::new()))
            .unwrap_err();
        // The complete-schema path requires column mapping on the read table, so it rejects the
        // request before the rule against combining a mode change with schema changes runs.
        assert!(matches!(error, KernelError::Unsupported(_)), "{error}");
        assert!(
            error
                .to_string()
                .contains("requires column mapping to be enabled"),
            "{error}"
        );

        let reloaded = Snapshot::builder_for("memory:///").build(&engine)?;
        assert_eq!(reloaded.version(), snapshot.version());
        Ok(())
    }

    #[test]
    fn updated_schema_rename_keeps_clustering_column() -> Result<()> {
        let (engine, snapshot) = create_clustered_column_mapping_table()?;
        let schema = snapshot.schema();
        let renamed = StructType::try_new([
            schema.field("id").unwrap().clone(),
            schema.field("name").unwrap().with_name("label"),
        ])?;
        let committed = snapshot
            .clone()
            .update_table(UpdateTableOperation::AlterTable, "test-engine", false)
            .with_updated_schema(Arc::new(renamed))
            .build(&engine, Box::new(FileSystemCommitter::new()))?
            .commit(&engine)?
            .unwrap_post_commit_snapshot();

        let reader = committed.reader();
        assert_eq!(
            reader.get_physical_clustering_columns(&engine)?,
            snapshot.reader().get_physical_clustering_columns(&engine)?
        );
        let infos = reader.get_clustering_column_infos(&engine)?.unwrap();
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].logical_column, column_name!("label"));
        Ok(())
    }

    #[test]
    fn enabling_iceberg_compat_v2_with_supported_deletion_vectors_commits_nothing() -> Result<()> {
        let engine = SyncEngine::new_with_store(Arc::new(InMemory::new()));
        let snapshot = create_table("memory:///", test_schema_flat(), "test", true)
            .with_table_properties([
                (COLUMN_MAPPING_MODE, "name"),
                ("delta.feature.deletionVectors", "supported"),
            ])
            .build(&engine, Box::new(FileSystemCommitter::new()))?
            .commit(&engine)?
            .unwrap_post_commit_snapshot();
        let version = snapshot.version();
        let mut builder =
            snapshot.update_table(UpdateTableOperation::AlterTable, "test-engine", false);
        builder
            .local_state
            .properties_added
            .insert(ENABLE_ICEBERG_COMPAT_V2.to_string(), "true".to_string());
        let error = builder
            .build(&engine, Box::new(FileSystemCommitter::new()))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("'deletionVectors' is incompatible with 'icebergCompatV2'"),
            "{error}"
        );

        let reloaded = Snapshot::builder_for("memory:///").build(&engine)?;
        assert_eq!(reloaded.version(), version);
        assert!(
            !reloaded
                .table_configuration()
                .is_feature_supported(&TableFeature::IcebergCompatV2)
        );
        Ok(())
    }

    #[rstest]
    #[case::enable_row_tracking(ENABLE_ROW_TRACKING, "true")]
    #[case::enable_column_mapping_id_mode(COLUMN_MAPPING_MODE, "id")]
    fn rejected_property_change_commits_nothing(
        #[case] key: &str,
        #[case] value: &str,
    ) -> Result<()> {
        let (url, _tempdir) = copy_test_table("table-without-dv-small")?;
        let engine = SyncEngine::new();
        let snapshot = Snapshot::builder_for(url.as_str()).build(&engine)?;
        let version = snapshot.version();
        let mut builder =
            snapshot.update_table(UpdateTableOperation::AlterTable, "test-engine", false);
        builder
            .local_state
            .properties_added
            .insert(key.to_string(), value.to_string());
        let error = builder
            .build(&engine, Box::new(FileSystemCommitter::new()))
            .unwrap_err();
        assert!(matches!(error, KernelError::Unsupported(_)), "{error}");

        let reloaded = Snapshot::builder_for(url.as_str()).build(&engine)?;
        assert_eq!(reloaded.version(), version);
        assert_eq!(
            reloaded
                .table_configuration()
                .metadata()
                .configuration()
                .get(key),
            None
        );
        Ok(())
    }

    #[test]
    fn protocol_only_alter_table_commits() -> Result<()> {
        let (url, _tempdir) = copy_test_table("table-without-dv-small")?;
        let engine = SyncEngine::new();
        let snapshot = Snapshot::builder_for(url.as_str()).build(&engine)?;
        let version = snapshot.version();
        let mut builder =
            snapshot.update_table(UpdateTableOperation::AlterTable, "test-engine", false);
        builder.local_state.properties_added.insert(
            "delta.feature.domainMetadata".to_string(),
            "supported".to_string(),
        );
        let transaction = builder.build(&engine, Box::new(FileSystemCommitter::new()))?;
        assert!(!transaction.should_emit_metadata);
        assert!(transaction.should_emit_protocol);
        transaction.commit(&engine)?.unwrap_committed();

        let reloaded = Snapshot::builder_for(url.as_str()).build(&engine)?;
        assert_eq!(reloaded.version(), version + 1);
        assert!(
            reloaded
                .table_configuration()
                .is_feature_supported(&TableFeature::DomainMetadata)
        );
        Ok(())
    }

    #[test]
    fn unchanged_property_alter_table_commits_without_metadata_or_protocol() -> Result<()> {
        let engine = SyncEngine::new_with_store(Arc::new(InMemory::new()));
        let snapshot = create_table("memory:///", test_schema_flat(), "test", true)
            .with_table_properties([("app.key", "value")])
            .build(&engine, Box::new(FileSystemCommitter::new()))?
            .commit(&engine)?
            .unwrap_post_commit_snapshot();
        let mut builder =
            snapshot
                .clone()
                .update_table(UpdateTableOperation::AlterTable, "test-engine", false);
        builder
            .local_state
            .properties_added
            .insert("app.key".to_string(), "value".to_string());
        let transaction = builder.build(&engine, Box::new(FileSystemCommitter::new()))?;
        assert!(!transaction.should_emit_metadata);
        assert!(!transaction.should_emit_protocol);
        transaction.commit(&engine)?.unwrap_committed();

        let reloaded = Snapshot::builder_for("memory:///").build(&engine)?;
        assert_eq!(reloaded.version(), snapshot.version() + 1);
        let (read_config, config) = (
            snapshot.table_configuration(),
            reloaded.table_configuration(),
        );
        assert_eq!(config.metadata(), read_config.metadata());
        assert_eq!(config.protocol(), read_config.protocol());
        Ok(())
    }

    #[rstest]
    fn enabling_iceberg_compat_v2_assigns_nested_field_ids(
        #[values(false, true)] enable_column_mapping_in_same_request: bool,
    ) -> Result<()> {
        let engine = SyncEngine::new_with_store(Arc::new(InMemory::new()));
        let schema = Arc::new(StructType::try_new([
            StructField::nullable("id", DataType::INTEGER),
            StructField::nullable("tags", ArrayType::new(DataType::STRING, true)),
            StructField::nullable(
                "s",
                StructType::try_new([StructField::nullable(
                    "arr",
                    ArrayType::new(DataType::INTEGER, true),
                )])?,
            ),
            StructField::nullable(
                "m",
                MapType::new(
                    DataType::STRING,
                    StructType::try_new([
                        StructField::nullable("y", DataType::INTEGER),
                        StructField::nullable("z", ArrayType::new(DataType::STRING, true)),
                    ])?,
                    true,
                ),
            ),
        ])?);
        let initial_mode = if enable_column_mapping_in_same_request {
            "none"
        } else {
            "name"
        };
        let snapshot = create_table("memory:///", schema, "test", true)
            .with_table_properties([(COLUMN_MAPPING_MODE, initial_mode)])
            .build(&engine, Box::new(FileSystemCommitter::new()))?
            .commit(&engine)?
            .unwrap_post_commit_snapshot();

        let mut builder =
            snapshot.update_table(UpdateTableOperation::AlterTable, "test-engine", false);
        let properties = &mut builder.local_state.properties_added;
        properties.insert(ENABLE_ICEBERG_COMPAT_V2.to_string(), "true".to_string());
        if enable_column_mapping_in_same_request {
            properties.insert(COLUMN_MAPPING_MODE.to_string(), "name".to_string());
        }
        builder
            .build(&engine, Box::new(FileSystemCommitter::new()))?
            .commit(&engine)?
            .unwrap_committed();

        let reloaded = Snapshot::builder_for("memory:///").build(&engine)?;
        let config = reloaded.table_configuration();
        assert!(config.is_feature_enabled(&TableFeature::IcebergCompatV2));
        assert_eq!(config.column_mapping_mode(), ColumnMappingMode::Name);
        let schema = config.logical_schema();
        let struct_field = |field: &StructField, name: &str| -> StructField {
            match field.data_type() {
                DataType::Struct(inner) => inner.field(name).unwrap().clone(),
                DataType::Map(map) => match map.value_type() {
                    DataType::Struct(inner) => inner.field(name).unwrap().clone(),
                    other => panic!("unexpected map value {other:?}"),
                },
                other => panic!("unexpected type {other:?}"),
            }
        };
        let m = schema.field("m").unwrap().clone();
        let s = schema.field("s").unwrap().clone();
        let fields = [
            (schema.field("tags").unwrap().clone(), &["element"][..]),
            (struct_field(&s, "arr"), &["element"][..]),
            (m.clone(), &["key", "value"][..]),
            (struct_field(&m, "z"), &["element"][..]),
        ];
        let mut nested_ids = Vec::new();
        for (field, slots) in &fields {
            let physical = field.physical_name(ColumnMappingMode::Name);
            if enable_column_mapping_in_same_request {
                assert_eq!(physical, field.name());
            }
            let Some(MetadataValue::Other(serde_json::Value::Object(ids))) =
                field.get_config_value(&ColumnMetadataKey::ColumnMappingNestedIds)
            else {
                panic!("expected nested ids on '{}'", field.name());
            };
            assert_eq!(ids.len(), slots.len(), "{ids:?}");
            for slot in *slots {
                nested_ids.push(ids[&format!("{physical}.{slot}")].as_i64().unwrap());
            }
        }
        // Seven struct fields hold ids 1..=7; the five array and map slots follow.
        nested_ids.sort_unstable();
        assert_eq!(nested_ids, (8..=12).collect::<Vec<_>>());
        assert_eq!(
            config.table_properties().column_mapping_max_column_id,
            Some(12)
        );
        Ok(())
    }

    #[test]
    fn enabling_column_mapping_name_mode_keeps_existing_data_readable() -> Result<()> {
        fn read_columns(snapshot: &SnapshotRef, engine: &Arc<SyncEngine>) -> Result<Vec<ArrayRef>> {
            let mut columns = Vec::new();
            for data in snapshot
                .clone()
                .scan_builder()
                .build()?
                .execute(engine.clone())?
            {
                let batch = ArrowEngineData::try_from_engine_data(data?)?;
                columns.extend(batch.record_batch().columns().iter().cloned());
            }
            Ok(columns)
        }

        let (url, _tempdir) = copy_test_table("table-without-dv-small")?;
        let engine = Arc::new(SyncEngine::new());
        let snapshot = Snapshot::builder_for(url.as_str()).build(engine.as_ref())?;
        let columns = read_columns(&snapshot, &engine)?;
        assert!(!columns.is_empty());

        let mut builder =
            snapshot.update_table(UpdateTableOperation::AlterTable, "test-engine", false);
        builder
            .local_state
            .properties_added
            .insert(COLUMN_MAPPING_MODE.to_string(), "name".to_string());
        builder
            .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?
            .commit(engine.as_ref())?
            .unwrap_committed();

        let reloaded = Snapshot::builder_for(url.as_str()).build(engine.as_ref())?;
        assert_eq!(
            reloaded.table_configuration().column_mapping_mode(),
            ColumnMappingMode::Name
        );
        assert_eq!(read_columns(&reloaded, &engine)?, columns);
        Ok(())
    }

    #[rstest]
    #[case::iceberg_compat_v2(TableFeature::IcebergCompatV2, "icebergCompatV2 enabled")]
    #[case::iceberg_compat_v3(TableFeature::IcebergCompatV3, "icebergCompatV3 enabled")]
    #[case::column_defaults(TableFeature::AllowColumnDefaults, "allowColumnDefaults enabled")]
    fn updated_schema_is_rejected_on_tables_kernel_cannot_evolve_yet(
        #[case] feature: TableFeature,
        #[case] expected: &str,
    ) -> Result<()> {
        let (engine, snapshot, _tempdir) = create_schema_validation_table(Some(&feature), true);
        let error = snapshot
            .clone()
            .update_table(UpdateTableOperation::AlterTable, "test-engine", false)
            .with_updated_schema(snapshot.schema())
            .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))
            .unwrap_err();
        assert!(matches!(error, KernelError::Unsupported(_)), "{error}");
        assert!(
            error.to_string().contains(&format!(
                "Schema changes are not yet supported on tables with {expected}"
            )),
            "{error}"
        );

        let reloaded = Snapshot::builder_for(snapshot.table_root()).build(engine.as_ref())?;
        assert_eq!(reloaded.version(), snapshot.version());
        Ok(())
    }

    fn create_schema_validation_table(
        feature: Option<&TableFeature>,
        feature_enabled: bool,
    ) -> SchemaValidationTable {
        let (engine, snapshot, tempdir): SchemaValidationTable =
            if feature == Some(&TableFeature::AllowColumnDefaults) {
                load_test_table("table-with-column-defaults").expect("load column-defaults table")
            } else {
                let property_value = if feature_enabled { "true" } else { "false" };
                let properties = match feature {
                    None => vec![],
                    Some(TableFeature::IcebergCompatV2) => vec![
                        ("delta.feature.icebergCompatV2", "supported"),
                        (ENABLE_ICEBERG_COMPAT_V2, property_value),
                        ("delta.columnMapping.mode", "name"),
                    ],
                    Some(TableFeature::IcebergCompatV3) => vec![
                        ("delta.feature.icebergCompatV3", "supported"),
                        (ENABLE_ICEBERG_COMPAT_V3, property_value),
                        ("delta.columnMapping.mode", "name"),
                        ("delta.enableRowTracking", "true"),
                    ],
                    Some(feature) => panic!("unsupported schema-validation fixture: {feature}"),
                };
                let engine = Arc::new(SyncEngine::new_with_store(Arc::new(InMemory::new())));
                let snapshot = create_table("memory:///", test_schema_flat(), "test", true)
                    .with_table_properties(properties)
                    .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))
                    .expect("valid table properties")
                    .commit(engine.as_ref())
                    .expect("create schema-validation table")
                    .unwrap_post_commit_snapshot();
                (engine, snapshot, None)
            };
        if let Some(feature) = feature {
            assert!(snapshot.table_configuration().is_feature_supported(feature));
            assert_eq!(
                snapshot.table_configuration().is_feature_enabled(feature),
                feature_enabled
            );
        }
        (engine, snapshot, tempdir)
    }
}
