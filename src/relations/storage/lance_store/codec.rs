//! Arrow schemas and row codecs for the relations tables.
//!
//! Definitions and references are flat rows; enums travel as their serde
//! snake_case string form so the stored value matches what a `only_if` filter
//! written from Rust enum values will compare against.

use anyhow::{Context, Result};
use arrow_array::{Array, Int64Array, RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

use crate::relations::types::{
    Definition, Reference, ReferenceKind, SymbolId, SymbolKind, Visibility,
};

/// Schema of the definitions table.
pub fn definitions_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("file_path", DataType::Utf8, false),
        Field::new("root_path", DataType::Utf8, true),
        Field::new("project", DataType::Utf8, true),
        Field::new("name", DataType::Utf8, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("start_line", DataType::UInt32, false),
        Field::new("start_col", DataType::UInt32, false),
        Field::new("end_line", DataType::UInt32, false),
        Field::new("end_col", DataType::UInt32, false),
        Field::new("signature", DataType::Utf8, false),
        Field::new("doc_comment", DataType::Utf8, true),
        Field::new("visibility", DataType::Utf8, false),
        Field::new("parent_id", DataType::Utf8, true),
        Field::new("indexed_at", DataType::Int64, false),
    ]))
}

/// Schema of the references table.
pub fn references_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("file_path", DataType::Utf8, false),
        Field::new("root_path", DataType::Utf8, true),
        Field::new("project", DataType::Utf8, true),
        Field::new("start_line", DataType::UInt32, false),
        Field::new("end_line", DataType::UInt32, false),
        Field::new("start_col", DataType::UInt32, false),
        Field::new("end_col", DataType::UInt32, false),
        Field::new("target_symbol_id", DataType::Utf8, false),
        Field::new("reference_kind", DataType::Utf8, false),
        Field::new("indexed_at", DataType::Int64, false),
    ]))
}

/// Serde snake_case form of an enum value, without the JSON quotes.
pub fn enum_to_str<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value)
        .unwrap_or_default()
        .trim_matches('"')
        .to_string()
}

fn enum_from_str<T: serde::de::DeserializeOwned>(s: &str) -> Option<T> {
    serde_json::from_str(&format!("\"{}\"", s)).ok()
}

/// Escape a string for use inside a single-quoted SQL literal.
pub fn escape_sql(s: &str) -> String {
    s.replace('\'', "''")
}

/// Quoted, escaped, comma-joined list for an `IN (...)` filter.
pub fn sql_in_list<S: AsRef<str>>(values: &[S]) -> String {
    values
        .iter()
        .map(|v| format!("'{}'", escape_sql(v.as_ref())))
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn definitions_to_batch(definitions: &[Definition]) -> Result<RecordBatch> {
    let ids = StringArray::from(
        definitions
            .iter()
            .map(|d| d.to_storage_id())
            .collect::<Vec<_>>(),
    );
    let file_paths = StringArray::from(
        definitions
            .iter()
            .map(|d| d.file_path())
            .collect::<Vec<_>>(),
    );
    let root_paths = StringArray::from(
        definitions
            .iter()
            .map(|d| d.root_path.as_deref())
            .collect::<Vec<_>>(),
    );
    let projects = StringArray::from(
        definitions
            .iter()
            .map(|d| d.project.as_deref())
            .collect::<Vec<_>>(),
    );
    let names = StringArray::from(definitions.iter().map(|d| d.name()).collect::<Vec<_>>());
    let kinds = StringArray::from(
        definitions
            .iter()
            .map(|d| enum_to_str(&d.symbol_id.kind))
            .collect::<Vec<_>>(),
    );
    let start_lines = UInt32Array::from(
        definitions
            .iter()
            .map(|d| d.symbol_id.start_line as u32)
            .collect::<Vec<_>>(),
    );
    let start_cols = UInt32Array::from(
        definitions
            .iter()
            .map(|d| d.symbol_id.start_col as u32)
            .collect::<Vec<_>>(),
    );
    let end_lines = UInt32Array::from(
        definitions
            .iter()
            .map(|d| d.end_line as u32)
            .collect::<Vec<_>>(),
    );
    let end_cols = UInt32Array::from(
        definitions
            .iter()
            .map(|d| d.end_col as u32)
            .collect::<Vec<_>>(),
    );
    let signatures = StringArray::from(
        definitions
            .iter()
            .map(|d| d.signature.as_str())
            .collect::<Vec<_>>(),
    );
    let doc_comments = StringArray::from(
        definitions
            .iter()
            .map(|d| d.doc_comment.as_deref())
            .collect::<Vec<_>>(),
    );
    let visibilities = StringArray::from(
        definitions
            .iter()
            .map(|d| enum_to_str(&d.visibility))
            .collect::<Vec<_>>(),
    );
    let parent_ids = StringArray::from(
        definitions
            .iter()
            .map(|d| d.parent_id.as_deref())
            .collect::<Vec<_>>(),
    );
    let indexed_ats =
        Int64Array::from(definitions.iter().map(|d| d.indexed_at).collect::<Vec<_>>());

    RecordBatch::try_new(
        definitions_schema(),
        vec![
            Arc::new(ids),
            Arc::new(file_paths),
            Arc::new(root_paths),
            Arc::new(projects),
            Arc::new(names),
            Arc::new(kinds),
            Arc::new(start_lines),
            Arc::new(start_cols),
            Arc::new(end_lines),
            Arc::new(end_cols),
            Arc::new(signatures),
            Arc::new(doc_comments),
            Arc::new(visibilities),
            Arc::new(parent_ids),
            Arc::new(indexed_ats),
        ],
    )
    .context("Failed to build definitions RecordBatch")
}

pub fn references_to_batch(references: &[Reference]) -> Result<RecordBatch> {
    let ids = StringArray::from(
        references
            .iter()
            .map(|r| r.to_storage_id())
            .collect::<Vec<_>>(),
    );
    let file_paths = StringArray::from(
        references
            .iter()
            .map(|r| r.file_path.as_str())
            .collect::<Vec<_>>(),
    );
    let root_paths = StringArray::from(
        references
            .iter()
            .map(|r| r.root_path.as_deref())
            .collect::<Vec<_>>(),
    );
    let projects = StringArray::from(
        references
            .iter()
            .map(|r| r.project.as_deref())
            .collect::<Vec<_>>(),
    );
    let start_lines = UInt32Array::from(
        references
            .iter()
            .map(|r| r.start_line as u32)
            .collect::<Vec<_>>(),
    );
    let end_lines = UInt32Array::from(
        references
            .iter()
            .map(|r| r.end_line as u32)
            .collect::<Vec<_>>(),
    );
    let start_cols = UInt32Array::from(
        references
            .iter()
            .map(|r| r.start_col as u32)
            .collect::<Vec<_>>(),
    );
    let end_cols = UInt32Array::from(
        references
            .iter()
            .map(|r| r.end_col as u32)
            .collect::<Vec<_>>(),
    );
    let targets = StringArray::from(
        references
            .iter()
            .map(|r| r.target_symbol_id.as_str())
            .collect::<Vec<_>>(),
    );
    let kinds = StringArray::from(
        references
            .iter()
            .map(|r| enum_to_str(&r.reference_kind))
            .collect::<Vec<_>>(),
    );
    let indexed_ats = Int64Array::from(references.iter().map(|r| r.indexed_at).collect::<Vec<_>>());

    RecordBatch::try_new(
        references_schema(),
        vec![
            Arc::new(ids),
            Arc::new(file_paths),
            Arc::new(root_paths),
            Arc::new(projects),
            Arc::new(start_lines),
            Arc::new(end_lines),
            Arc::new(start_cols),
            Arc::new(end_cols),
            Arc::new(targets),
            Arc::new(kinds),
            Arc::new(indexed_ats),
        ],
    )
    .context("Failed to build references RecordBatch")
}

fn str_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    batch
        .column_by_name(name)
        .with_context(|| format!("Missing column {}", name))?
        .as_any()
        .downcast_ref::<StringArray>()
        .with_context(|| format!("Column {} is not Utf8", name))
}

fn u32_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt32Array> {
    batch
        .column_by_name(name)
        .with_context(|| format!("Missing column {}", name))?
        .as_any()
        .downcast_ref::<UInt32Array>()
        .with_context(|| format!("Column {} is not UInt32", name))
}

fn i64_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array> {
    batch
        .column_by_name(name)
        .with_context(|| format!("Missing column {}", name))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .with_context(|| format!("Column {} is not Int64", name))
}

fn opt_str(array: &StringArray, i: usize) -> Option<String> {
    if array.is_null(i) {
        None
    } else {
        Some(array.value(i).to_string())
    }
}

pub fn batch_to_definitions(batch: &RecordBatch) -> Result<Vec<Definition>> {
    let file_paths = str_col(batch, "file_path")?;
    let root_paths = str_col(batch, "root_path")?;
    let projects = str_col(batch, "project")?;
    let names = str_col(batch, "name")?;
    let kinds = str_col(batch, "kind")?;
    let start_lines = u32_col(batch, "start_line")?;
    let start_cols = u32_col(batch, "start_col")?;
    let end_lines = u32_col(batch, "end_line")?;
    let end_cols = u32_col(batch, "end_col")?;
    let signatures = str_col(batch, "signature")?;
    let doc_comments = str_col(batch, "doc_comment")?;
    let visibilities = str_col(batch, "visibility")?;
    let parent_ids = str_col(batch, "parent_id")?;
    let indexed_ats = i64_col(batch, "indexed_at")?;

    let mut out = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        let kind = enum_from_str::<SymbolKind>(kinds.value(i)).unwrap_or(SymbolKind::Unknown);
        let visibility = enum_from_str::<Visibility>(visibilities.value(i)).unwrap_or_default();
        out.push(Definition {
            symbol_id: SymbolId::new(
                file_paths.value(i),
                names.value(i),
                kind,
                start_lines.value(i) as usize,
                start_cols.value(i) as usize,
            ),
            root_path: opt_str(root_paths, i),
            project: opt_str(projects, i),
            end_line: end_lines.value(i) as usize,
            end_col: end_cols.value(i) as usize,
            signature: signatures.value(i).to_string(),
            doc_comment: opt_str(doc_comments, i),
            visibility,
            parent_id: opt_str(parent_ids, i),
            indexed_at: indexed_ats.value(i),
        });
    }
    Ok(out)
}

pub fn batch_to_references(batch: &RecordBatch) -> Result<Vec<Reference>> {
    let file_paths = str_col(batch, "file_path")?;
    let root_paths = str_col(batch, "root_path")?;
    let projects = str_col(batch, "project")?;
    let start_lines = u32_col(batch, "start_line")?;
    let end_lines = u32_col(batch, "end_line")?;
    let start_cols = u32_col(batch, "start_col")?;
    let end_cols = u32_col(batch, "end_col")?;
    let targets = str_col(batch, "target_symbol_id")?;
    let kinds = str_col(batch, "reference_kind")?;
    let indexed_ats = i64_col(batch, "indexed_at")?;

    let mut out = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        let reference_kind =
            enum_from_str::<ReferenceKind>(kinds.value(i)).unwrap_or(ReferenceKind::Unknown);
        out.push(Reference {
            file_path: file_paths.value(i).to_string(),
            root_path: opt_str(root_paths, i),
            project: opt_str(projects, i),
            start_line: start_lines.value(i) as usize,
            end_line: end_lines.value(i) as usize,
            start_col: start_cols.value(i) as usize,
            end_col: end_cols.value(i) as usize,
            target_symbol_id: targets.value(i).to_string(),
            reference_kind,
            indexed_at: indexed_ats.value(i),
        });
    }
    Ok(out)
}
