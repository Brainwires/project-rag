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
    Definition, DispatchKind, EvidenceKind, LinkageKind, LocationRole, Reference,
    ReferenceCandidate, ReferenceKind, ResolutionStatus, SourceLocation, SymbolId, SymbolKind,
    Visibility,
};

/// Schema of the definitions table.
pub fn definitions_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("file_path", DataType::Utf8, false),
        Field::new("root_path", DataType::Utf8, true),
        Field::new("project", DataType::Utf8, true),
        Field::new("project_id", DataType::Utf8, false),
        Field::new("language", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("qualified_name", DataType::Utf8, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("canonical_signature", DataType::Utf8, false),
        Field::new("linkage", DataType::Utf8, false),
        Field::new("scope_discriminator", DataType::Utf8, true),
        Field::new("location_id", DataType::Utf8, false),
        Field::new("location_role", DataType::Utf8, false),
        Field::new("location_start_line", DataType::UInt32, false),
        Field::new("location_start_col", DataType::UInt32, false),
        Field::new("location_end_line", DataType::UInt32, false),
        Field::new("location_end_col", DataType::UInt32, false),
        Field::new("start_line", DataType::UInt32, false),
        Field::new("start_col", DataType::UInt32, false),
        Field::new("end_line", DataType::UInt32, false),
        Field::new("end_col", DataType::UInt32, false),
        Field::new("signature", DataType::Utf8, false),
        Field::new("doc_comment", DataType::Utf8, true),
        Field::new("visibility", DataType::Utf8, false),
        Field::new("parent_id", DataType::Utf8, true),
        Field::new("parser", DataType::Utf8, false),
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
        Field::new("location_id", DataType::Utf8, false),
        Field::new("source_symbol_id", DataType::Utf8, true),
        Field::new("target_symbol_id", DataType::Utf8, false),
        Field::new("target_name", DataType::Utf8, false),
        Field::new("candidates", DataType::Utf8, false),
        Field::new("reference_kind", DataType::Utf8, false),
        Field::new("resolution_status", DataType::Utf8, false),
        Field::new("evidence_kind", DataType::Utf8, false),
        Field::new("dispatch_kind", DataType::Utf8, false),
        Field::new("language", DataType::Utf8, false),
        Field::new("parser", DataType::Utf8, false),
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
    let project_ids = StringArray::from(
        definitions
            .iter()
            .map(|d| d.symbol_id.project_id.as_str())
            .collect::<Vec<_>>(),
    );
    let languages = StringArray::from(
        definitions
            .iter()
            .map(|d| d.symbol_id.language.as_str())
            .collect::<Vec<_>>(),
    );
    let names = StringArray::from(definitions.iter().map(|d| d.name()).collect::<Vec<_>>());
    let qualified_names = StringArray::from(
        definitions
            .iter()
            .map(|d| d.symbol_id.qualified_name.as_str())
            .collect::<Vec<_>>(),
    );
    let kinds = StringArray::from(
        definitions
            .iter()
            .map(|d| enum_to_str(&d.symbol_id.kind))
            .collect::<Vec<_>>(),
    );
    let canonical_signatures = StringArray::from(
        definitions
            .iter()
            .map(|d| d.symbol_id.canonical_signature.as_str())
            .collect::<Vec<_>>(),
    );
    let linkages = StringArray::from(
        definitions
            .iter()
            .map(|d| enum_to_str(&d.symbol_id.linkage))
            .collect::<Vec<_>>(),
    );
    let scope_discriminators = StringArray::from(
        definitions
            .iter()
            .map(|d| d.symbol_id.scope_discriminator.as_deref())
            .collect::<Vec<_>>(),
    );
    let location_ids = StringArray::from(
        definitions
            .iter()
            .map(|d| d.location.to_storage_id())
            .collect::<Vec<_>>(),
    );
    let location_roles = StringArray::from(
        definitions
            .iter()
            .map(|d| enum_to_str(&d.location.role))
            .collect::<Vec<_>>(),
    );
    let location_start_lines = UInt32Array::from(
        definitions
            .iter()
            .map(|d| d.location.start_line as u32)
            .collect::<Vec<_>>(),
    );
    let location_start_cols = UInt32Array::from(
        definitions
            .iter()
            .map(|d| d.location.start_col as u32)
            .collect::<Vec<_>>(),
    );
    let location_end_lines = UInt32Array::from(
        definitions
            .iter()
            .map(|d| d.location.end_line as u32)
            .collect::<Vec<_>>(),
    );
    let location_end_cols = UInt32Array::from(
        definitions
            .iter()
            .map(|d| d.location.end_col as u32)
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
    let parsers = StringArray::from(
        definitions
            .iter()
            .map(|d| d.parser.as_str())
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
            Arc::new(project_ids),
            Arc::new(languages),
            Arc::new(names),
            Arc::new(qualified_names),
            Arc::new(kinds),
            Arc::new(canonical_signatures),
            Arc::new(linkages),
            Arc::new(scope_discriminators),
            Arc::new(location_ids),
            Arc::new(location_roles),
            Arc::new(location_start_lines),
            Arc::new(location_start_cols),
            Arc::new(location_end_lines),
            Arc::new(location_end_cols),
            Arc::new(start_lines),
            Arc::new(start_cols),
            Arc::new(end_lines),
            Arc::new(end_cols),
            Arc::new(signatures),
            Arc::new(doc_comments),
            Arc::new(visibilities),
            Arc::new(parent_ids),
            Arc::new(parsers),
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
    let location_ids = StringArray::from(
        references
            .iter()
            .map(|r| r.location_id.as_str())
            .collect::<Vec<_>>(),
    );
    let source_symbol_ids = StringArray::from(
        references
            .iter()
            .map(|r| r.source_symbol_id.as_deref())
            .collect::<Vec<_>>(),
    );
    let targets = StringArray::from(
        references
            .iter()
            .map(|r| r.target_symbol_id.as_str())
            .collect::<Vec<_>>(),
    );
    let target_names = StringArray::from(
        references
            .iter()
            .map(|r| r.target_name.as_str())
            .collect::<Vec<_>>(),
    );
    let candidates = StringArray::from(
        references
            .iter()
            .map(|r| serde_json::to_string(&r.candidates).unwrap_or_else(|_| "[]".to_string()))
            .collect::<Vec<_>>(),
    );
    let kinds = StringArray::from(
        references
            .iter()
            .map(|r| enum_to_str(&r.reference_kind))
            .collect::<Vec<_>>(),
    );
    let resolution_statuses = StringArray::from(
        references
            .iter()
            .map(|r| enum_to_str(&r.resolution_status))
            .collect::<Vec<_>>(),
    );
    let evidence_kinds = StringArray::from(
        references
            .iter()
            .map(|r| enum_to_str(&r.evidence_kind))
            .collect::<Vec<_>>(),
    );
    let dispatch_kinds = StringArray::from(
        references
            .iter()
            .map(|r| enum_to_str(&r.dispatch_kind))
            .collect::<Vec<_>>(),
    );
    let languages = StringArray::from(
        references
            .iter()
            .map(|r| r.language.as_str())
            .collect::<Vec<_>>(),
    );
    let parsers = StringArray::from(
        references
            .iter()
            .map(|r| r.parser.as_str())
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
            Arc::new(location_ids),
            Arc::new(source_symbol_ids),
            Arc::new(targets),
            Arc::new(target_names),
            Arc::new(candidates),
            Arc::new(kinds),
            Arc::new(resolution_statuses),
            Arc::new(evidence_kinds),
            Arc::new(dispatch_kinds),
            Arc::new(languages),
            Arc::new(parsers),
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
    let project_ids = str_col(batch, "project_id")?;
    let languages = str_col(batch, "language")?;
    let names = str_col(batch, "name")?;
    let qualified_names = str_col(batch, "qualified_name")?;
    let kinds = str_col(batch, "kind")?;
    let canonical_signatures = str_col(batch, "canonical_signature")?;
    let linkages = str_col(batch, "linkage")?;
    let scope_discriminators = str_col(batch, "scope_discriminator")?;
    let location_roles = str_col(batch, "location_role")?;
    let location_start_lines = u32_col(batch, "location_start_line")?;
    let location_start_cols = u32_col(batch, "location_start_col")?;
    let location_end_lines = u32_col(batch, "location_end_line")?;
    let location_end_cols = u32_col(batch, "location_end_col")?;
    let start_lines = u32_col(batch, "start_line")?;
    let start_cols = u32_col(batch, "start_col")?;
    let end_lines = u32_col(batch, "end_line")?;
    let end_cols = u32_col(batch, "end_col")?;
    let signatures = str_col(batch, "signature")?;
    let doc_comments = str_col(batch, "doc_comment")?;
    let visibilities = str_col(batch, "visibility")?;
    let parent_ids = str_col(batch, "parent_id")?;
    let parsers = str_col(batch, "parser")?;
    let indexed_ats = i64_col(batch, "indexed_at")?;

    let mut out = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        let kind = enum_from_str::<SymbolKind>(kinds.value(i)).unwrap_or(SymbolKind::Unknown);
        let visibility = enum_from_str::<Visibility>(visibilities.value(i)).unwrap_or_default();
        let linkage = enum_from_str::<LinkageKind>(linkages.value(i)).unwrap_or_default();
        let location_role =
            enum_from_str::<LocationRole>(location_roles.value(i)).unwrap_or_default();
        out.push(Definition {
            symbol_id: SymbolId::new_logical(
                project_ids.value(i),
                languages.value(i),
                qualified_names.value(i),
                names.value(i),
                kind,
                canonical_signatures.value(i),
                linkage,
                opt_str(scope_discriminators, i),
                file_paths.value(i),
                start_lines.value(i) as usize,
                start_cols.value(i) as usize,
            ),
            location: SourceLocation {
                project_id: project_ids.value(i).to_string(),
                file_path: file_paths.value(i).to_string(),
                start_line: location_start_lines.value(i) as usize,
                start_col: location_start_cols.value(i) as usize,
                end_line: location_end_lines.value(i) as usize,
                end_col: location_end_cols.value(i) as usize,
                role: location_role,
            },
            root_path: opt_str(root_paths, i),
            project: opt_str(projects, i),
            end_line: end_lines.value(i) as usize,
            end_col: end_cols.value(i) as usize,
            signature: signatures.value(i).to_string(),
            doc_comment: opt_str(doc_comments, i),
            visibility,
            parent_id: opt_str(parent_ids, i),
            parser: parsers.value(i).to_string(),
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
    let location_ids = str_col(batch, "location_id")?;
    let source_symbol_ids = str_col(batch, "source_symbol_id")?;
    let targets = str_col(batch, "target_symbol_id")?;
    let target_names = str_col(batch, "target_name")?;
    let candidates = str_col(batch, "candidates")?;
    let kinds = str_col(batch, "reference_kind")?;
    let resolution_statuses = str_col(batch, "resolution_status")?;
    let evidence_kinds = str_col(batch, "evidence_kind")?;
    let dispatch_kinds = str_col(batch, "dispatch_kind")?;
    let languages = str_col(batch, "language")?;
    let parsers = str_col(batch, "parser")?;
    let indexed_ats = i64_col(batch, "indexed_at")?;

    let mut out = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        let reference_kind =
            enum_from_str::<ReferenceKind>(kinds.value(i)).unwrap_or(ReferenceKind::Unknown);
        let resolution_status =
            enum_from_str::<ResolutionStatus>(resolution_statuses.value(i)).unwrap_or_default();
        let evidence_kind =
            enum_from_str::<EvidenceKind>(evidence_kinds.value(i)).unwrap_or_default();
        let dispatch_kind =
            enum_from_str::<DispatchKind>(dispatch_kinds.value(i)).unwrap_or_default();
        let parsed_candidates: Vec<ReferenceCandidate> =
            serde_json::from_str(candidates.value(i)).unwrap_or_default();
        out.push(Reference {
            file_path: file_paths.value(i).to_string(),
            root_path: opt_str(root_paths, i),
            project: opt_str(projects, i),
            start_line: start_lines.value(i) as usize,
            end_line: end_lines.value(i) as usize,
            start_col: start_cols.value(i) as usize,
            end_col: end_cols.value(i) as usize,
            location_id: location_ids.value(i).to_string(),
            source_symbol_id: opt_str(source_symbol_ids, i),
            target_symbol_id: targets.value(i).to_string(),
            target_name: target_names.value(i).to_string(),
            candidates: parsed_candidates,
            reference_kind,
            resolution_status,
            evidence_kind,
            dispatch_kind,
            language: languages.value(i).to_string(),
            parser: parsers.value(i).to_string(),
            indexed_at: indexed_ats.value(i),
        });
    }
    Ok(out)
}
