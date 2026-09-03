//! Atomic multi-file patch transaction contracts.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FilePatch {
    /// Canonical project-relative path or an absolute alias inside the project.
    pub file_path: String,
    /// Replacement content. Newlines are converted to the existing file's style.
    #[serde(default)]
    pub content: String,
    /// First line to replace, 1-based inclusive. Omit with end_line for a whole-file patch.
    #[serde(default)]
    pub start_line: Option<usize>,
    /// Last line to replace, 1-based inclusive. Use start_line=end_line+1 for insertion.
    #[serde(default)]
    pub end_line: Option<usize>,
    /// SHA256 of the current raw file bytes. Required for existing files.
    #[serde(default)]
    pub expected_hash: Option<String>,
    /// Delete the file. Content and line ranges must be omitted/empty.
    #[serde(default)]
    pub delete: bool,
}

impl FilePatch {
    pub fn validate(&self) -> Result<(), String> {
        if self.file_path.trim().is_empty() {
            return Err("file_path cannot be empty".to_string());
        }
        if self.delete {
            if !self.content.is_empty() || self.start_line.is_some() || self.end_line.is_some() {
                return Err("delete patches cannot contain content or a line range".to_string());
            }
            return Ok(());
        }
        match (self.start_line, self.end_line) {
            (None, None) => Ok(()),
            (Some(start), Some(end)) if start > 0 && start <= end.saturating_add(1) => Ok(()),
            (Some(0), Some(_)) => Err("start_line must be >= 1".to_string()),
            (Some(_), Some(_)) => {
                Err("start_line must be <= end_line + 1 (end_line + 1 is insertion)".to_string())
            }
            _ => Err("start_line and end_line must both be set or both omitted".to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ApplyPatchRequest {
    /// All patches in one filesystem and indexing transaction.
    pub patches: Vec<FilePatch>,
    /// Validate and stage the complete result without writing or reindexing.
    #[serde(default)]
    pub dry_run: bool,
    /// Stable project id when relative paths could match more than one indexed root.
    #[serde(default)]
    pub project: Option<String>,
}

impl ApplyPatchRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.patches.is_empty() {
            return Err("patches must contain at least one file patch".to_string());
        }
        if self.patches.len() > 100 {
            return Err("patches cannot contain more than 100 file patches".to_string());
        }
        for patch in &self.patches {
            patch.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PatchedFileResult {
    pub file_path: String,
    pub operation: String,
    pub previous_hash: Option<String>,
    pub new_hash: Option<String>,
    pub total_lines: Option<usize>,
    pub encoding: String,
    pub newline: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PatchConflict {
    pub file_path: String,
    pub expected_hash: Option<String>,
    pub actual_hash: Option<String>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ApplyPatchResponse {
    /// applied, validated, conflict, validation_failed, or reindex_failed.
    pub status: String,
    pub dry_run: bool,
    pub files: Vec<PatchedFileResult>,
    #[serde(default)]
    pub conflicts: Vec<PatchConflict>,
    #[serde(default)]
    pub validation_errors: Vec<String>,
    pub filesystem_committed: bool,
    pub reindexed: bool,
    pub index_stale: bool,
    pub generation_before: u64,
    pub generation_after: u64,
    #[serde(default)]
    pub warning: Option<String>,
    pub duration_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_rejects_empty_duplicate_shape_and_invalid_delete_payloads() {
        assert!(
            ApplyPatchRequest {
                patches: vec![],
                dry_run: false,
                project: None
            }
            .validate()
            .is_err()
        );
        assert!(
            FilePatch {
                file_path: "a.rs".into(),
                content: "x".into(),
                start_line: None,
                end_line: None,
                expected_hash: None,
                delete: true,
            }
            .validate()
            .is_err()
        );
        assert!(
            FilePatch {
                file_path: "a.rs".into(),
                content: "x".into(),
                start_line: Some(3),
                end_line: Some(1),
                expected_hash: None,
                delete: false,
            }
            .validate()
            .is_err()
        );
    }
}
