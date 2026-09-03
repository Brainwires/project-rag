//! Request/response types for reading and editing files that live inside an
//! already-indexed project root.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Request to read a bounded slice of a file's current on-disk content.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ReadFileRequest {
    /// File path (relative or absolute). Must be inside an already-indexed project root.
    pub file_path: String,
    /// First line to read, 1-indexed inclusive. Omit to read from the start of the file.
    #[serde(default)]
    pub start_line: Option<usize>,
    /// Number of lines to read. Defaults to 30 and may exceed 30 explicitly,
    /// subject to the server hard response limit.
    #[serde(default)]
    pub line_count: Option<usize>,
    /// Legacy inclusive end line. Preserved for compatibility; callers should
    /// prefer `line_count`. It cannot be combined with `line_count`.
    #[serde(default)]
    pub end_line: Option<usize>,
}

impl ReadFileRequest {
    /// Validate the read file request
    pub fn validate(&self) -> Result<(), String> {
        if self.file_path.is_empty() {
            return Err("file_path cannot be empty".to_string());
        }
        if self.start_line == Some(0) {
            return Err("start_line must be >= 1".to_string());
        }
        if self.line_count == Some(0) {
            return Err("line_count must be >= 1".to_string());
        }
        if self.line_count.is_some() && self.end_line.is_some() {
            return Err("line_count and end_line cannot both be set".to_string());
        }
        if let (Some(start), Some(end)) = (self.start_line, self.end_line)
            && start > end
        {
            return Err("start_line must be <= end_line".to_string());
        }
        Ok(())
    }
}

/// Response from read_file
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ReadFileResponse {
    /// The file content for the requested (and possibly clamped/capped) line range
    pub content: String,
    /// Canonical project-relative path.
    pub file_path: String,
    pub requested_start_line: usize,
    pub requested_line_count: usize,
    pub returned_start_line: usize,
    pub returned_end_line: usize,
    pub returned_line_count: usize,
    /// Whether the returned range reaches EOF (including an empty range beyond EOF).
    pub eof: bool,
    /// Whether the requested range extended beyond valid file bounds.
    pub range_clamped: bool,
    /// Whether a server hard response limit omitted otherwise valid requested content.
    pub content_truncated: bool,
    /// Next readable line when file content remains.
    pub next_start_line: Option<usize>,
    /// Compatibility alias for `returned_start_line`.
    pub start_line: usize,
    /// Compatibility alias for `returned_end_line`.
    pub end_line: usize,
    /// Total number of lines in the file
    pub total_lines: usize,
    /// Compatibility alias for `content_truncated`. EOF clamping is reported
    /// separately through `range_clamped`.
    pub truncated: bool,
    /// SHA256 hash of the full current file content. Pass this as `expected_hash`
    /// to edit_file to guard against editing a file that changed since this read.
    pub file_hash: String,
    /// Detected language, if any
    pub language: Option<String>,
}

/// Request to replace a line range (or the whole file) with new content
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EditFileRequest {
    /// File path (relative or absolute). Must be inside an already-indexed project root.
    pub file_path: String,
    /// New content. Replaces the given line range, or the entire file (creating it if it
    /// doesn't exist yet) when start_line/end_line are omitted.
    pub content: String,
    /// First line to replace, 1-indexed inclusive. Omit (with end_line) to replace/create
    /// the whole file. Set to `end_line + 1` to insert `content` before that line without
    /// deleting anything.
    #[serde(default)]
    pub start_line: Option<usize>,
    /// Last line to replace, 1-indexed inclusive. Omit (with start_line) to replace/create
    /// the whole file.
    #[serde(default)]
    pub end_line: Option<usize>,
    /// SHA256 hash the file is expected to currently have (from a prior read_file/edit_file
    /// call). If it doesn't match the file's actual current hash, the edit is rejected
    /// instead of applied. Omit only when creating a brand new file.
    #[serde(default)]
    pub expected_hash: Option<String>,
    /// Project name to tag re-indexed embeddings with; should match what index_codebase
    /// used for this root, if anything.
    #[serde(default)]
    pub project: Option<String>,
}

impl EditFileRequest {
    /// Validate the edit file request
    pub fn validate(&self) -> Result<(), String> {
        if self.file_path.is_empty() {
            return Err("file_path cannot be empty".to_string());
        }
        match (self.start_line, self.end_line) {
            (Some(start), Some(end)) => {
                if start == 0 {
                    return Err("start_line must be >= 1".to_string());
                }
                if start > end + 1 {
                    return Err(
                        "start_line must be <= end_line + 1 (use end_line + 1 to insert without deleting)"
                            .to_string(),
                    );
                }
            }
            (None, None) => {}
            _ => {
                return Err("start_line and end_line must both be set or both omitted".to_string());
            }
        }
        Ok(())
    }
}

/// Response from edit_file
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EditFileResponse {
    /// Outcome of the request: "ok" or "hash_conflict"
    pub status: String,
    /// SHA256 hash of the file after the edit (only set when status == "ok")
    #[serde(default)]
    pub file_hash: Option<String>,
    /// Total number of lines in the file after the edit (only set when status == "ok")
    #[serde(default)]
    pub total_lines: Option<usize>,
    /// The hash the caller expected the file to have (only set when status == "hash_conflict")
    #[serde(default)]
    pub expected_hash: Option<String>,
    /// The file's actual current hash, or null if the file doesn't exist yet
    /// (only set when status == "hash_conflict")
    #[serde(default)]
    pub actual_hash: Option<String>,
    /// Whether the index was successfully refreshed for this file after the write
    pub reindexed: bool,
    /// Set if the write succeeded but reindexing failed; the affected root is marked
    /// dirty and will be repaired by the next index_codebase call
    #[serde(default)]
    pub warning: Option<String>,
    /// Time taken in milliseconds
    pub duration_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_file_request_validate_empty_path() {
        let req = ReadFileRequest {
            file_path: String::new(),
            start_line: None,
            line_count: None,
            end_line: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_read_file_request_validate_zero_start() {
        let req = ReadFileRequest {
            file_path: "a.rs".to_string(),
            start_line: Some(0),
            line_count: None,
            end_line: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_read_file_request_validate_start_after_end() {
        let req = ReadFileRequest {
            file_path: "a.rs".to_string(),
            start_line: Some(10),
            line_count: None,
            end_line: Some(5),
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_read_file_request_validate_ok() {
        let req = ReadFileRequest {
            file_path: "a.rs".to_string(),
            start_line: Some(1),
            line_count: None,
            end_line: Some(5),
        };
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_read_file_request_serde_roundtrip() {
        let req = ReadFileRequest {
            file_path: "a.rs".to_string(),
            start_line: Some(1),
            line_count: None,
            end_line: Some(5),
        };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: ReadFileRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req.file_path, deserialized.file_path);
        assert_eq!(req.start_line, deserialized.start_line);
        assert_eq!(req.end_line, deserialized.end_line);
    }

    #[test]
    fn test_read_file_request_defaults_when_omitted() {
        let json = r#"{"file_path":"a.rs"}"#;
        let req: ReadFileRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.start_line, None);
        assert_eq!(req.end_line, None);
    }

    #[test]
    fn test_edit_file_request_validate_empty_path() {
        let req = EditFileRequest {
            file_path: String::new(),
            content: "x".to_string(),
            start_line: None,
            end_line: None,
            expected_hash: None,
            project: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_edit_file_request_validate_whole_file_ok() {
        let req = EditFileRequest {
            file_path: "a.rs".to_string(),
            content: "x".to_string(),
            start_line: None,
            end_line: None,
            expected_hash: None,
            project: None,
        };
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_edit_file_request_validate_mixed_none_some_rejected() {
        let req = EditFileRequest {
            file_path: "a.rs".to_string(),
            content: "x".to_string(),
            start_line: Some(1),
            end_line: None,
            expected_hash: None,
            project: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_edit_file_request_validate_range_ok() {
        let req = EditFileRequest {
            file_path: "a.rs".to_string(),
            content: "x".to_string(),
            start_line: Some(2),
            end_line: Some(5),
            expected_hash: None,
            project: None,
        };
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_edit_file_request_validate_insert_only_ok() {
        // start_line == end_line + 1 means "insert before start_line, delete nothing"
        let req = EditFileRequest {
            file_path: "a.rs".to_string(),
            content: "x".to_string(),
            start_line: Some(6),
            end_line: Some(5),
            expected_hash: None,
            project: None,
        };
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_edit_file_request_validate_start_too_far_past_end_rejected() {
        let req = EditFileRequest {
            file_path: "a.rs".to_string(),
            content: "x".to_string(),
            start_line: Some(7),
            end_line: Some(5),
            expected_hash: None,
            project: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_edit_file_request_validate_zero_start_rejected() {
        let req = EditFileRequest {
            file_path: "a.rs".to_string(),
            content: "x".to_string(),
            start_line: Some(0),
            end_line: Some(0),
            expected_hash: None,
            project: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_edit_file_response_serde_roundtrip() {
        let resp = EditFileResponse {
            status: "ok".to_string(),
            file_hash: Some("abc123".to_string()),
            total_lines: Some(42),
            expected_hash: None,
            actual_hash: None,
            reindexed: true,
            warning: None,
            duration_ms: 12,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: EditFileResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp.status, deserialized.status);
        assert_eq!(resp.file_hash, deserialized.file_hash);
        assert_eq!(resp.total_lines, deserialized.total_lines);
        assert_eq!(resp.reindexed, deserialized.reindexed);
    }

    #[test]
    fn test_read_file_response_serde_roundtrip() {
        let resp = ReadFileResponse {
            content: "fn main() {}\n".to_string(),
            file_path: "src/main.rs".to_string(),
            requested_start_line: 1,
            requested_line_count: 30,
            returned_start_line: 1,
            returned_end_line: 1,
            returned_line_count: 1,
            eof: true,
            range_clamped: true,
            content_truncated: false,
            next_start_line: None,
            start_line: 1,
            end_line: 1,
            total_lines: 1,
            truncated: false,
            file_hash: "abc123".to_string(),
            language: Some("Rust".to_string()),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: ReadFileResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp.content, deserialized.content);
        assert_eq!(resp.file_hash, deserialized.file_hash);
        assert_eq!(resp.language, deserialized.language);
    }
}
