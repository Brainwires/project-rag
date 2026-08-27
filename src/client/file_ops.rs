//! Reading and editing individual files inside an already-indexed project root,
//! with automatic incremental reindexing after a successful write.
//!
//! Both operations are scoped to files that already live under an indexed root
//! (tracked in `HashCache.roots`) - this is a deliberate security boundary that
//! prevents reading or writing arbitrary filesystem paths outside a project the
//! user has explicitly asked us to index.

use super::RagClient;
use crate::types::*;
use anyhow::{Context, Result};
#[cfg(test)]
use sha2::{Digest, Sha256};
#[cfg(test)]
use std::path::Path;

/// Maximum number of lines returned by a single read_file call. Larger ranges
/// are capped (not silently dropped - `truncated` is set so the caller can page).
const READ_MAX_LINES: usize = 2000;
const READ_DEFAULT_LINES: usize = 30;

#[derive(Debug, PartialEq, Eq)]
struct ReadWindow {
    returned_start: usize,
    returned_end: usize,
    returned_count: usize,
    eof: bool,
    range_clamped: bool,
    content_truncated: bool,
    next_start: Option<usize>,
}

fn compute_read_window(
    total_lines: usize,
    requested_start: usize,
    requested_count: usize,
) -> ReadWindow {
    let requested_end = requested_start.saturating_add(requested_count - 1);
    let range_clamped =
        total_lines == 0 || requested_start > total_lines || requested_end > total_lines;
    let (returned_start, returned_end, content_truncated) =
        if total_lines == 0 || requested_start > total_lines {
            (0, 0, false)
        } else {
            let valid_end = requested_end.min(total_lines);
            let capped_end = valid_end.min(
                requested_start
                    .saturating_add(READ_MAX_LINES)
                    .saturating_sub(1),
            );
            (requested_start, capped_end, capped_end < valid_end)
        };
    let returned_count = if returned_start == 0 {
        0
    } else {
        returned_end - returned_start + 1
    };
    let eof = returned_end == 0 || returned_end >= total_lines;
    let next_start = if returned_end > 0 && returned_end < total_lines {
        Some(returned_end + 1)
    } else {
        None
    };

    ReadWindow {
        returned_start,
        returned_end,
        returned_count,
        eof,
        range_clamped,
        content_truncated,
        next_start,
    }
}

#[cfg(test)]
fn sha256_hex(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Same heuristic as `FileWalker::is_text_file`: >=30% non-printable bytes means binary.
pub(crate) fn is_probably_binary(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    let non_printable = bytes
        .iter()
        .filter(|&&b| b < 0x20 && b != b'\n' && b != b'\r' && b != b'\t')
        .count();
    (non_printable as f64 / bytes.len() as f64) >= 0.3
}

/// Split text into lines, each slice retaining its original trailing `\n`/`\r\n`
/// (except possibly the last, if the file has no trailing newline). An empty
/// string yields zero lines.
fn split_lines(text: &str) -> Vec<&str> {
    if text.is_empty() {
        Vec::new()
    } else {
        text.split_inclusive('\n').collect()
    }
}

#[cfg(test)]
fn count_lines(text: &str) -> usize {
    split_lines(text).len()
}

/// Splice `request.content` into `current_text` at the requested line range and
/// return the resulting full file content plus its new line count.
#[cfg(test)]
fn build_new_content(
    current_text: Option<&str>,
    request: &EditFileRequest,
) -> Result<(String, usize)> {
    match (request.start_line, request.end_line) {
        (None, None) => {
            let new_full = request.content.clone();
            let total = count_lines(&new_full);
            Ok((new_full, total))
        }
        (Some(start_line), Some(end_line)) => {
            let current_text = current_text.ok_or_else(|| {
                anyhow::anyhow!(
                    "Cannot edit a line range on a file that does not exist yet: {}",
                    request.file_path
                )
            })?;
            let lines = split_lines(current_text);
            // start_idx/end_idx are 0-indexed; the removed range is [start_idx, end_idx).
            // When start_line == end_line + 1 this is an empty range - a pure insertion.
            let start_idx = start_line - 1;
            let end_idx = end_line;
            if start_idx > lines.len() || end_idx > lines.len() {
                anyhow::bail!(
                    "start_line/end_line out of range: file has {} lines, requested [{}, {}]",
                    lines.len(),
                    start_line,
                    end_line
                );
            }

            let mut new_full = String::new();
            new_full.push_str(&lines[..start_idx].concat());
            // If the last kept line has no trailing newline (the original file didn't
            // end with one), give it one before splicing in new content so the two
            // don't run together on the same line.
            if !request.content.is_empty() && start_idx > 0 && !lines[start_idx - 1].ends_with('\n')
            {
                new_full.push('\n');
            }
            new_full.push_str(&request.content);
            // Keep whatever follows on its own line, unless we're deleting to EOF or
            // the caller's content already ends with a newline.
            if !request.content.is_empty()
                && !request.content.ends_with('\n')
                && end_idx < lines.len()
            {
                new_full.push('\n');
            }
            new_full.push_str(&lines[end_idx..].concat());

            let total = count_lines(&new_full);
            Ok((new_full, total))
        }
        _ => unreachable!(
            "EditFileRequest::validate ensures start_line/end_line are both set or both omitted"
        ),
    }
}

impl RagClient {
    /// Read a slice (or all) of a file's current on-disk content.
    ///
    /// The file must live under an already-indexed project root. Returns a
    /// SHA256 hash of the full file that can be passed as `expected_hash` to
    /// `edit_file` to detect concurrent modification.
    pub async fn read_file(&self, request: ReadFileRequest) -> Result<ReadFileResponse> {
        request.validate().map_err(|e| anyhow::anyhow!(e))?;
        let (resolved, _root) = self
            .resolve_project_path(&request.file_path, None, false)
            .await?;
        let canonical_path = resolved.relative;
        let canonical = resolved.absolute;

        let bytes = std::fs::read(&canonical)
            .with_context(|| format!("Failed to read file: {}", request.file_path))?;
        let decoded = super::patching::decode_text_file(&bytes)
            .with_context(|| format!("Cannot decode text file: {}", request.file_path))?;
        let text = decoded.text;

        let lines = split_lines(&text);
        let total_lines = lines.len();

        let requested_start_line = request.start_line.unwrap_or(1);
        let requested_line_count = if let Some(line_count) = request.line_count {
            line_count
        } else if let Some(end_line) = request.end_line {
            end_line.saturating_sub(requested_start_line) + 1
        } else {
            READ_DEFAULT_LINES
        };

        let window = compute_read_window(total_lines, requested_start_line, requested_line_count);
        let returned_start_line = window.returned_start;
        let returned_end_line = window.returned_end;
        let returned_line_count = window.returned_count;

        let content = if returned_line_count == 0 {
            String::new()
        } else {
            lines[returned_start_line - 1..returned_end_line].concat()
        };

        let extension = canonical
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_string());
        let language = extension
            .as_ref()
            .and_then(|ext| crate::indexer::detect_language(ext));

        Ok(ReadFileResponse {
            content,
            file_path: canonical_path,
            requested_start_line,
            requested_line_count,
            returned_start_line,
            returned_end_line,
            returned_line_count,
            eof: window.eof,
            range_clamped: window.range_clamped,
            content_truncated: window.content_truncated,
            next_start_line: window.next_start,
            start_line: returned_start_line,
            end_line: returned_end_line,
            total_lines,
            truncated: window.content_truncated,
            file_hash: super::patching::sha256_bytes(&bytes),
            language,
        })
    }

    /// Replace a line range (or the whole file) with new content, then
    /// incrementally reindex the affected project root.
    ///
    /// The file must live under an already-indexed project root, or (for a
    /// brand new file) its parent directory must. If `expected_hash` is set
    /// and doesn't match the file's current hash, the edit is rejected and
    /// `status: "hash_conflict"` is returned instead of applied. If the write
    /// succeeds but reindexing fails, the write is kept (it's the source of
    /// truth), the affected root is left marked dirty via the same mechanism
    /// `index_codebase` uses, and `reindexed: false` is returned so the caller
    /// knows search results for this file may be stale until the next
    /// `index_codebase` call repairs it.
    pub async fn edit_file(&self, request: EditFileRequest) -> Result<EditFileResponse> {
        request.validate().map_err(|e| anyhow::anyhow!(e))?;
        let patch_response = self
            .apply_patch_internal(
                ApplyPatchRequest {
                    patches: vec![FilePatch {
                        file_path: request.file_path,
                        content: request.content,
                        start_line: request.start_line,
                        end_line: request.end_line,
                        expected_hash: request.expected_hash.clone(),
                        delete: false,
                    }],
                    dry_run: false,
                    project: request.project,
                },
                false,
            )
            .await?;
        if let Some(error) = patch_response.validation_errors.first() {
            anyhow::bail!("{}", error);
        }
        let file = patch_response.files.first();
        let conflict = patch_response.conflicts.first();
        Ok(EditFileResponse {
            status: match patch_response.status.as_str() {
                "conflict" => "hash_conflict".to_string(),
                "applied" | "reindex_failed" => "ok".to_string(),
                _ => patch_response.status,
            },
            file_hash: file.and_then(|file| file.new_hash.clone()),
            total_lines: file.and_then(|file| file.total_lines),
            expected_hash: conflict.and_then(|conflict| conflict.expected_hash.clone()),
            actual_hash: conflict.and_then(|conflict| conflict.actual_hash.clone()),
            reindexed: patch_response.reindexed,
            warning: patch_response.warning,
            duration_ms: patch_response.duration_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_client() -> (RagClient, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
        let cache_path = temp_dir.path().join("cache.json");
        let client = RagClient::new_with_db_path(&db_path, cache_path)
            .await
            .unwrap();
        (client, temp_dir)
    }

    async fn index_dir(client: &RagClient, dir: &Path) {
        let response = client
            .index_codebase(IndexRequest {
                path: dir.to_string_lossy().to_string(),
                project: None,
                include_patterns: vec![],
                exclude_patterns: vec![],
                max_file_size: 1_048_576,
            })
            .await
            .unwrap();
        assert!(
            response.errors.is_empty(),
            "indexing errors: {:?}",
            response.errors
        );
    }

    #[test]
    fn test_is_probably_binary() {
        assert!(!is_probably_binary(b""));
        assert!(!is_probably_binary(b"fn main() {}\n"));
        assert!(is_probably_binary(&[
            0u8, 1, 2, 3, 255, 254, 253, 252, 0, 1
        ]));
    }

    #[test]
    fn test_split_lines_and_count() {
        assert_eq!(split_lines("").len(), 0);
        assert_eq!(count_lines("a\nb\nc\n"), 3);
        assert_eq!(count_lines("a\nb\nc"), 3);
        assert_eq!(count_lines("a"), 1);
    }

    #[test]
    fn test_default_read_window_is_thirty_lines() {
        let window = compute_read_window(100, 1, READ_DEFAULT_LINES);
        assert_eq!(window.returned_start, 1);
        assert_eq!(window.returned_end, 30);
        assert_eq!(window.returned_count, 30);
        assert!(!window.eof);
        assert!(!window.range_clamped);
        assert!(!window.content_truncated);
        assert_eq!(window.next_start, Some(31));
    }

    #[test]
    fn test_read_through_eof_is_clamped_not_truncated() {
        let window = compute_read_window(112, 100, 30);
        assert_eq!(window.returned_start, 100);
        assert_eq!(window.returned_end, 112);
        assert_eq!(window.returned_count, 13);
        assert!(window.eof);
        assert!(window.range_clamped);
        assert!(!window.content_truncated);
        assert_eq!(window.next_start, None);
    }

    #[test]
    fn test_read_beyond_eof_returns_empty_flagged_range() {
        let window = compute_read_window(10, 100, 30);
        assert_eq!(window.returned_count, 0);
        assert_eq!(window.returned_start, 0);
        assert_eq!(window.returned_end, 0);
        assert!(window.eof);
        assert!(window.range_clamped);
        assert!(!window.content_truncated);
    }

    #[test]
    fn test_explicit_large_read_is_allowed_until_hard_limit() {
        let within_limit = compute_read_window(500, 100, 200);
        assert_eq!(within_limit.returned_count, 200);
        assert!(!within_limit.content_truncated);

        let capped = compute_read_window(5_000, 1, 4_000);
        assert_eq!(capped.returned_count, READ_MAX_LINES);
        assert!(capped.content_truncated);
        assert_eq!(capped.next_start, Some(READ_MAX_LINES + 1));
    }

    #[test]
    fn test_build_new_content_whole_file() {
        let req = EditFileRequest {
            file_path: "x.rs".to_string(),
            content: "fn main() {}\n".to_string(),
            start_line: None,
            end_line: None,
            expected_hash: None,
            project: None,
        };
        let (content, total) = build_new_content(None, &req).unwrap();
        assert_eq!(content, "fn main() {}\n");
        assert_eq!(total, 1);
    }

    #[test]
    fn test_build_new_content_replace_range() {
        let original = "line1\nline2\nline3\n";
        let req = EditFileRequest {
            file_path: "x.rs".to_string(),
            content: "REPLACED\n".to_string(),
            start_line: Some(2),
            end_line: Some(2),
            expected_hash: None,
            project: None,
        };
        let (content, total) = build_new_content(Some(original), &req).unwrap();
        assert_eq!(content, "line1\nREPLACED\nline3\n");
        assert_eq!(total, 3);
    }

    #[test]
    fn test_build_new_content_insert_only() {
        let original = "line1\nline2\n";
        // start_line == end_line + 1 -> insert before line 2, delete nothing
        let req = EditFileRequest {
            file_path: "x.rs".to_string(),
            content: "INSERTED\n".to_string(),
            start_line: Some(2),
            end_line: Some(1),
            expected_hash: None,
            project: None,
        };
        let (content, _total) = build_new_content(Some(original), &req).unwrap();
        assert_eq!(content, "line1\nINSERTED\nline2\n");
    }

    #[test]
    fn test_build_new_content_append_past_eof_no_trailing_newline() {
        // File's last line has no trailing newline; appending past EOF must not
        // glue the new content onto the end of that line.
        let original = "line1\nline2";
        let req = EditFileRequest {
            file_path: "x.rs".to_string(),
            content: "line3\n".to_string(),
            start_line: Some(3),
            end_line: Some(2),
            expected_hash: None,
            project: None,
        };
        let (content, total) = build_new_content(Some(original), &req).unwrap();
        assert_eq!(content, "line1\nline2\nline3\n");
        assert_eq!(total, 3);
    }

    #[test]
    fn test_build_new_content_delete_range() {
        let original = "line1\nline2\nline3\n";
        let req = EditFileRequest {
            file_path: "x.rs".to_string(),
            content: String::new(),
            start_line: Some(2),
            end_line: Some(2),
            expected_hash: None,
            project: None,
        };
        let (content, _total) = build_new_content(Some(original), &req).unwrap();
        assert_eq!(content, "line1\nline3\n");
    }

    #[test]
    fn test_build_new_content_out_of_range_errors() {
        let original = "line1\n";
        let req = EditFileRequest {
            file_path: "x.rs".to_string(),
            content: "x".to_string(),
            start_line: Some(5),
            end_line: Some(5),
            expected_hash: None,
            project: None,
        };
        assert!(build_new_content(Some(original), &req).is_err());
    }

    #[tokio::test]
    async fn test_read_file_outside_indexed_root_errors() {
        let (client, temp_dir) = create_test_client().await;
        let file = temp_dir.path().join("orphan.rs");
        std::fs::write(&file, "fn main() {}").unwrap();

        let result = client
            .read_file(ReadFileRequest {
                file_path: file.to_string_lossy().to_string(),
                start_line: None,
                line_count: None,
                end_line: None,
            })
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_read_file_whole_file() {
        let (client, temp_dir) = create_test_client().await;
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        let file = data_dir.join("a.rs");
        std::fs::write(&file, "fn a() {}\nfn b() {}\nfn c() {}\n").unwrap();
        index_dir(&client, &data_dir).await;

        let response = client
            .read_file(ReadFileRequest {
                file_path: file.to_string_lossy().to_string(),
                start_line: None,
                line_count: None,
                end_line: None,
            })
            .await
            .unwrap();

        assert_eq!(response.total_lines, 3);
        assert_eq!(response.start_line, 1);
        assert_eq!(response.end_line, 3);
        assert!(!response.truncated);
        assert_eq!(response.content, "fn a() {}\nfn b() {}\nfn c() {}\n");
        assert_eq!(
            response.file_hash,
            sha256_hex("fn a() {}\nfn b() {}\nfn c() {}\n")
        );
    }

    #[tokio::test]
    async fn test_read_file_range_clamped() {
        let (client, temp_dir) = create_test_client().await;
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        let file = data_dir.join("a.rs");
        std::fs::write(&file, "1\n2\n3\n").unwrap();
        index_dir(&client, &data_dir).await;

        let response = client
            .read_file(ReadFileRequest {
                file_path: file.to_string_lossy().to_string(),
                start_line: Some(2),
                line_count: None,
                end_line: Some(100),
            })
            .await
            .unwrap();

        assert_eq!(response.start_line, 2);
        assert_eq!(response.end_line, 3);
        assert!(response.range_clamped);
        assert!(!response.content_truncated);
        assert!(!response.truncated);
        assert!(response.eof);
        assert_eq!(response.content, "2\n3\n");
    }

    #[tokio::test]
    async fn test_read_file_not_found() {
        let (client, temp_dir) = create_test_client().await;
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        std::fs::write(data_dir.join("a.rs"), "fn a() {}\n").unwrap();
        index_dir(&client, &data_dir).await;

        let missing = data_dir.join("missing.rs");
        let result = client
            .read_file(ReadFileRequest {
                file_path: missing.to_string_lossy().to_string(),
                start_line: None,
                line_count: None,
                end_line: None,
            })
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_edit_file_create_new_file() {
        let (client, temp_dir) = create_test_client().await;
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        std::fs::write(data_dir.join("seed.rs"), "fn seed() {}\n").unwrap();
        index_dir(&client, &data_dir).await;

        let new_file = data_dir.join("new.rs");
        let response = client
            .edit_file(EditFileRequest {
                file_path: new_file.to_string_lossy().to_string(),
                content: "fn brand_new() {}\n".to_string(),
                start_line: None,
                end_line: None,
                expected_hash: None,
                project: None,
            })
            .await
            .unwrap();

        assert_eq!(response.status, "ok");
        assert!(response.reindexed);
        assert_eq!(response.total_lines, Some(1));
        assert_eq!(
            std::fs::read_to_string(&new_file).unwrap(),
            "fn brand_new() {}\n"
        );
    }

    #[tokio::test]
    async fn test_edit_file_replace_range_and_reindex() {
        let (client, temp_dir) = create_test_client().await;
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        let file = data_dir.join("a.rs");
        std::fs::write(&file, "fn a() {}\nfn b() {}\nfn c() {}\n").unwrap();
        index_dir(&client, &data_dir).await;

        let response = client
            .edit_file(EditFileRequest {
                file_path: file.to_string_lossy().to_string(),
                content: "fn b_renamed() {}\n".to_string(),
                start_line: Some(2),
                end_line: Some(2),
                expected_hash: None,
                project: None,
            })
            .await
            .unwrap();

        assert_eq!(response.status, "ok");
        assert!(response.reindexed);
        let on_disk = std::fs::read_to_string(&file).unwrap();
        assert_eq!(on_disk, "fn a() {}\nfn b_renamed() {}\nfn c() {}\n");
        assert_eq!(response.file_hash, Some(sha256_hex(&on_disk)));
    }

    #[tokio::test]
    async fn test_edit_file_hash_conflict() {
        let (client, temp_dir) = create_test_client().await;
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        let file = data_dir.join("a.rs");
        std::fs::write(&file, "fn a() {}\n").unwrap();
        index_dir(&client, &data_dir).await;

        let response = client
            .edit_file(EditFileRequest {
                file_path: file.to_string_lossy().to_string(),
                content: "fn changed() {}\n".to_string(),
                start_line: None,
                end_line: None,
                expected_hash: Some("stale-hash-that-does-not-match".to_string()),
                project: None,
            })
            .await
            .unwrap();

        assert_eq!(response.status, "hash_conflict");
        assert!(!response.reindexed);
        // File on disk must be untouched
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "fn a() {}\n");
    }

    #[tokio::test]
    async fn test_edit_file_matching_hash_succeeds() {
        let (client, temp_dir) = create_test_client().await;
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        let file = data_dir.join("a.rs");
        let original = "fn a() {}\n";
        std::fs::write(&file, original).unwrap();
        index_dir(&client, &data_dir).await;

        let response = client
            .edit_file(EditFileRequest {
                file_path: file.to_string_lossy().to_string(),
                content: "fn changed() {}\n".to_string(),
                start_line: None,
                end_line: None,
                expected_hash: Some(sha256_hex(original)),
                project: None,
            })
            .await
            .unwrap();

        assert_eq!(response.status, "ok");
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "fn changed() {}\n");
    }

    #[tokio::test]
    async fn test_edit_file_content_too_large_rejected() {
        let (client, temp_dir) = create_test_client().await;
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        let file = data_dir.join("a.rs");
        std::fs::write(&file, "fn a() {}\n").unwrap();
        index_dir(&client, &data_dir).await;

        let huge = "x".repeat(client.config().indexing.max_file_size + 1);
        let result = client
            .edit_file(EditFileRequest {
                file_path: file.to_string_lossy().to_string(),
                content: huge,
                start_line: None,
                end_line: None,
                expected_hash: None,
                project: None,
            })
            .await;
        assert!(result.is_err());
        // Original content must be untouched
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "fn a() {}\n");
    }

    #[tokio::test]
    async fn test_edit_file_outside_indexed_root_errors() {
        let (client, temp_dir) = create_test_client().await;
        let file = temp_dir.path().join("orphan.rs");
        std::fs::write(&file, "fn main() {}").unwrap();

        let result = client
            .edit_file(EditFileRequest {
                file_path: file.to_string_lossy().to_string(),
                content: "fn changed() {}\n".to_string(),
                start_line: None,
                end_line: None,
                expected_hash: None,
                project: None,
            })
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_edit_file_binary_rejected() {
        let (client, temp_dir) = create_test_client().await;
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        let file = data_dir.join("a.rs");
        std::fs::write(&file, "fn a() {}\n").unwrap();
        index_dir(&client, &data_dir).await;

        // Overwrite on disk (outside the tool) with binary content, then try to edit it
        std::fs::write(&file, [0u8, 1, 2, 3, 255, 254, 253, 252, 0, 1]).unwrap();

        let result = client
            .edit_file(EditFileRequest {
                file_path: file.to_string_lossy().to_string(),
                content: "fn changed() {}\n".to_string(),
                start_line: None,
                end_line: None,
                expected_hash: None,
                project: None,
            })
            .await;
        assert!(result.is_err());
    }
}
