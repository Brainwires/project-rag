//! Atomic multi-file patch transactions with one coherent reindex publication.

use super::{IndexLockResult, RagClient};
use crate::types::{
    ApplyPatchRequest, ApplyPatchResponse, FilePatch, IndexResponse, IndexingMode, PatchConflict,
    PatchedFileResult,
};
use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextEncoding {
    Utf8,
    Utf8Bom,
    Utf16Le,
    Utf16Be,
}

impl TextEncoding {
    fn label(self) -> &'static str {
        match self {
            Self::Utf8 => "utf-8",
            Self::Utf8Bom => "utf-8-bom",
            Self::Utf16Le => "utf-16le",
            Self::Utf16Be => "utf-16be",
        }
    }
}

#[derive(Debug)]
pub(crate) struct DecodedText {
    pub text: String,
    pub encoding: TextEncoding,
    pub newline: &'static str,
}

pub(crate) fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn newline_style(text: &str) -> &'static str {
    let crlf = text.matches("\r\n").count();
    let lf_only = text.matches('\n').count().saturating_sub(crlf);
    if crlf > 0 && crlf >= lf_only {
        "\r\n"
    } else {
        "\n"
    }
}

pub(crate) fn decode_text_file(bytes: &[u8]) -> Result<DecodedText> {
    let (text, encoding) = if bytes.starts_with(&[0xef, 0xbb, 0xbf]) {
        (
            String::from_utf8(bytes[3..].to_vec()).context("File has invalid UTF-8 after BOM")?,
            TextEncoding::Utf8Bom,
        )
    } else if bytes.starts_with(&[0xff, 0xfe]) {
        if !(bytes.len() - 2).is_multiple_of(2) {
            anyhow::bail!("UTF-16LE file has an odd byte count");
        }
        let units = bytes[2..]
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect::<Vec<_>>();
        (
            String::from_utf16(&units).context("File has invalid UTF-16LE")?,
            TextEncoding::Utf16Le,
        )
    } else if bytes.starts_with(&[0xfe, 0xff]) {
        if !(bytes.len() - 2).is_multiple_of(2) {
            anyhow::bail!("UTF-16BE file has an odd byte count");
        }
        let units = bytes[2..]
            .chunks_exact(2)
            .map(|pair| u16::from_be_bytes([pair[0], pair[1]]))
            .collect::<Vec<_>>();
        (
            String::from_utf16(&units).context("File has invalid UTF-16BE")?,
            TextEncoding::Utf16Be,
        )
    } else {
        if super::file_ops::is_probably_binary(bytes) {
            anyhow::bail!("File appears to be binary");
        }
        (
            String::from_utf8(bytes.to_vec()).context("File is not valid UTF-8")?,
            TextEncoding::Utf8,
        )
    };
    Ok(DecodedText {
        newline: newline_style(&text),
        text,
        encoding,
    })
}

fn encode_text(text: &str, encoding: TextEncoding) -> Vec<u8> {
    match encoding {
        TextEncoding::Utf8 => text.as_bytes().to_vec(),
        TextEncoding::Utf8Bom => [vec![0xef, 0xbb, 0xbf], text.as_bytes().to_vec()].concat(),
        TextEncoding::Utf16Le => {
            let mut bytes = vec![0xff, 0xfe];
            for unit in text.encode_utf16() {
                bytes.extend_from_slice(&unit.to_le_bytes());
            }
            bytes
        }
        TextEncoding::Utf16Be => {
            let mut bytes = vec![0xfe, 0xff];
            for unit in text.encode_utf16() {
                bytes.extend_from_slice(&unit.to_be_bytes());
            }
            bytes
        }
    }
}

fn normalize_newlines(content: &str, newline: &str) -> String {
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
    if newline == "\r\n" {
        normalized.replace('\n', "\r\n")
    } else {
        normalized
    }
}

fn split_lines(text: &str) -> Vec<&str> {
    if text.is_empty() {
        Vec::new()
    } else {
        text.split_inclusive('\n').collect()
    }
}

fn apply_content_patch(current: Option<&str>, patch: &FilePatch, newline: &str) -> Result<String> {
    let content = normalize_newlines(&patch.content, newline);
    match (patch.start_line, patch.end_line) {
        (None, None) => Ok(content),
        (Some(start), Some(end)) => {
            let current = current.ok_or_else(|| {
                anyhow::anyhow!(
                    "Cannot apply a line patch to new file '{}'",
                    patch.file_path
                )
            })?;
            let lines = split_lines(current);
            let start_index = start - 1;
            let end_index = end;
            if start_index > lines.len() || end_index > lines.len() {
                anyhow::bail!(
                    "Line range [{}, {}] exceeds {} lines in '{}'",
                    start,
                    end,
                    lines.len(),
                    patch.file_path
                );
            }
            let mut result = lines[..start_index].concat();
            if !content.is_empty() && start_index > 0 && !lines[start_index - 1].ends_with('\n') {
                result.push_str(newline);
            }
            result.push_str(&content);
            if !content.is_empty() && !content.ends_with('\n') && end_index < lines.len() {
                result.push_str(newline);
            }
            result.push_str(&lines[end_index..].concat());
            Ok(result)
        }
        _ => unreachable!("validated patch range"),
    }
}

#[derive(Debug)]
struct PreparedPatch {
    relative: String,
    path: PathBuf,
    replacement: Option<Vec<u8>>,
    result: PatchedFileResult,
}

fn unique_sibling(path: &Path, suffix: &str, index: usize) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("file");
    path.with_file_name(format!(
        ".{name}.project-rag-{}-{nonce}-{index}.{suffix}",
        std::process::id()
    ))
}

fn transaction_path_identity(path: &Path) -> String {
    let identity = path.to_string_lossy().into_owned();
    #[cfg(windows)]
    return identity.to_lowercase();
    #[cfg(not(windows))]
    identity
}

struct StagedPatch {
    prepared: PreparedPatch,
    temp: Option<PathBuf>,
    backup: PathBuf,
    committed: bool,
}

fn stage_patches(prepared: Vec<PreparedPatch>) -> Result<Vec<StagedPatch>> {
    let mut staged = Vec::with_capacity(prepared.len());
    for (index, prepared) in prepared.into_iter().enumerate() {
        let temp = match (|| -> Result<Option<PathBuf>> {
            let Some(replacement) = &prepared.replacement else {
                return Ok(None);
            };
            let temp = unique_sibling(&prepared.path, "tmp", index);
            let stage_result = (|| -> Result<()> {
                let mut file = OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(&temp)
                    .with_context(|| format!("Failed to stage '{}'", prepared.relative))?;
                file.write_all(replacement)?;
                file.sync_all()?;
                Ok(())
            })();
            if let Err(error) = stage_result {
                let _ = fs::remove_file(&temp);
                return Err(error);
            }
            Ok(Some(temp))
        })() {
            Ok(temp) => temp,
            Err(error) => {
                rollback(&mut staged);
                return Err(error);
            }
        };
        let backup = unique_sibling(&prepared.path, "bak", index);
        staged.push(StagedPatch {
            prepared,
            temp,
            backup,
            committed: false,
        });
    }
    Ok(staged)
}

fn rollback(staged: &mut [StagedPatch]) {
    for item in staged.iter_mut().rev() {
        if item.committed {
            if item.prepared.path.exists() {
                let _ = fs::remove_file(&item.prepared.path);
            }
            if item.backup.exists() {
                let _ = fs::rename(&item.backup, &item.prepared.path);
            }
            item.committed = false;
        }
        if let Some(temp) = &item.temp
            && temp.exists()
        {
            let _ = fs::remove_file(temp);
        }
    }
}

fn commit_staged(staged: &mut [StagedPatch], fail_after: Option<usize>) -> Result<()> {
    for index in 0..staged.len() {
        if fail_after == Some(index) {
            rollback(staged);
            anyhow::bail!("simulated transaction commit failure");
        }
        let relative = staged[index].prepared.relative.clone();
        let operation = (|| -> Result<()> {
            let item = &mut staged[index];
            if item.prepared.path.exists() {
                fs::rename(&item.prepared.path, &item.backup)
                    .with_context(|| format!("Failed to back up '{}'", item.prepared.relative))?;
            }
            if let Some(temp) = item.temp.take()
                && let Err(error) = fs::rename(&temp, &item.prepared.path)
            {
                if item.backup.exists() {
                    let _ = fs::rename(&item.backup, &item.prepared.path);
                }
                return Err(error)
                    .with_context(|| format!("Failed to publish '{}'", item.prepared.relative));
            }
            item.committed = true;
            Ok(())
        })();
        if let Err(error) = operation {
            rollback(staged);
            return Err(error)
                .with_context(|| format!("Atomic transaction failed at '{}'", relative));
        }
    }
    for item in staged.iter() {
        if item.backup.exists()
            && let Err(error) = fs::remove_file(&item.backup)
        {
            tracing::warn!(
                "Committed '{}' but could not remove transaction backup '{}': {}",
                item.prepared.relative,
                item.backup.display(),
                error
            );
        }
    }
    Ok(())
}

impl RagClient {
    pub async fn apply_patch(&self, request: ApplyPatchRequest) -> Result<ApplyPatchResponse> {
        self.apply_patch_internal(request, true).await
    }

    pub(crate) async fn apply_patch_internal(
        &self,
        request: ApplyPatchRequest,
        require_expected_hash: bool,
    ) -> Result<ApplyPatchResponse> {
        let started = Instant::now();
        request.validate().map_err(anyhow::Error::msg)?;

        let mut resolved = Vec::with_capacity(request.patches.len());
        let mut transaction_root = None;
        let mut seen = HashSet::new();
        for patch in &request.patches {
            let (path, root) = self
                .resolve_project_path(&patch.file_path, request.project.as_deref(), true)
                .await?;
            if transaction_root
                .as_ref()
                .is_some_and(|existing| existing != &root)
            {
                anyhow::bail!(
                    "All patches in one transaction must belong to the same project root"
                );
            }
            transaction_root = Some(root);
            if !seen.insert(transaction_path_identity(&path.absolute)) {
                anyhow::bail!("Duplicate canonical patch path: {}", path.relative);
            }
            resolved.push((patch.clone(), path));
        }
        let root = transaction_root.expect("non-empty validated patches");
        let generation_before = self.hash_cache.read().await.generation(&root);

        let lock = if request.dry_run {
            None
        } else {
            Some(match self.try_acquire_index_lock(&root).await? {
                IndexLockResult::Acquired(lock) => lock,
                _ => anyhow::bail!(
                    "Another indexing transaction is active for '{}'; retry shortly",
                    root
                ),
            })
        };

        let mut prepared = Vec::with_capacity(resolved.len());
        let mut conflicts = Vec::new();
        let mut validation_errors = Vec::new();
        for (patch, path) in resolved {
            let original = if path.absolute.exists() {
                Some(
                    fs::read(&path.absolute)
                        .with_context(|| format!("Failed to read '{}'", path.relative))?,
                )
            } else {
                None
            };
            let actual_hash = original.as_deref().map(sha256_bytes);
            if require_expected_hash && original.is_some() && patch.expected_hash.is_none() {
                conflicts.push(PatchConflict {
                    file_path: path.relative.clone(),
                    expected_hash: None,
                    actual_hash: actual_hash.clone(),
                    reason: "expected_hash is required for an existing file".to_string(),
                });
                continue;
            }
            if let Some(expected) = &patch.expected_hash
                && actual_hash.as_deref() != Some(expected)
            {
                conflicts.push(PatchConflict {
                    file_path: path.relative.clone(),
                    expected_hash: Some(expected.clone()),
                    actual_hash: actual_hash.clone(),
                    reason: "content hash mismatch".to_string(),
                });
                continue;
            }
            if patch.delete && original.is_none() {
                validation_errors.push(format!("Cannot delete missing file '{}'", path.relative));
                continue;
            }
            if !patch.delete && original.is_none() && patch.start_line.is_some() {
                validation_errors.push(format!(
                    "Cannot line-patch missing file '{}'",
                    path.relative
                ));
                continue;
            }
            if original.is_none() && path.absolute.parent().is_none_or(|parent| !parent.is_dir()) {
                validation_errors.push(format!(
                    "Parent directory does not exist for '{}'",
                    path.relative
                ));
                continue;
            }

            let (replacement, encoding, newline, total_lines) = if patch.delete {
                (None, "unchanged".to_string(), "unchanged".to_string(), None)
            } else {
                let decoded = if let Some(bytes) = &original {
                    match decode_text_file(bytes) {
                        Ok(decoded) => decoded,
                        Err(error) => {
                            validation_errors.push(format!("{}: {:#}", path.relative, error));
                            continue;
                        }
                    }
                } else {
                    DecodedText {
                        text: String::new(),
                        encoding: TextEncoding::Utf8,
                        newline: newline_style(&patch.content),
                    }
                };
                let updated = match apply_content_patch(
                    original.as_ref().map(|_| decoded.text.as_str()),
                    &patch,
                    decoded.newline,
                ) {
                    Ok(updated) => updated,
                    Err(error) => {
                        validation_errors.push(format!("{}: {:#}", path.relative, error));
                        continue;
                    }
                };
                let bytes = encode_text(&updated, decoded.encoding);
                if bytes.len() > self.config.indexing.max_file_size {
                    validation_errors.push(format!(
                        "{} would be {} bytes, above max_file_size {}",
                        path.relative,
                        bytes.len(),
                        self.config.indexing.max_file_size
                    ));
                    continue;
                }
                (
                    Some(bytes),
                    decoded.encoding.label().to_string(),
                    if decoded.newline == "\r\n" {
                        "crlf"
                    } else {
                        "lf"
                    }
                    .to_string(),
                    Some(split_lines(&updated).len()),
                )
            };
            let operation = if patch.delete {
                "delete"
            } else if original.is_some() {
                "modify"
            } else {
                "create"
            };
            let new_hash = replacement.as_deref().map(sha256_bytes);
            prepared.push(PreparedPatch {
                relative: path.relative.clone(),
                path: path.absolute,
                replacement,
                result: PatchedFileResult {
                    file_path: path.relative,
                    operation: operation.to_string(),
                    previous_hash: actual_hash,
                    new_hash,
                    total_lines,
                    encoding,
                    newline,
                },
            });
        }

        let results = prepared
            .iter()
            .map(|item| item.result.clone())
            .collect::<Vec<_>>();
        if !conflicts.is_empty() || !validation_errors.is_empty() {
            if let Some(lock) = lock {
                lock.release().await;
            }
            return Ok(ApplyPatchResponse {
                status: if conflicts.is_empty() {
                    "validation_failed"
                } else {
                    "conflict"
                }
                .to_string(),
                dry_run: request.dry_run,
                files: results,
                conflicts,
                validation_errors,
                filesystem_committed: false,
                reindexed: false,
                index_stale: false,
                generation_before,
                generation_after: generation_before,
                warning: None,
                duration_ms: started.elapsed().as_millis() as u64,
            });
        }
        if request.dry_run {
            return Ok(ApplyPatchResponse {
                status: "validated".to_string(),
                dry_run: true,
                files: results,
                conflicts,
                validation_errors,
                filesystem_committed: false,
                reindexed: false,
                index_stale: false,
                generation_before,
                generation_after: generation_before,
                warning: None,
                duration_ms: started.elapsed().as_millis() as u64,
            });
        }

        let lock = lock.expect("non-dry-run lock");
        {
            let mut cache = self.hash_cache.write().await;
            cache.mark_dirty(&root);
            cache.save(&self.cache_path)?;
        }
        let mut staged = match stage_patches(prepared) {
            Ok(staged) => staged,
            Err(error) => {
                let mut cache = self.hash_cache.write().await;
                cache.clear_dirty(&root);
                let _ = cache.save(&self.cache_path);
                lock.release().await;
                return Err(error);
            }
        };
        if let Err(error) = commit_staged(&mut staged, None) {
            let mut cache = self.hash_cache.write().await;
            cache.clear_dirty(&root);
            let _ = cache.save(&self.cache_path);
            lock.release().await;
            return Err(error);
        }

        let reindex = crate::client::indexing::do_index_smart_inner(
            self,
            root.clone(),
            request.project.clone(),
            vec![],
            vec![],
            self.config.indexing.max_file_size,
            None,
            None,
            tokio_util::sync::CancellationToken::new(),
            true,
        )
        .await;
        let response = match reindex {
            Ok(index_response) => {
                lock.broadcast_result(&index_response);
                let generation_after = self.hash_cache.read().await.generation(&root);
                ApplyPatchResponse {
                    status: "applied".to_string(),
                    dry_run: false,
                    files: results,
                    conflicts,
                    validation_errors,
                    filesystem_committed: true,
                    reindexed: true,
                    index_stale: false,
                    generation_before,
                    generation_after,
                    warning: None,
                    duration_ms: started.elapsed().as_millis() as u64,
                }
            }
            Err(error) => {
                let failed = IndexResponse {
                    mode: IndexingMode::Incremental,
                    files_indexed: 0,
                    chunks_created: 0,
                    embeddings_generated: 0,
                    duration_ms: 0,
                    errors: vec![format!("Reindex failed: {error:#}")],
                    files_updated: 0,
                    files_removed: 0,
                };
                lock.broadcast_result(&failed);
                ApplyPatchResponse {
                    status: "reindex_failed".to_string(),
                    dry_run: false,
                    files: results,
                    conflicts,
                    validation_errors,
                    filesystem_committed: true,
                    reindexed: false,
                    index_stale: true,
                    generation_before,
                    generation_after: generation_before,
                    warning: Some(format!(
                        "Files committed, but index generation {} remains current and analysis is blocked until reindex: {error:#}",
                        generation_before
                    )),
                    duration_ms: started.elapsed().as_millis() as u64,
                }
            }
        };
        lock.release().await;
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn prepared(path: PathBuf, original: &[u8], replacement: &[u8], name: &str) -> PreparedPatch {
        fs::write(&path, original).unwrap();
        PreparedPatch {
            relative: name.to_string(),
            path,
            replacement: Some(replacement.to_vec()),
            result: PatchedFileResult {
                file_path: name.to_string(),
                operation: "modify".to_string(),
                previous_hash: Some(sha256_bytes(original)),
                new_hash: Some(sha256_bytes(replacement)),
                total_lines: Some(1),
                encoding: "utf-8".to_string(),
                newline: "lf".to_string(),
            },
        }
    }

    #[test]
    fn rollback_restores_every_file_after_mid_commit_failure() {
        let directory = TempDir::new().unwrap();
        let a = directory.path().join("a.txt");
        let b = directory.path().join("b.txt");
        let mut staged = stage_patches(vec![
            prepared(a.clone(), b"old-a", b"new-a", "a.txt"),
            prepared(b.clone(), b"old-b", b"new-b", "b.txt"),
        ])
        .unwrap();
        assert!(commit_staged(&mut staged, Some(1)).is_err());
        assert_eq!(fs::read(a).unwrap(), b"old-a");
        assert_eq!(fs::read(b).unwrap(), b"old-b");
    }

    #[test]
    fn utf16_bom_and_crlf_are_preserved() {
        let original = encode_text("one\r\ntwo\r\n", TextEncoding::Utf16Le);
        let decoded = decode_text_file(&original).unwrap();
        let patch = FilePatch {
            file_path: "a.txt".to_string(),
            content: "changed\n".to_string(),
            start_line: Some(2),
            end_line: Some(2),
            expected_hash: None,
            delete: false,
        };
        let updated = apply_content_patch(Some(&decoded.text), &patch, decoded.newline).unwrap();
        let bytes = encode_text(&updated, decoded.encoding);
        assert!(bytes.starts_with(&[0xff, 0xfe]));
        assert_eq!(decode_text_file(&bytes).unwrap().text, "one\r\nchanged\r\n");
    }

    #[test]
    fn staging_failure_removes_all_transaction_temporaries() {
        let directory = TempDir::new().unwrap();
        let first = directory.path().join("first.txt");
        let missing_parent = directory.path().join("missing").join("second.txt");
        let patches = vec![
            prepared(first, b"old", b"new", "first.txt"),
            PreparedPatch {
                relative: "missing/second.txt".to_string(),
                path: missing_parent,
                replacement: Some(b"new".to_vec()),
                result: PatchedFileResult {
                    file_path: "missing/second.txt".to_string(),
                    operation: "create".to_string(),
                    previous_hash: None,
                    new_hash: Some(sha256_bytes(b"new")),
                    total_lines: Some(1),
                    encoding: "utf-8".to_string(),
                    newline: "lf".to_string(),
                },
            },
        ];

        assert!(stage_patches(patches).is_err());
        let leftovers = fs::read_dir(directory.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains("project-rag"))
            .collect::<Vec<_>>();
        assert!(
            leftovers.is_empty(),
            "leftover staging files: {leftovers:?}"
        );
    }

    #[cfg(windows)]
    #[test]
    fn transaction_identity_rejects_windows_case_aliases() {
        assert_eq!(
            transaction_path_identity(Path::new(r"C:\Project\Src\File.rs")),
            transaction_path_identity(Path::new(r"c:\project\src\file.rs"))
        );
    }

    #[test]
    #[ignore = "manual M5 atomic staging throughput measurement"]
    fn benchmark_m5_stage_and_commit_fifty_files() {
        let directory = TempDir::new().unwrap();
        let prepared = (0..50)
            .map(|index| {
                prepared(
                    directory.path().join(format!("{index}.txt")),
                    b"old\n",
                    b"new\n",
                    &format!("{index}.txt"),
                )
            })
            .collect::<Vec<_>>();
        let started = Instant::now();
        let mut staged = stage_patches(prepared).unwrap();
        commit_staged(&mut staged, None).unwrap();
        println!(
            "m5 patch transaction: files=50 elapsed_ms={}",
            started.elapsed().as_millis()
        );
    }
}
