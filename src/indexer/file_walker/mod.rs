//! File walking functionality for directory traversal

use super::file_info::FileInfo;
use super::language::detect_language;
use super::pdf_extractor::extract_pdf_to_markdown;
use anyhow::{Context, Result};
use ignore::WalkBuilder;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub struct FileWalker {
    pub(crate) root: PathBuf,
    pub(crate) project: Option<String>,
    pub(crate) max_file_size: usize,
    pub(crate) include_patterns: Vec<String>,
    pub(crate) exclude_patterns: Vec<String>,
    /// Optional cancellation flag - if set to true, walk() will exit early
    cancelled: Option<Arc<AtomicBool>>,
}

impl FileWalker {
    pub fn new(root: impl AsRef<Path>, max_file_size: usize) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            project: None,
            max_file_size,
            include_patterns: vec![],
            exclude_patterns: vec![],
            cancelled: None,
        }
    }

    /// Set a cancellation flag that will be checked during the walk.
    /// If the flag is set to true, the walk will exit early.
    pub fn with_cancellation_flag(mut self, cancelled: Arc<AtomicBool>) -> Self {
        self.cancelled = Some(cancelled);
        self
    }

    /// Check if cancellation has been requested
    fn is_cancelled(&self) -> bool {
        self.cancelled
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
    }

    pub fn with_project(mut self, project: Option<String>) -> Self {
        self.project = project;
        self
    }

    pub fn with_patterns(
        mut self,
        include_patterns: Vec<String>,
        exclude_patterns: Vec<String>,
    ) -> Self {
        self.include_patterns = include_patterns;
        self.exclude_patterns = exclude_patterns;
        self
    }

    /// Walk the directory and collect all eligible files
    pub fn walk(&self) -> Result<Vec<FileInfo>> {
        // Verify root directory exists
        if !self.root.exists() {
            anyhow::bail!("Root directory does not exist: {:?}", self.root);
        }
        if !self.root.is_dir() {
            anyhow::bail!("Root path is not a directory: {:?}", self.root);
        }

        let mut files = Vec::new();

        let walker = WalkBuilder::new(&self.root)
            .standard_filters(true) // Respect .gitignore, .ignore, etc.
            .hidden(false) // Don't skip hidden files by default
            .git_ignore(true) // Respect .gitignore files
            .git_exclude(true) // Respect .git/info/exclude
            .git_global(true) // Respect global gitignore
            .require_git(false) // Don't require a .git directory
            .build();

        for entry in walker {
            // Check for cancellation at the start of each iteration
            if self.is_cancelled() {
                tracing::info!("File walk cancelled after {} files", files.len());
                anyhow::bail!("Indexing was cancelled");
            }

            let entry = entry.context("Failed to read directory entry")?;
            let path = entry.path();

            // Skip directories
            if path.is_dir() {
                continue;
            }

            // Explicitly skip .git directory contents
            if path.components().any(|c| c.as_os_str() == ".git") {
                tracing::debug!("Skipping .git directory file: {:?}", path);
                continue;
            }

            // Check file size
            if let Ok(metadata) = fs::metadata(path)
                && metadata.len() > self.max_file_size as u64
            {
                tracing::debug!("Skipping large file: {:?}", path);
                continue;
            }

            // Check if file is text (binary detection), but allow PDFs
            let is_pdf = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase() == "pdf")
                .unwrap_or(false);

            if !is_pdf && !self.is_text_file(path)? {
                tracing::debug!("Skipping binary file: {:?}", path);
                continue;
            }

            // Apply include/exclude patterns
            if !self.matches_patterns(path) {
                continue;
            }

            // Read file content - extract text from PDFs or read as UTF-8
            let content = if is_pdf {
                match extract_pdf_to_markdown(path) {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::warn!("Failed to extract PDF {:?}: {}", path, e);
                        continue;
                    }
                }
            } else {
                match fs::read_to_string(path) {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::debug!(
                            "Skipping file that can't be read as UTF-8: {:?}: {}",
                            path,
                            e
                        );
                        continue;
                    }
                }
            };

            // Calculate hash
            let hash = self.calculate_hash(&content);

            // Get relative path
            let relative_path = path
                .strip_prefix(&self.root)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();

            // Detect language
            let extension = path.extension().and_then(|e| e.to_str()).map(String::from);
            let language = extension.as_ref().and_then(|ext| detect_language(ext));

            files.push(FileInfo {
                path: path.to_path_buf(),
                relative_path,
                root_path: self.root.to_string_lossy().to_string(),
                project: self.project.clone(),
                extension,
                language,
                content,
                hash,
            });
        }

        tracing::info!("Found {} files to index", files.len());
        Ok(files)
    }

    /// Check if a file is likely text (not binary)
    pub(crate) fn is_text_file(&self, path: &Path) -> Result<bool> {
        let content = fs::read(path).context("Failed to read file")?;

        // Simple heuristic: if more than 30% of bytes are non-printable, it's binary
        let non_printable = content
            .iter()
            .filter(|&&b| b < 0x20 && b != b'\n' && b != b'\r' && b != b'\t')
            .count();

        Ok((non_printable as f64 / content.len() as f64) < 0.3)
    }

    /// Check if file matches include/exclude patterns
    pub(crate) fn matches_patterns(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();

        // If include patterns are specified, file must match at least one
        if !self.include_patterns.is_empty() {
            let matches_include = self
                .include_patterns
                .iter()
                .any(|pattern| path_str.contains(pattern));
            if !matches_include {
                return false;
            }
        }

        // File must not match any exclude pattern
        if self
            .exclude_patterns
            .iter()
            .any(|pattern| path_str.contains(pattern))
        {
            return false;
        }

        true
    }

    pub(crate) fn calculate_hash(&self, content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

#[cfg(test)]
mod tests;
