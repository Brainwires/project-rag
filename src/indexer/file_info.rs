//! File information structure for indexed files

use std::path::PathBuf;

/// Information about a discovered file
#[derive(Debug, Clone)]
pub struct FileInfo {
    pub path: PathBuf,
    pub relative_path: String,
    pub root_path: String,
    pub project: Option<String>,
    pub extension: Option<String>,
    pub language: Option<String>,
    pub content: String,
    pub hash: String,
}
