use anyhow::{Context, Result};
use ignore::WalkBuilder;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Information about a discovered file
#[derive(Debug, Clone)]
pub struct FileInfo {
    pub path: PathBuf,
    pub relative_path: String,
    pub project: Option<String>,
    pub extension: Option<String>,
    pub language: Option<String>,
    pub content: String,
    pub hash: String,
}

pub struct FileWalker {
    root: PathBuf,
    project: Option<String>,
    max_file_size: usize,
    include_patterns: Vec<String>,
    exclude_patterns: Vec<String>,
}

impl FileWalker {
    pub fn new(root: impl AsRef<Path>, max_file_size: usize) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            project: None,
            max_file_size,
            include_patterns: vec![],
            exclude_patterns: vec![],
        }
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
            let entry = entry.context("Failed to read directory entry")?;
            let path = entry.path();

            // Skip directories
            if path.is_dir() {
                continue;
            }

            // Check file size
            if let Ok(metadata) = fs::metadata(path) {
                if metadata.len() > self.max_file_size as u64 {
                    tracing::debug!("Skipping large file: {:?}", path);
                    continue;
                }
            }

            // Check if file is text (binary detection)
            if !self.is_text_file(path)? {
                tracing::debug!("Skipping binary file: {:?}", path);
                continue;
            }

            // Apply include/exclude patterns
            if !self.matches_patterns(path) {
                continue;
            }

            // Read file content (skip files that can't be read as UTF-8)
            let content = match fs::read_to_string(path) {
                Ok(c) => c,
                Err(e) => {
                    tracing::debug!("Skipping file that can't be read as UTF-8: {:?}: {}", path, e);
                    continue;
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
    fn is_text_file(&self, path: &Path) -> Result<bool> {
        let content = fs::read(path).context("Failed to read file")?;

        // Simple heuristic: if more than 30% of bytes are non-printable, it's binary
        let non_printable = content
            .iter()
            .filter(|&&b| b < 0x20 && b != b'\n' && b != b'\r' && b != b'\t')
            .count();

        Ok((non_printable as f64 / content.len() as f64) < 0.3)
    }

    /// Check if file matches include/exclude patterns
    fn matches_patterns(&self, path: &Path) -> bool {
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

    fn calculate_hash(&self, content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// Detect programming language from file extension
fn detect_language(extension: &str) -> Option<String> {
    let lang = match extension.to_lowercase().as_str() {
        "rs" => "Rust",
        "py" => "Python",
        "js" | "mjs" | "cjs" => "JavaScript",
        "ts" => "TypeScript",
        "jsx" => "JavaScript (JSX)",
        "tsx" => "TypeScript (TSX)",
        "java" => "Java",
        "cpp" | "cc" | "cxx" => "C++",
        "c" => "C",
        "h" | "hpp" => "C/C++ Header",
        "go" => "Go",
        "rb" => "Ruby",
        "php" => "PHP",
        "swift" => "Swift",
        "kt" | "kts" => "Kotlin",
        "scala" => "Scala",
        "sh" | "bash" => "Shell",
        "sql" => "SQL",
        "html" | "htm" => "HTML",
        "css" => "CSS",
        "scss" | "sass" => "SCSS",
        "json" => "JSON",
        "yaml" | "yml" => "YAML",
        "toml" => "TOML",
        "xml" => "XML",
        "md" | "markdown" => "Markdown",
        "txt" => "Text",
        _ => return None,
    };

    Some(lang.to_string())
}

/// Load file hashes from a previous index (for incremental updates)
pub fn load_file_hashes(_root: &Path) -> Result<HashMap<String, String>> {
    // This would read from a cache file in a real implementation
    // For now, return empty HashMap
    Ok(HashMap::new())
}

/// Save file hashes for future incremental updates
pub fn save_file_hashes(_root: &Path, _hashes: HashMap<String, String>) -> Result<()> {
    // This would write to a cache file in a real implementation
    // For now, do nothing
    Ok(())
}
