use super::pdf_extractor::extract_pdf_to_markdown;
use anyhow::{Context, Result};
use ignore::WalkBuilder;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

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
        // Programming languages
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

        // Web technologies
        "html" | "htm" => "HTML",
        "css" => "CSS",
        "scss" | "sass" => "SCSS",

        // Data formats and config files
        "json" => "JSON",
        "yaml" | "yml" => "YAML",
        "toml" => "TOML",
        "xml" => "XML",
        "ini" => "INI",
        "conf" | "config" | "cfg" => "Config",
        "properties" => "Properties",
        "env" => "Environment",

        // Documentation formats
        "md" | "markdown" => "Markdown",
        "rst" => "reStructuredText",
        "adoc" | "asciidoc" => "AsciiDoc",
        "org" => "Org Mode",
        "txt" => "Text",
        "log" => "Log",
        "pdf" => "PDF",

        _ => return None,
    };

    Some(lang.to_string())
}

// Note: File hash persistence is now handled by HashCache (src/cache.rs)
// These stub functions were removed as they were replaced by the HashCache implementation.

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_new() {
        let walker = FileWalker::new("/tmp", 1024);
        assert_eq!(walker.root, PathBuf::from("/tmp"));
        assert_eq!(walker.max_file_size, 1024);
        assert!(walker.project.is_none());
        assert!(walker.include_patterns.is_empty());
        assert!(walker.exclude_patterns.is_empty());
    }

    #[test]
    fn test_with_project() {
        let walker = FileWalker::new("/tmp", 1024).with_project(Some("test-project".to_string()));
        assert_eq!(walker.project, Some("test-project".to_string()));
    }

    #[test]
    fn test_with_project_none() {
        let walker = FileWalker::new("/tmp", 1024).with_project(None);
        assert!(walker.project.is_none());
    }

    #[test]
    fn test_with_patterns() {
        let walker = FileWalker::new("/tmp", 1024).with_patterns(
            vec!["*.rs".to_string(), "*.toml".to_string()],
            vec!["target".to_string()],
        );
        assert_eq!(walker.include_patterns, vec!["*.rs", "*.toml"]);
        assert_eq!(walker.exclude_patterns, vec!["target"]);
    }

    #[test]
    fn test_with_patterns_empty() {
        let walker = FileWalker::new("/tmp", 1024).with_patterns(vec![], vec![]);
        assert!(walker.include_patterns.is_empty());
        assert!(walker.exclude_patterns.is_empty());
    }

    #[test]
    fn test_builder_pattern_chaining() {
        let walker = FileWalker::new("/tmp", 1024)
            .with_project(Some("test".to_string()))
            .with_patterns(vec!["*.rs".to_string()], vec!["target".to_string()]);
        assert_eq!(walker.project, Some("test".to_string()));
        assert_eq!(walker.include_patterns, vec!["*.rs"]);
        assert_eq!(walker.exclude_patterns, vec!["target"]);
    }

    #[test]
    fn test_walk_nonexistent_directory() {
        let walker = FileWalker::new("/nonexistent/path/12345", 1024);
        let result = walker.walk();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    #[test]
    fn test_walk_not_a_directory() {
        // Create a temp file
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("notadir.txt");
        fs::write(&file_path, "test").unwrap();

        let walker = FileWalker::new(&file_path, 1024);
        let result = walker.walk();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not a directory"));
    }

    #[test]
    fn test_walk_empty_directory() {
        let temp_dir = TempDir::new().unwrap();
        let walker = FileWalker::new(temp_dir.path(), 1024);
        let files = walker.walk().unwrap();
        assert_eq!(files.len(), 0);
    }

    #[test]
    fn test_walk_simple_directory() {
        let temp_dir = TempDir::new().unwrap();
        let file1 = temp_dir.path().join("test1.txt");
        let file2 = temp_dir.path().join("test2.rs");
        fs::write(&file1, "Hello world").unwrap();
        fs::write(&file2, "fn main() {}").unwrap();

        let walker = FileWalker::new(temp_dir.path(), 1024);
        let files = walker.walk().unwrap();
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn test_walk_nested_directories() {
        let temp_dir = TempDir::new().unwrap();
        let subdir = temp_dir.path().join("subdir");
        fs::create_dir(&subdir).unwrap();
        let file1 = temp_dir.path().join("root.txt");
        let file2 = subdir.join("nested.txt");
        fs::write(&file1, "root").unwrap();
        fs::write(&file2, "nested").unwrap();

        let walker = FileWalker::new(temp_dir.path(), 1024);
        let files = walker.walk().unwrap();
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn test_walk_max_file_size() {
        let temp_dir = TempDir::new().unwrap();
        let small_file = temp_dir.path().join("small.txt");
        let large_file = temp_dir.path().join("large.txt");
        fs::write(&small_file, "small").unwrap();
        fs::write(&large_file, "a".repeat(2000)).unwrap();

        let walker = FileWalker::new(temp_dir.path(), 100);
        let files = walker.walk().unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].path.ends_with("small.txt"));
    }

    #[test]
    fn test_walk_with_include_patterns() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(temp_dir.path().join("test.rs"), "rust").unwrap();
        fs::write(temp_dir.path().join("test.txt"), "text").unwrap();
        fs::write(temp_dir.path().join("test.toml"), "toml").unwrap();

        let walker =
            FileWalker::new(temp_dir.path(), 1024).with_patterns(vec![".rs".to_string()], vec![]);
        let files = walker.walk().unwrap();

        // Debug: print what we found
        eprintln!("Found {} files", files.len());
        for f in &files {
            eprintln!("  - {:?}", f.path);
        }

        assert_eq!(files.len(), 1, "Expected 1 file matching .rs pattern");
        assert!(
            files[0].relative_path.contains(".rs"),
            "Expected file to contain .rs"
        );
    }

    #[test]
    fn test_walk_with_exclude_patterns() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(temp_dir.path().join("include.rs"), "include").unwrap();
        fs::write(temp_dir.path().join("exclude.txt"), "exclude").unwrap();

        let walker =
            FileWalker::new(temp_dir.path(), 1024).with_patterns(vec![], vec![".txt".to_string()]);
        let files = walker.walk().unwrap();
        assert_eq!(files.len(), 1, "Expected 1 file after excluding .txt");
        assert!(
            files[0].relative_path.contains(".rs"),
            "Expected file to contain .rs"
        );
    }

    #[test]
    fn test_walk_with_include_and_exclude_patterns() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(temp_dir.path().join("src.rs"), "source").unwrap();
        fs::write(temp_dir.path().join("test.rs"), "test").unwrap();
        fs::write(temp_dir.path().join("other.txt"), "other").unwrap();

        let walker = FileWalker::new(temp_dir.path(), 1024)
            .with_patterns(vec![".rs".to_string()], vec!["test".to_string()]);
        let files = walker.walk().unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].path.ends_with("src.rs"));
    }

    #[test]
    fn test_walk_file_info_fields() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.rs");
        fs::write(&file_path, "fn main() {}").unwrap();

        let walker =
            FileWalker::new(temp_dir.path(), 1024).with_project(Some("test-proj".to_string()));
        let files = walker.walk().unwrap();
        assert_eq!(files.len(), 1);

        let file_info = &files[0];
        assert_eq!(file_info.path, file_path);
        assert_eq!(file_info.relative_path, "test.rs");
        assert_eq!(file_info.project, Some("test-proj".to_string()));
        assert_eq!(file_info.extension, Some("rs".to_string()));
        assert_eq!(file_info.language, Some("Rust".to_string()));
        assert_eq!(file_info.content, "fn main() {}");
        assert!(!file_info.hash.is_empty());
    }

    #[test]
    fn test_is_text_file_text() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("text.txt");
        fs::write(&file_path, "Hello world\nThis is text").unwrap();

        let walker = FileWalker::new(temp_dir.path(), 1024);
        assert!(walker.is_text_file(&file_path).unwrap());
    }

    #[test]
    fn test_is_text_file_binary() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("binary.bin");
        // Create file with 50% non-printable bytes (exceeds 30% threshold)
        let binary_content: Vec<u8> = (0..100)
            .map(|i| if i % 2 == 0 { 0x00 } else { b'A' })
            .collect();
        fs::write(&file_path, binary_content).unwrap();

        let walker = FileWalker::new(temp_dir.path(), 1024);
        assert!(!walker.is_text_file(&file_path).unwrap());
    }

    #[test]
    fn test_is_text_file_with_newlines_and_tabs() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("text.txt");
        fs::write(&file_path, "Line 1\nLine 2\r\nTabbed\ttext").unwrap();

        let walker = FileWalker::new(temp_dir.path(), 1024);
        assert!(walker.is_text_file(&file_path).unwrap());
    }

    #[test]
    fn test_is_text_file_exactly_threshold() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("threshold.bin");
        // Create file with exactly 30% non-printable bytes (should be text)
        let mut content = vec![b'A'; 70];
        content.extend(vec![0x00; 30]);
        fs::write(&file_path, content).unwrap();

        let walker = FileWalker::new(temp_dir.path(), 1024);
        assert!(!walker.is_text_file(&file_path).unwrap());
    }

    #[test]
    fn test_is_text_file_nonexistent() {
        let walker = FileWalker::new("/tmp", 1024);
        let result = walker.is_text_file(Path::new("/nonexistent/file.txt"));
        assert!(result.is_err());
    }

    #[test]
    fn test_matches_patterns_no_patterns() {
        let walker = FileWalker::new("/tmp", 1024);
        assert!(walker.matches_patterns(Path::new("/tmp/test.rs")));
        assert!(walker.matches_patterns(Path::new("/tmp/test.txt")));
    }

    #[test]
    fn test_matches_patterns_include_match() {
        let walker = FileWalker::new("/tmp", 1024).with_patterns(vec![".rs".to_string()], vec![]);
        assert!(walker.matches_patterns(Path::new("/tmp/test.rs")));
        assert!(!walker.matches_patterns(Path::new("/tmp/test.txt")));
    }

    #[test]
    fn test_matches_patterns_include_multiple() {
        let walker = FileWalker::new("/tmp", 1024)
            .with_patterns(vec![".rs".to_string(), ".toml".to_string()], vec![]);
        assert!(walker.matches_patterns(Path::new("/tmp/test.rs")));
        assert!(walker.matches_patterns(Path::new("/tmp/Cargo.toml")));
        assert!(!walker.matches_patterns(Path::new("/tmp/test.txt")));
    }

    #[test]
    fn test_matches_patterns_exclude_match() {
        let walker =
            FileWalker::new("/tmp", 1024).with_patterns(vec![], vec!["target".to_string()]);
        assert!(walker.matches_patterns(Path::new("/tmp/src/main.rs")));
        assert!(!walker.matches_patterns(Path::new("/tmp/target/debug/main")));
    }

    #[test]
    fn test_matches_patterns_exclude_multiple() {
        let walker = FileWalker::new("/tmp", 1024).with_patterns(
            vec![],
            vec!["target".to_string(), "node_modules".to_string()],
        );
        assert!(walker.matches_patterns(Path::new("/tmp/src/main.rs")));
        assert!(!walker.matches_patterns(Path::new("/tmp/target/debug/main")));
        assert!(!walker.matches_patterns(Path::new("/tmp/node_modules/package.json")));
    }

    #[test]
    fn test_matches_patterns_include_and_exclude() {
        let walker = FileWalker::new("/tmp", 1024)
            .with_patterns(vec![".rs".to_string()], vec!["test".to_string()]);
        assert!(walker.matches_patterns(Path::new("/tmp/src/main.rs")));
        assert!(!walker.matches_patterns(Path::new("/tmp/src/test.rs")));
        assert!(!walker.matches_patterns(Path::new("/tmp/src/main.txt")));
    }

    #[test]
    fn test_calculate_hash_consistency() {
        let walker = FileWalker::new("/tmp", 1024);
        let content = "test content";
        let hash1 = walker.calculate_hash(content);
        let hash2 = walker.calculate_hash(content);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_calculate_hash_different_content() {
        let walker = FileWalker::new("/tmp", 1024);
        let hash1 = walker.calculate_hash("content1");
        let hash2 = walker.calculate_hash("content2");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_calculate_hash_empty_string() {
        let walker = FileWalker::new("/tmp", 1024);
        let hash = walker.calculate_hash("");
        assert!(!hash.is_empty());
        // SHA256 of empty string
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_calculate_hash_format() {
        let walker = FileWalker::new("/tmp", 1024);
        let hash = walker.calculate_hash("test");
        // SHA256 hashes are 64 hex characters
        assert_eq!(hash.len(), 64);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_detect_language_rust() {
        assert_eq!(detect_language("rs"), Some("Rust".to_string()));
    }

    #[test]
    fn test_detect_language_python() {
        assert_eq!(detect_language("py"), Some("Python".to_string()));
    }

    #[test]
    fn test_detect_language_javascript() {
        assert_eq!(detect_language("js"), Some("JavaScript".to_string()));
        assert_eq!(detect_language("mjs"), Some("JavaScript".to_string()));
        assert_eq!(detect_language("cjs"), Some("JavaScript".to_string()));
    }

    #[test]
    fn test_detect_language_typescript() {
        assert_eq!(detect_language("ts"), Some("TypeScript".to_string()));
        assert_eq!(detect_language("tsx"), Some("TypeScript (TSX)".to_string()));
    }

    #[test]
    fn test_detect_language_jsx() {
        assert_eq!(detect_language("jsx"), Some("JavaScript (JSX)".to_string()));
    }

    #[test]
    fn test_detect_language_java() {
        assert_eq!(detect_language("java"), Some("Java".to_string()));
    }

    #[test]
    fn test_detect_language_cpp() {
        assert_eq!(detect_language("cpp"), Some("C++".to_string()));
        assert_eq!(detect_language("cc"), Some("C++".to_string()));
        assert_eq!(detect_language("cxx"), Some("C++".to_string()));
    }

    #[test]
    fn test_detect_language_c() {
        assert_eq!(detect_language("c"), Some("C".to_string()));
    }

    #[test]
    fn test_detect_language_headers() {
        assert_eq!(detect_language("h"), Some("C/C++ Header".to_string()));
        assert_eq!(detect_language("hpp"), Some("C/C++ Header".to_string()));
    }

    #[test]
    fn test_detect_language_go() {
        assert_eq!(detect_language("go"), Some("Go".to_string()));
    }

    #[test]
    fn test_detect_language_ruby() {
        assert_eq!(detect_language("rb"), Some("Ruby".to_string()));
    }

    #[test]
    fn test_detect_language_php() {
        assert_eq!(detect_language("php"), Some("PHP".to_string()));
    }

    #[test]
    fn test_detect_language_swift() {
        assert_eq!(detect_language("swift"), Some("Swift".to_string()));
    }

    #[test]
    fn test_detect_language_kotlin() {
        assert_eq!(detect_language("kt"), Some("Kotlin".to_string()));
        assert_eq!(detect_language("kts"), Some("Kotlin".to_string()));
    }

    #[test]
    fn test_detect_language_scala() {
        assert_eq!(detect_language("scala"), Some("Scala".to_string()));
    }

    #[test]
    fn test_detect_language_shell() {
        assert_eq!(detect_language("sh"), Some("Shell".to_string()));
        assert_eq!(detect_language("bash"), Some("Shell".to_string()));
    }

    #[test]
    fn test_detect_language_sql() {
        assert_eq!(detect_language("sql"), Some("SQL".to_string()));
    }

    #[test]
    fn test_detect_language_html() {
        assert_eq!(detect_language("html"), Some("HTML".to_string()));
        assert_eq!(detect_language("htm"), Some("HTML".to_string()));
    }

    #[test]
    fn test_detect_language_css() {
        assert_eq!(detect_language("css"), Some("CSS".to_string()));
        assert_eq!(detect_language("scss"), Some("SCSS".to_string()));
        assert_eq!(detect_language("sass"), Some("SCSS".to_string()));
    }

    #[test]
    fn test_detect_language_json() {
        assert_eq!(detect_language("json"), Some("JSON".to_string()));
    }

    #[test]
    fn test_detect_language_yaml() {
        assert_eq!(detect_language("yaml"), Some("YAML".to_string()));
        assert_eq!(detect_language("yml"), Some("YAML".to_string()));
    }

    #[test]
    fn test_detect_language_toml() {
        assert_eq!(detect_language("toml"), Some("TOML".to_string()));
    }

    #[test]
    fn test_detect_language_xml() {
        assert_eq!(detect_language("xml"), Some("XML".to_string()));
    }

    #[test]
    fn test_detect_language_markdown() {
        assert_eq!(detect_language("md"), Some("Markdown".to_string()));
        assert_eq!(detect_language("markdown"), Some("Markdown".to_string()));
    }

    #[test]
    fn test_detect_language_text() {
        assert_eq!(detect_language("txt"), Some("Text".to_string()));
    }

    #[test]
    fn test_detect_language_config_files() {
        assert_eq!(detect_language("ini"), Some("INI".to_string()));
        assert_eq!(detect_language("conf"), Some("Config".to_string()));
        assert_eq!(detect_language("config"), Some("Config".to_string()));
        assert_eq!(detect_language("cfg"), Some("Config".to_string()));
        assert_eq!(
            detect_language("properties"),
            Some("Properties".to_string())
        );
        assert_eq!(detect_language("env"), Some("Environment".to_string()));
    }

    #[test]
    fn test_detect_language_documentation() {
        assert_eq!(detect_language("rst"), Some("reStructuredText".to_string()));
        assert_eq!(detect_language("adoc"), Some("AsciiDoc".to_string()));
        assert_eq!(detect_language("asciidoc"), Some("AsciiDoc".to_string()));
        assert_eq!(detect_language("org"), Some("Org Mode".to_string()));
        assert_eq!(detect_language("log"), Some("Log".to_string()));
        assert_eq!(detect_language("pdf"), Some("PDF".to_string()));
    }

    #[test]
    fn test_detect_language_case_insensitive() {
        assert_eq!(detect_language("RS"), Some("Rust".to_string()));
        assert_eq!(detect_language("Py"), Some("Python".to_string()));
        assert_eq!(detect_language("JS"), Some("JavaScript".to_string()));
        assert_eq!(detect_language("TOML"), Some("TOML".to_string()));
        assert_eq!(detect_language("CONF"), Some("Config".to_string()));
    }

    #[test]
    fn test_detect_language_unknown() {
        assert_eq!(detect_language("unknown"), None);
        assert_eq!(detect_language("xyz"), None);
        assert_eq!(detect_language(""), None);
    }

    #[test]
    fn test_walk_skips_binary_files() {
        let temp_dir = TempDir::new().unwrap();
        let text_file = temp_dir.path().join("text.txt");
        let binary_file = temp_dir.path().join("binary.bin");
        fs::write(&text_file, "text content").unwrap();
        // Binary content with >30% non-printable
        fs::write(&binary_file, vec![0x00; 100]).unwrap();

        let walker = FileWalker::new(temp_dir.path(), 1024);
        let files = walker.walk().unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].path.ends_with("text.txt"));
    }

    #[test]
    fn test_walk_skips_invalid_utf8() {
        let temp_dir = TempDir::new().unwrap();
        let valid_file = temp_dir.path().join("valid.txt");
        let invalid_file = temp_dir.path().join("invalid.txt");
        fs::write(&valid_file, "valid UTF-8").unwrap();
        // Invalid UTF-8 sequence
        fs::write(&invalid_file, [0xFF, 0xFE, 0xFD]).unwrap();

        let walker = FileWalker::new(temp_dir.path(), 1024);
        let files = walker.walk().unwrap();
        // Should only find the valid UTF-8 file
        assert_eq!(files.len(), 1);
        assert!(files[0].path.ends_with("valid.txt"));
    }

    #[test]
    fn test_walk_respects_gitignore() {
        let temp_dir = TempDir::new().unwrap();

        // Create .gitignore
        fs::write(temp_dir.path().join(".gitignore"), "ignored.txt\n").unwrap();

        // Create files
        fs::write(temp_dir.path().join("included.txt"), "include").unwrap();
        fs::write(temp_dir.path().join("ignored.txt"), "ignore").unwrap();

        let walker = FileWalker::new(temp_dir.path(), 1024);
        let files = walker.walk().unwrap();

        // Should find included.txt and .gitignore, but NOT ignored.txt (filtered by gitignore)
        let filenames: Vec<_> = files
            .iter()
            .map(|f| f.path.file_name().unwrap().to_str().unwrap())
            .collect();
        assert!(filenames.contains(&"included.txt"));
        assert!(!filenames.contains(&"ignored.txt"));
        assert!(filenames.contains(&".gitignore"));
    }
}
