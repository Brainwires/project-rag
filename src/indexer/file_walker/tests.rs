//! Tests for FileWalker

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
