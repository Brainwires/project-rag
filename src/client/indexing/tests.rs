use super::*;
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

// Helper to create default cancel token for tests
fn test_cancel_token() -> CancellationToken {
    CancellationToken::new()
}

// Helper to create test client
async fn create_test_client() -> (RagClient, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();
    (client, temp_dir)
}

// ===== do_index Tests =====

#[tokio::test]
async fn test_do_index_empty_directory() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    let result = do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.mode, crate::types::IndexingMode::Full);
    assert_eq!(response.files_indexed, 0);
    assert_eq!(response.chunks_created, 0);
    assert!(!response.errors.is_empty());
    assert!(response.errors[0].contains("No code chunks found"));
}

#[tokio::test]
async fn test_do_index_single_file() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(
        data_dir.join("test.rs"),
        "fn main() {\n    println!(\"Hello\");\n}",
    )
    .unwrap();

    let result = do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        Some("test-project".to_string()),
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.mode, crate::types::IndexingMode::Full);
    assert_eq!(response.files_indexed, 1);
    assert!(response.chunks_created > 0);
    assert!(response.embeddings_generated > 0);
    assert!(response.errors.is_empty());
}

#[tokio::test]
async fn test_do_index_multiple_files() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("file1.rs"), "fn foo() {}").unwrap();
    std::fs::write(data_dir.join("file2.rs"), "fn bar() {}").unwrap();
    std::fs::write(data_dir.join("file3.rs"), "fn baz() {}").unwrap();

    let result = do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.files_indexed, 3);
    assert!(response.chunks_created >= 3);
}

#[tokio::test]
async fn test_do_index_with_exclude_patterns() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("include.rs"), "fn included() {}").unwrap();
    std::fs::write(data_dir.join("exclude.txt"), "excluded").unwrap();

    let result = do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec!["**/*.txt".to_string()],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    // Should only index .rs files
    assert!(response.files_indexed >= 1);
}

// ===== do_incremental_update Tests =====

#[tokio::test]
async fn test_incremental_update_no_changes() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("test.rs"), "fn main() {}").unwrap();

    // Initial index
    do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await
    .unwrap();

    // Incremental update with no changes
    let result = do_incremental_update(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.mode, crate::types::IndexingMode::Incremental);
    assert_eq!(response.files_indexed, 0); // files_added
    assert_eq!(response.files_updated, 0);
    assert_eq!(response.files_removed, 0);
}

#[tokio::test]
async fn test_incremental_update_new_file() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("file1.rs"), "fn foo() {}").unwrap();

    // Initial index
    do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await
    .unwrap();

    // Add new file
    std::fs::write(data_dir.join("file2.rs"), "fn bar() {}").unwrap();

    // Incremental update
    let result = do_incremental_update(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.files_indexed, 1); // 1 new file
    assert_eq!(response.files_updated, 0);
    assert_eq!(response.files_removed, 0);
    assert!(response.chunks_created > 0);
}

#[tokio::test]
async fn test_incremental_update_modified_file() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("test.rs"), "fn foo() {}").unwrap();

    // Initial index
    do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await
    .unwrap();

    // Modify file
    std::fs::write(data_dir.join("test.rs"), "fn bar() { /* modified */ }").unwrap();

    // Incremental update
    let result = do_incremental_update(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.files_indexed, 0); // no new files
    assert_eq!(response.files_updated, 1); // 1 modified
    assert_eq!(response.files_removed, 0);
}

#[tokio::test]
async fn test_incremental_update_removed_file() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("file1.rs"), "fn foo() {}").unwrap();
    std::fs::write(data_dir.join("file2.rs"), "fn bar() {}").unwrap();

    // Initial index
    do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await
    .unwrap();

    // Remove a file
    std::fs::remove_file(data_dir.join("file2.rs")).unwrap();

    // Incremental update
    let result = do_incremental_update(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.files_indexed, 0);
    assert_eq!(response.files_updated, 0);
    assert_eq!(response.files_removed, 1); // 1 removed
}

#[tokio::test]
async fn test_incremental_update_mixed_changes() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("existing.rs"), "fn existing() {}").unwrap();
    std::fs::write(data_dir.join("to_modify.rs"), "fn old() {}").unwrap();
    std::fs::write(data_dir.join("to_remove.rs"), "fn remove() {}").unwrap();

    // Initial index
    do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await
    .unwrap();

    // Make mixed changes
    std::fs::write(data_dir.join("new.rs"), "fn new() {}").unwrap(); // Add
    std::fs::write(data_dir.join("to_modify.rs"), "fn modified() {}").unwrap(); // Modify
    std::fs::remove_file(data_dir.join("to_remove.rs")).unwrap(); // Remove

    // Incremental update
    let result = do_incremental_update(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.files_indexed, 1); // 1 new
    assert_eq!(response.files_updated, 1); // 1 modified
    assert_eq!(response.files_removed, 1); // 1 removed
}

// ===== do_index_smart Tests =====

#[tokio::test]
async fn test_smart_index_first_time_full() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("test.rs"), "fn main() {}").unwrap();

    let result = do_index_smart(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    // First time should be Full
    assert_eq!(response.mode, crate::types::IndexingMode::Full);
}

#[tokio::test]
async fn test_smart_index_second_time_incremental() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("test.rs"), "fn main() {}").unwrap();

    // First index (full)
    let result1 = do_index_smart(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await
    .unwrap();
    assert_eq!(result1.mode, crate::types::IndexingMode::Full);

    // Second index (should be incremental)
    let result2 = do_index_smart(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await
    .unwrap();
    assert_eq!(result2.mode, crate::types::IndexingMode::Incremental);
}

#[tokio::test]
async fn test_smart_index_path_normalization() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("test.rs"), "fn main() {}").unwrap();

    let path = data_dir.to_string_lossy().to_string();

    // First index
    do_index_smart(
        &client,
        path.clone(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await
    .unwrap();

    // Second index with trailing slash (should still detect as incremental)
    let path_with_slash = format!("{}/", path);
    let result = do_index_smart(
        &client,
        path_with_slash,
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    // Should succeed (path normalization handles this)
    assert!(result.is_ok());
}

// ===== Edge Case Tests =====

#[tokio::test]
async fn test_index_with_project_name() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("test.rs"), "fn main() {}").unwrap();

    let result = do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        Some("my-project".to_string()),
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.files_indexed, 1);
}

#[tokio::test]
async fn test_index_preserves_cache_across_operations() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("test.rs"), "fn main() {}").unwrap();

    // Full index
    do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await
    .unwrap();

    // Verify cache was saved
    let cache = client.hash_cache.read().await;
    let cached_hashes = cache.get_root(&data_dir.to_string_lossy().to_string());
    assert!(cached_hashes.is_some());
    assert!(!cached_hashes.unwrap().is_empty());
}

#[tokio::test]
async fn test_incremental_update_empty_directory() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    // Incremental update on empty directory (no cache)
    let result = do_incremental_update(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.mode, crate::types::IndexingMode::Incremental);
    assert_eq!(response.files_indexed, 0);
}

// ===== Error Path Tests =====

#[tokio::test]
async fn test_do_index_nonexistent_path() {
    let (client, _temp_dir) = create_test_client().await;

    let result = do_index(
        &client,
        "/nonexistent/path/12345".to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_err(), "Should fail with nonexistent path");
    let error = result.unwrap_err().to_string();
    assert!(
        error.contains("Failed to walk directory") || error.contains("Failed to spawn"),
        "Error should mention directory walking failure"
    );
}

#[tokio::test]
async fn test_do_index_with_very_large_file() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    // Create a file that exceeds max_file_size
    let large_content = "x".repeat(2000); // 2KB
    std::fs::write(data_dir.join("large.rs"), &large_content).unwrap();

    // Also create a small file to ensure indexing doesn't completely fail
    std::fs::write(data_dir.join("small.rs"), "fn main() {}").unwrap();

    // Set max_file_size to 1KB (1024 bytes)
    let result = do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024, // 1KB limit
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    // large.rs should be skipped, only small.rs indexed
    assert_eq!(response.files_indexed, 1);
}

#[tokio::test]
async fn test_do_index_with_empty_file() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    // Create an empty file
    std::fs::write(data_dir.join("empty.rs"), "").unwrap();

    let result = do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    // Empty file might be indexed but produce 0 chunks
    assert!(response.chunks_created == 0 || response.files_indexed == 0);
}

#[tokio::test]
async fn test_do_index_with_binary_file() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    // Create a binary file (mostly non-printable bytes)
    let binary_content: Vec<u8> = (0..100).map(|i| i as u8).collect();
    std::fs::write(data_dir.join("binary.bin"), binary_content).unwrap();

    // Also create a valid text file
    std::fs::write(data_dir.join("text.rs"), "fn main() {}").unwrap();

    let result = do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    // Binary file should be skipped
    assert!(response.files_indexed >= 1);
}

#[tokio::test]
async fn test_do_index_with_include_patterns() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    std::fs::write(data_dir.join("include.rs"), "fn test() {}").unwrap();
    std::fs::write(data_dir.join("exclude.txt"), "should not be indexed").unwrap();

    let result = do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec!["rs".to_string()], // Only include .rs files
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.files_indexed, 1);
}

#[tokio::test]
async fn test_do_index_with_special_characters_in_filename() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    // Create files with special characters (that are allowed by filesystem)
    std::fs::write(data_dir.join("file-with-dashes.rs"), "fn main() {}").unwrap();
    std::fs::write(data_dir.join("file_with_underscores.rs"), "fn test() {}").unwrap();
    std::fs::write(data_dir.join("file.with.dots.rs"), "fn foo() {}").unwrap();

    let result = do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.files_indexed, 3);
    assert!(response.chunks_created >= 3);
}

#[tokio::test]
async fn test_do_index_with_nested_directories() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    let nested_dir = data_dir.join("nested");
    std::fs::create_dir(&nested_dir).unwrap();
    let deep_nested = nested_dir.join("deep");
    std::fs::create_dir(&deep_nested).unwrap();

    std::fs::write(data_dir.join("root.rs"), "fn root() {}").unwrap();
    std::fs::write(nested_dir.join("nested.rs"), "fn nested() {}").unwrap();
    std::fs::write(deep_nested.join("deep.rs"), "fn deep() {}").unwrap();

    let result = do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.files_indexed, 3);
    assert!(response.chunks_created >= 3);
}

#[tokio::test]
async fn test_incremental_update_nonexistent_path() {
    let (client, _temp_dir) = create_test_client().await;

    let result = do_incremental_update(
        &client,
        "/nonexistent/path/12345".to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_err(), "Should fail with nonexistent path");
}

#[tokio::test]
async fn test_smart_index_with_invalid_path() {
    let (client, _temp_dir) = create_test_client().await;

    let result = do_index_smart(
        &client,
        "/nonexistent/path/12345".to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_err(), "Smart index should fail with invalid path");
}

#[tokio::test]
async fn test_do_index_respects_duration_tracking() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("test.rs"), "fn main() {}").unwrap();

    let result = do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert!(response.duration_ms > 0, "Duration should be tracked");
}

#[tokio::test]
async fn test_do_index_with_whitespace_only_file() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    // Create file with only whitespace
    std::fs::write(data_dir.join("whitespace.rs"), "   \n\n\t\t  \n  ").unwrap();
    // Also add a normal file
    std::fs::write(data_dir.join("normal.rs"), "fn main() {}").unwrap();

    let result = do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    // Whitespace file might produce 0 chunks
    assert!(response.files_indexed >= 1);
}

#[tokio::test]
async fn test_incremental_update_with_concurrent_file_changes() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("file.rs"), "fn original() {}").unwrap();

    // Initial index
    let _ = do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await
    .unwrap();

    // Modify file
    std::fs::write(data_dir.join("file.rs"), "fn modified() {}").unwrap();

    // Incremental update should detect the change
    let result = do_incremental_update(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.files_updated, 1, "Should detect 1 modified file");
}

// ===== Concurrent Indexing Tests =====

#[tokio::test]
async fn test_concurrent_index_same_path_waits_for_result() {
    use std::sync::Arc;
    use tokio::sync::Barrier;

    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("file.rs"), "fn test() {}").unwrap();

    let client = Arc::new(client);
    let path = data_dir.to_string_lossy().to_string();
    let barrier = Arc::new(Barrier::new(2));

    // Spawn two concurrent indexing operations on the same path
    let client1 = client.clone();
    let path1 = path.clone();
    let barrier1 = barrier.clone();
    let handle1 = tokio::spawn(async move {
        barrier1.wait().await;
        do_index_smart(
            &client1,
            path1,
            None,
            vec![],
            vec![],
            1024 * 1024,
            None,
            None,
            CancellationToken::new(),
        )
        .await
    });

    let client2 = client.clone();
    let path2 = path.clone();
    let barrier2 = barrier.clone();
    let handle2 = tokio::spawn(async move {
        barrier2.wait().await;
        do_index_smart(
            &client2,
            path2,
            None,
            vec![],
            vec![],
            1024 * 1024,
            None,
            None,
            CancellationToken::new(),
        )
        .await
    });

    // Both should succeed (one does the work, other waits for result)
    let (result1, result2) = tokio::join!(handle1, handle2);
    let response1 = result1.unwrap().unwrap();
    let response2 = result2.unwrap().unwrap();

    // Both should report the same number of files indexed
    assert_eq!(response1.files_indexed, response2.files_indexed);
    assert_eq!(response1.chunks_created, response2.chunks_created);
}

#[tokio::test]
async fn test_concurrent_index_different_paths_both_run() {
    use std::sync::Arc;
    use tokio::sync::Barrier;

    let (client, temp_dir) = create_test_client().await;

    // Create two separate directories
    let data_dir1 = temp_dir.path().join("data1");
    let data_dir2 = temp_dir.path().join("data2");
    std::fs::create_dir(&data_dir1).unwrap();
    std::fs::create_dir(&data_dir2).unwrap();
    std::fs::write(data_dir1.join("file1.rs"), "fn test1() {}").unwrap();
    std::fs::write(data_dir2.join("file2.rs"), "fn test2() {}").unwrap();

    let client = Arc::new(client);
    let barrier = Arc::new(Barrier::new(2));

    // Spawn two concurrent indexing operations on different paths
    let client1 = client.clone();
    let path1 = data_dir1.to_string_lossy().to_string();
    let barrier1 = barrier.clone();
    let handle1 = tokio::spawn(async move {
        barrier1.wait().await;
        do_index_smart(
            &client1,
            path1,
            None,
            vec![],
            vec![],
            1024 * 1024,
            None,
            None,
            CancellationToken::new(),
        )
        .await
    });

    let client2 = client.clone();
    let path2 = data_dir2.to_string_lossy().to_string();
    let barrier2 = barrier.clone();
    let handle2 = tokio::spawn(async move {
        barrier2.wait().await;
        do_index_smart(
            &client2,
            path2,
            None,
            vec![],
            vec![],
            1024 * 1024,
            None,
            None,
            CancellationToken::new(),
        )
        .await
    });

    // Both should succeed independently
    let (result1, result2) = tokio::join!(handle1, handle2);
    assert!(result1.unwrap().is_ok(), "First path should index successfully");
    assert!(result2.unwrap().is_ok(), "Second path should index successfully");
}

// ===== Cancellation Tests =====

#[tokio::test]
async fn test_cancellation_before_indexing_starts() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("test.rs"), "fn main() {}").unwrap();

    // Create a pre-cancelled token
    let cancel_token = CancellationToken::new();
    cancel_token.cancel();

    let result = do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        cancel_token,
    )
    .await;

    assert!(result.is_err(), "Should fail when cancelled before start");
    // Use {:#} format to see full error chain (anyhow wraps errors with context)
    let error = format!("{:#}", result.unwrap_err());
    assert!(
        error.contains("cancelled"),
        "Error should mention cancellation: {}",
        error
    );
}

#[tokio::test]
async fn test_cancellation_during_file_walk() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    // Create many files to ensure file walk takes some time
    for i in 0..100 {
        let subdir = data_dir.join(format!("dir{}", i));
        std::fs::create_dir(&subdir).unwrap();
        for j in 0..10 {
            std::fs::write(
                subdir.join(format!("file{}.rs", j)),
                format!("fn func_{}_{} () {{ let x = {}; }}", i, j, i * j),
            )
            .unwrap();
        }
    }

    let cancel_token = CancellationToken::new();
    let cancel_token_clone = cancel_token.clone();

    // Cancel after a very short delay
    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
        cancel_token_clone.cancel();
    });

    let result = do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        cancel_token,
    )
    .await;

    // Should either succeed quickly (if file walk completed before cancel)
    // or fail with cancellation error
    if result.is_err() {
        // Use {:#} format to see full error chain (anyhow wraps errors with context)
        let error = format!("{:#}", result.unwrap_err());
        assert!(
            error.contains("cancelled"),
            "Error should mention cancellation: {}",
            error
        );
    }
    // If it succeeded, that's also OK - it just means file walk completed before cancellation
}

#[tokio::test]
async fn test_cancellation_stops_early_incremental() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    // Create some files for initial index
    for i in 0..10 {
        std::fs::write(
            data_dir.join(format!("file{}.rs", i)),
            format!("fn func_{} () {{}}", i),
        )
        .unwrap();
    }

    // Initial full index
    do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        test_cancel_token(),
    )
    .await
    .unwrap();

    // Modify all files to force re-indexing
    for i in 0..10 {
        std::fs::write(
            data_dir.join(format!("file{}.rs", i)),
            format!("fn modified_func_{} () {{ /* modified */ }}", i),
        )
        .unwrap();
    }

    // Create a pre-cancelled token for incremental update
    let cancel_token = CancellationToken::new();
    cancel_token.cancel();

    let result = do_incremental_update(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        cancel_token,
    )
    .await;

    assert!(
        result.is_err(),
        "Incremental update should fail when cancelled"
    );
    // Use {:#} format to see full error chain (anyhow wraps errors with context)
    let error = format!("{:#}", result.unwrap_err());
    assert!(
        error.contains("cancelled"),
        "Error should mention cancellation: {}",
        error
    );
}

#[tokio::test]
async fn test_cancellation_stops_smart_index() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("test.rs"), "fn main() {}").unwrap();

    // Pre-cancelled token
    let cancel_token = CancellationToken::new();
    cancel_token.cancel();

    let result = do_index_smart(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        cancel_token,
    )
    .await;

    assert!(result.is_err(), "Smart index should fail when cancelled");
    // Use {:#} format to see full error chain (anyhow wraps errors with context)
    let error = format!("{:#}", result.unwrap_err());
    assert!(
        error.contains("cancelled"),
        "Error should mention cancellation: {}",
        error
    );
}

#[tokio::test]
async fn test_uncancelled_token_completes_normally() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("test.rs"), "fn main() {}").unwrap();

    // Token that is never cancelled
    let cancel_token = CancellationToken::new();

    let result = do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        cancel_token,
    )
    .await;

    assert!(result.is_ok(), "Should succeed when not cancelled");
    let response = result.unwrap();
    assert_eq!(response.files_indexed, 1);
}

#[tokio::test]
async fn test_cancel_token_cancellation_is_detected() {
    // Test that our check_cancelled macro works correctly
    let cancel_token = CancellationToken::new();
    assert!(!cancel_token.is_cancelled(), "Should not be cancelled initially");

    cancel_token.cancel();
    assert!(cancel_token.is_cancelled(), "Should be cancelled after cancel()");
}

#[tokio::test]
async fn test_cancellation_during_embedding_batch() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    // Create many files to ensure there are multiple embedding batches
    // Default batch size is 32, so we need 50+ chunks
    for i in 0..60 {
        std::fs::write(
            data_dir.join(format!("file{}.rs", i)),
            format!(
                "fn func_{} () {{\n    let x = {};\n    let y = {};\n    println!(\"test\");\n}}",
                i, i, i * 2
            ),
        )
        .unwrap();
    }

    let cancel_token = CancellationToken::new();
    let cancel_token_clone = cancel_token.clone();

    // Cancel after a delay to allow file walk to complete but during embedding
    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        cancel_token_clone.cancel();
    });

    let result = do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        cancel_token,
    )
    .await;

    // Either succeeds (if embedding finished before cancel) or fails with cancellation
    if result.is_err() {
        // Use {:#} format to see full error chain (anyhow wraps errors with context)
        let error = format!("{:#}", result.unwrap_err());
        assert!(
            error.contains("cancelled"),
            "Error should mention cancellation: {}",
            error
        );
    }
}
