use super::*;
use tempfile::TempDir;

#[tokio::test]
async fn test_new_creates_server() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");

    let server = RagMcpServer::new_with_db_path(&db_path, cache_path).await;
    assert!(server.is_ok(), "Server creation should succeed");

    let server = server.unwrap();
    assert_eq!(server.embedding_provider.dimension(), 384);
}

#[tokio::test]
async fn test_get_info() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let server = RagMcpServer::new_with_db_path(&db_path, cache_path).await.unwrap();

    let info = server.get_info();

    assert_eq!(info.server_info.name, "project");
    assert!(info.server_info.title.is_some());
    assert!(info.instructions.is_some());
    assert!(info.capabilities.tools.is_some());
    assert!(info.capabilities.prompts.is_some());
}

#[test]
fn test_normalize_path_valid() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().to_string_lossy().to_string();

    let normalized = RagMcpServer::normalize_path(&path);
    assert!(normalized.is_ok());

    let normalized_path = normalized.unwrap();
    assert!(!normalized_path.is_empty());
}

#[test]
fn test_normalize_path_nonexistent() {
    let result = RagMcpServer::normalize_path("/nonexistent/path/12345");
    assert!(result.is_err());
}

#[test]
fn test_normalize_path_current_dir() {
    let result = RagMcpServer::normalize_path(".");
    assert!(result.is_ok());
    let normalized = result.unwrap();
    assert!(!normalized.is_empty());
}

#[tokio::test]
async fn test_do_index_empty_directory() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let server = RagMcpServer::new_with_db_path(&db_path, cache_path).await.unwrap();

    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    let result = server
        .do_index(
            data_dir.to_string_lossy().to_string(),
            None,
            vec![],
            vec![],
            1024 * 1024,
            None,
            None,
        )
        .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.mode, IndexingMode::Full);
    assert_eq!(response.files_indexed, 0);
    assert!(!response.errors.is_empty());
}

#[tokio::test]
async fn test_do_index_with_files() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let server = RagMcpServer::new_with_db_path(&db_path, cache_path).await.unwrap();

    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    // Create a test file
    let test_file = data_dir.join("test.rs");
    std::fs::write(&test_file, "fn main() { println!(\"test\"); }").unwrap();

    let result = server
        .do_index(
            data_dir.to_string_lossy().to_string(),
            Some("test-project".to_string()),
            vec![],
            vec![],
            1024 * 1024,
            None,
            None,
        )
        .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.mode, IndexingMode::Full);
    assert_eq!(response.files_indexed, 1);
    assert!(response.chunks_created > 0);
    assert!(response.embeddings_generated > 0);
}

#[tokio::test]
async fn test_do_index_with_exclude_patterns() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let server = RagMcpServer::new_with_db_path(&db_path, cache_path).await.unwrap();

    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    // Create test files
    std::fs::write(data_dir.join("include.rs"), "fn test() {}").unwrap();
    std::fs::write(data_dir.join("exclude.txt"), "exclude this").unwrap();

    let result = server
        .do_index(
            data_dir.to_string_lossy().to_string(),
            None,
            vec![],
            vec![".txt".to_string()],
            1024 * 1024,
            None,
            None,
        )
        .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.files_indexed, 1); // Only .rs file
}

#[tokio::test]
async fn test_do_incremental_update_no_cache() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let server = RagMcpServer::new_with_db_path(&db_path, cache_path).await.unwrap();

    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    // Create a test file
    std::fs::write(data_dir.join("test.rs"), "fn main() {}").unwrap();

    let result = server
        .do_incremental_update(
            data_dir.to_string_lossy().to_string(),
            None,
            vec![],
            vec![],
            1024 * 1024,
            None,
            None,
        )
        .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.mode, IndexingMode::Incremental);
}

#[tokio::test]
async fn test_do_index_smart_new_codebase() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let server = RagMcpServer::new_with_db_path(&db_path, cache_path).await.unwrap();

    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    std::fs::write(data_dir.join("test.rs"), "fn main() {}").unwrap();

    let result = server
        .do_index_smart(
            data_dir.to_string_lossy().to_string(),
            None,
            vec![],
            vec![],
            1024 * 1024,
            None,
            None,
        )
        .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    // First time should be Full
    assert_eq!(response.mode, IndexingMode::Full);
}

#[tokio::test]
async fn test_server_cloneable() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let server = RagMcpServer::new_with_db_path(&db_path, cache_path).await.unwrap();

    let _cloned = server.clone();
    // Should compile and run without errors
}
