use super::*;
use tempfile::TempDir;

// Helper to create a test client
async fn create_test_client() -> (RagClient, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();
    (client, temp_dir)
}

// ===== Client Initialization Tests =====

#[tokio::test]
async fn test_new_with_db_path() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");

    let result = RagClient::new_with_db_path(&db_path, cache_path).await;
    assert!(result.is_ok());

    let client = result.unwrap();
    assert_eq!(client.embedding_dimension(), 384);
}

#[tokio::test]
async fn test_client_clone() {
    let (client, _temp_dir) = create_test_client().await;
    let _cloned = client.clone();
    // Should compile and not panic
}

#[tokio::test]
async fn test_config_accessor() {
    let (client, _temp_dir) = create_test_client().await;
    let config = client.config();
    assert!(config.indexing.chunk_size > 0);
}

#[tokio::test]
async fn test_embedding_dimension_accessor() {
    let (client, _temp_dir) = create_test_client().await;
    let dimension = client.embedding_dimension();
    assert_eq!(dimension, 384); // all-MiniLM-L6-v2 has 384 dimensions
}

// ===== normalize_path Tests =====

#[test]
fn test_normalize_path_valid() {
    let result = RagClient::normalize_path(".");
    assert!(result.is_ok());
    let normalized = result.unwrap();
    assert!(!normalized.is_empty());
}

#[test]
fn test_normalize_path_nonexistent() {
    let result = RagClient::normalize_path("/nonexistent/path/12345");
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Failed to canonicalize")
    );
}

#[test]
fn test_normalize_path_absolute() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().to_string_lossy().to_string();

    let result = RagClient::normalize_path(&path);
    assert!(result.is_ok());
    let normalized = result.unwrap();
    assert!(normalized.starts_with('/'));
}

// ===== index_codebase Tests =====

#[tokio::test]
async fn test_index_codebase_empty_directory() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    let request = IndexRequest {
        path: data_dir.to_string_lossy().to_string(),
        project: None,
        include_patterns: vec![],
        exclude_patterns: vec![],
        max_file_size: 1024 * 1024,
    };

    let result = client.index_codebase(request).await;
    assert!(result.is_ok());

    let response = result.unwrap();
    assert_eq!(response.files_indexed, 0);
}

#[tokio::test]
async fn test_index_codebase_with_single_file() {
    let (client, temp_dir) = create_test_client().await;
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("test.rs"), "fn main() {}").unwrap();

    let request = IndexRequest {
        path: data_dir.to_string_lossy().to_string(),
        project: Some("test-project".to_string()),
        include_patterns: vec![],
        exclude_patterns: vec![],
        max_file_size: 1024 * 1024,
    };

    let result = client.index_codebase(request).await;
    assert!(result.is_ok());

    let response = result.unwrap();
    assert_eq!(response.files_indexed, 1);
    assert!(response.chunks_created > 0);
    assert!(response.embeddings_generated > 0);
}

#[tokio::test]
async fn test_index_codebase_validation_failure() {
    let (client, _temp_dir) = create_test_client().await;

    let request = IndexRequest {
        path: "/nonexistent/path".to_string(),
        project: None,
        include_patterns: vec![],
        exclude_patterns: vec![],
        max_file_size: 1024 * 1024,
    };

    let result = client.index_codebase(request).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("does not exist"));
}

// ===== query_codebase Tests =====

#[tokio::test]
async fn test_query_codebase_empty_index() {
    let (client, _temp_dir) = create_test_client().await;

    let request = QueryRequest {
        query: "test query".to_string(),
        path: None,
        project: None,
        limit: 10,
        min_score: 0.7,
        hybrid: true,
    };

    let result = client.query_codebase(request).await;
    assert!(result.is_ok());

    let response = result.unwrap();
    assert_eq!(response.results.len(), 0);
    assert_eq!(response.threshold_used, 0.7);
    assert!(!response.threshold_lowered);
}

#[tokio::test]
async fn test_query_codebase_with_data() {
    let (client, temp_dir) = create_test_client().await;

    // Index some data first
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(
        data_dir.join("test.rs"),
        "fn authenticate_user() { /* authentication logic */ }",
    )
    .unwrap();

    let index_req = IndexRequest {
        path: data_dir.to_string_lossy().to_string(),
        project: Some("test-project".to_string()),
        include_patterns: vec![],
        exclude_patterns: vec![],
        max_file_size: 1024 * 1024,
    };
    client.index_codebase(index_req).await.unwrap();

    // Now query
    let query_req = QueryRequest {
        query: "authentication".to_string(),
        path: None,
        project: Some("test-project".to_string()),
        limit: 10,
        min_score: 0.3,
        hybrid: true,
    };

    let result = client.query_codebase(query_req).await;
    assert!(result.is_ok());

    let response = result.unwrap();
    assert!(response.results.len() > 0);
    assert!(response.duration_ms > 0);
}

#[tokio::test]
async fn test_query_codebase_adaptive_threshold() {
    let (client, temp_dir) = create_test_client().await;

    // Index some data
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("test.rs"), "fn hello() {}").unwrap();

    let index_req = IndexRequest {
        path: data_dir.to_string_lossy().to_string(),
        project: None,
        include_patterns: vec![],
        exclude_patterns: vec![],
        max_file_size: 1024 * 1024,
    };
    client.index_codebase(index_req).await.unwrap();

    // Query with high threshold (might trigger adaptive lowering)
    let query_req = QueryRequest {
        query: "completely unrelated query about databases".to_string(),
        path: None,
        project: None,
        limit: 10,
        min_score: 0.9, // Very high threshold
        hybrid: true,
    };

    let result = client.query_codebase(query_req).await;
    assert!(result.is_ok());
    // Adaptive threshold may or may not lower depending on similarity
}

#[tokio::test]
async fn test_query_codebase_validation_failure() {
    let (client, _temp_dir) = create_test_client().await;

    let request = QueryRequest {
        query: "   ".to_string(), // Empty query
        path: None,
        project: None,
        limit: 10,
        min_score: 0.7,
        hybrid: true,
    };

    let result = client.query_codebase(request).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("cannot be empty"));
}

// ===== search_with_filters Tests =====

#[tokio::test]
async fn test_search_with_filters_empty_index() {
    let (client, _temp_dir) = create_test_client().await;

    let request = AdvancedSearchRequest {
        query: "test".to_string(),
        path: None,
        project: None,
        limit: 10,
        min_score: 0.7,
        file_extensions: vec!["rs".to_string()],
        languages: vec!["Rust".to_string()],
        path_patterns: vec!["src/**".to_string()],
    };

    let result = client.search_with_filters(request).await;
    assert!(result.is_ok());

    let response = result.unwrap();
    assert_eq!(response.results.len(), 0);
}

#[tokio::test]
async fn test_search_with_filters_validation_failure() {
    let (client, _temp_dir) = create_test_client().await;

    let request = AdvancedSearchRequest {
        query: "test".to_string(),
        path: None,
        project: None,
        limit: 10,
        min_score: 0.7,
        file_extensions: vec!["".to_string()], // Invalid
        languages: vec![],
        path_patterns: vec![],
    };

    let result = client.search_with_filters(request).await;
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("file extension cannot be empty")
    );
}

// ===== get_statistics Tests =====

#[tokio::test]
async fn test_get_statistics_empty() {
    let (client, _temp_dir) = create_test_client().await;

    let result = client.get_statistics().await;
    assert!(result.is_ok());

    let response = result.unwrap();
    assert_eq!(response.total_files, 0);
    assert_eq!(response.total_chunks, 0);
    assert_eq!(response.total_embeddings, 0);
}

#[tokio::test]
async fn test_get_statistics_with_data() {
    let (client, temp_dir) = create_test_client().await;

    // Index some data
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("test.rs"), "fn main() {}").unwrap();

    let request = IndexRequest {
        path: data_dir.to_string_lossy().to_string(),
        project: None,
        include_patterns: vec![],
        exclude_patterns: vec![],
        max_file_size: 1024 * 1024,
    };
    client.index_codebase(request).await.unwrap();

    // Get statistics
    let result = client.get_statistics().await;
    assert!(result.is_ok());

    let response = result.unwrap();
    assert!(response.total_files > 0);
    assert!(response.total_chunks > 0);
    assert!(response.total_embeddings > 0);
}

// ===== clear_index Tests =====

#[tokio::test]
async fn test_clear_index_empty() {
    let (client, _temp_dir) = create_test_client().await;

    let result = client.clear_index().await;
    assert!(result.is_ok());

    let response = result.unwrap();
    assert!(response.success);
}

#[tokio::test]
async fn test_clear_index_with_data() {
    let (client, temp_dir) = create_test_client().await;

    // Index some data
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("test.rs"), "fn main() {}").unwrap();

    let request = IndexRequest {
        path: data_dir.to_string_lossy().to_string(),
        project: None,
        include_patterns: vec![],
        exclude_patterns: vec![],
        max_file_size: 1024 * 1024,
    };
    client.index_codebase(request).await.unwrap();

    // Clear the index
    let result = client.clear_index().await;
    assert!(result.is_ok());

    let response = result.unwrap();
    assert!(response.success);
    assert!(response.message.contains("Successfully cleared"));

    // Verify it's empty
    let stats = client.get_statistics().await.unwrap();
    assert_eq!(stats.total_files, 0);
}

// ===== search_git_history Tests =====

#[tokio::test]
async fn test_search_git_history_validation_failure() {
    let (client, _temp_dir) = create_test_client().await;

    let request = SearchGitHistoryRequest {
        query: "  ".to_string(), // Empty query
        path: ".".to_string(),
        project: None,
        branch: None,
        max_commits: 10,
        limit: 10,
        min_score: 0.7,
        author: None,
        since: None,
        until: None,
        file_pattern: None,
    };

    let result = client.search_git_history(request).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("cannot be empty"));
}

#[tokio::test]
async fn test_search_git_history_nonexistent_path() {
    let (client, _temp_dir) = create_test_client().await;

    let request = SearchGitHistoryRequest {
        query: "test".to_string(),
        path: "/nonexistent/path".to_string(),
        project: None,
        branch: None,
        max_commits: 10,
        limit: 10,
        min_score: 0.7,
        author: None,
        since: None,
        until: None,
        file_pattern: None,
    };

    let result = client.search_git_history(request).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("does not exist"));
}

// ===== Integration Tests =====

#[tokio::test]
async fn test_full_workflow_index_query_clear() {
    let (client, temp_dir) = create_test_client().await;

    // Step 1: Index
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(
        data_dir.join("math.rs"),
        "fn add(a: i32, b: i32) -> i32 { a + b }",
    )
    .unwrap();

    let index_req = IndexRequest {
        path: data_dir.to_string_lossy().to_string(),
        project: Some("math-lib".to_string()),
        include_patterns: vec![],
        exclude_patterns: vec![],
        max_file_size: 1024 * 1024,
    };
    let index_resp = client.index_codebase(index_req).await.unwrap();
    assert_eq!(index_resp.files_indexed, 1);

    // Step 2: Query
    let query_req = QueryRequest {
        query: "addition function".to_string(),
        path: None,
        project: Some("math-lib".to_string()),
        limit: 5,
        min_score: 0.3,
        hybrid: true,
    };
    let query_resp = client.query_codebase(query_req).await.unwrap();
    assert!(query_resp.results.len() > 0);

    // Step 3: Statistics
    let stats = client.get_statistics().await.unwrap();
    assert!(stats.total_files > 0);

    // Step 4: Clear
    let clear_resp = client.clear_index().await.unwrap();
    assert!(clear_resp.success);

    // Step 5: Verify empty
    let stats_after = client.get_statistics().await.unwrap();
    assert_eq!(stats_after.total_files, 0);
}

#[tokio::test]
async fn test_project_isolation() {
    let (client, temp_dir) = create_test_client().await;

    // Index for project A
    let data_dir_a = temp_dir.path().join("project_a");
    std::fs::create_dir(&data_dir_a).unwrap();
    std::fs::write(data_dir_a.join("a.rs"), "fn project_a() {}").unwrap();

    let req_a = IndexRequest {
        path: data_dir_a.to_string_lossy().to_string(),
        project: Some("project-a".to_string()),
        include_patterns: vec![],
        exclude_patterns: vec![],
        max_file_size: 1024 * 1024,
    };
    client.index_codebase(req_a).await.unwrap();

    // Index for project B
    let data_dir_b = temp_dir.path().join("project_b");
    std::fs::create_dir(&data_dir_b).unwrap();
    std::fs::write(data_dir_b.join("b.rs"), "fn project_b() {}").unwrap();

    let req_b = IndexRequest {
        path: data_dir_b.to_string_lossy().to_string(),
        project: Some("project-b".to_string()),
        include_patterns: vec![],
        exclude_patterns: vec![],
        max_file_size: 1024 * 1024,
    };
    client.index_codebase(req_b).await.unwrap();

    // Query only project A
    let query_a = QueryRequest {
        query: "project".to_string(),
        path: None,
        project: Some("project-a".to_string()),
        limit: 10,
        min_score: 0.3,
        hybrid: true,
    };
    let results_a = client.query_codebase(query_a).await.unwrap();

    // Results should only be from project A
    for result in results_a.results {
        assert_eq!(result.project, Some("project-a".to_string()));
    }
}
