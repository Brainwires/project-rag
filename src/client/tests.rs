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

#[tokio::test]
async fn test_search_with_filters_with_data() {
    let (client, temp_dir) = create_test_client().await;

    // Create a directory structure for testing filters
    let data_dir = temp_dir.path().join("data");
    let src_dir = data_dir.join("src");
    std::fs::create_dir_all(&src_dir).unwrap();

    // Create Rust file
    std::fs::write(
        src_dir.join("auth.rs"),
        "fn authenticate_user(username: &str) -> bool { true }",
    )
    .unwrap();

    // Create Python file
    std::fs::write(
        src_dir.join("auth.py"),
        "def authenticate_user(username): return True",
    )
    .unwrap();

    // Index the data
    let index_req = IndexRequest {
        path: data_dir.to_string_lossy().to_string(),
        project: Some("filter-test".to_string()),
        include_patterns: vec![],
        exclude_patterns: vec![],
        max_file_size: 1024 * 1024,
    };
    client.index_codebase(index_req).await.unwrap();

    // Search with file extension filter for Rust only
    let request = AdvancedSearchRequest {
        query: "authenticate user".to_string(),
        path: None,
        project: Some("filter-test".to_string()),
        limit: 10,
        min_score: 0.3,
        file_extensions: vec!["rs".to_string()],
        languages: vec![],
        path_patterns: vec![],
    };

    let result = client.search_with_filters(request).await;
    assert!(result.is_ok());

    let response = result.unwrap();
    // Should find results and all should be .rs files
    for result in &response.results {
        assert!(
            result.file_path.ends_with(".rs"),
            "Expected .rs file, got: {}",
            result.file_path
        );
    }
}

#[tokio::test]
async fn test_search_with_filters_adaptive_threshold_lowering() {
    let (client, temp_dir) = create_test_client().await;

    // Index some data
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(
        data_dir.join("code.rs"),
        "fn process_data() { /* some logic here */ }",
    )
    .unwrap();

    let index_req = IndexRequest {
        path: data_dir.to_string_lossy().to_string(),
        project: Some("adaptive-test".to_string()),
        include_patterns: vec![],
        exclude_patterns: vec![],
        max_file_size: 1024 * 1024,
    };
    client.index_codebase(index_req).await.unwrap();

    // Search with high threshold - should trigger adaptive lowering
    let request = AdvancedSearchRequest {
        query: "process data function".to_string(),
        path: None,
        project: Some("adaptive-test".to_string()),
        limit: 10,
        min_score: 0.9, // Very high threshold that will likely not match
        file_extensions: vec![],
        languages: vec![],
        path_patterns: vec![],
    };

    let result = client.search_with_filters(request).await;
    assert!(result.is_ok());

    let response = result.unwrap();
    // If results were found, threshold should have been lowered
    if !response.results.is_empty() {
        assert!(
            response.threshold_lowered,
            "Expected threshold_lowered to be true when results found with high initial threshold"
        );
        assert!(
            response.threshold_used < 0.9,
            "Expected threshold_used ({}) to be lower than initial 0.9",
            response.threshold_used
        );
    }
}

#[tokio::test]
async fn test_search_with_filters_no_adaptive_when_results_found() {
    let (client, temp_dir) = create_test_client().await;

    // Index data with highly relevant content
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(
        data_dir.join("auth.rs"),
        "fn authenticate_user_with_password(username: &str, password: &str) -> Result<User, AuthError> { authenticate(username, password) }",
    )
    .unwrap();

    let index_req = IndexRequest {
        path: data_dir.to_string_lossy().to_string(),
        project: Some("no-adaptive-test".to_string()),
        include_patterns: vec![],
        exclude_patterns: vec![],
        max_file_size: 1024 * 1024,
    };
    client.index_codebase(index_req).await.unwrap();

    // Search with low threshold - should find results without lowering
    let request = AdvancedSearchRequest {
        query: "authenticate user password".to_string(),
        path: None,
        project: Some("no-adaptive-test".to_string()),
        limit: 10,
        min_score: 0.3, // Low threshold
        file_extensions: vec![],
        languages: vec![],
        path_patterns: vec![],
    };

    let result = client.search_with_filters(request).await;
    assert!(result.is_ok());

    let response = result.unwrap();
    if !response.results.is_empty() {
        // Threshold should NOT have been lowered since 0.3 is already low
        assert!(
            !response.threshold_lowered,
            "Expected threshold_lowered to be false when initial threshold is low"
        );
        assert_eq!(
            response.threshold_used, 0.3,
            "Expected threshold_used to remain at 0.3"
        );
    }
}

#[tokio::test]
async fn test_search_with_filters_language_filter() {
    let (client, temp_dir) = create_test_client().await;

    // Create files in different languages
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("main.rs"), "fn main() { println!(\"Hello\"); }").unwrap();
    std::fs::write(data_dir.join("main.py"), "def main(): print('Hello')").unwrap();
    std::fs::write(data_dir.join("main.js"), "function main() { console.log('Hello'); }").unwrap();

    let index_req = IndexRequest {
        path: data_dir.to_string_lossy().to_string(),
        project: Some("lang-filter-test".to_string()),
        include_patterns: vec![],
        exclude_patterns: vec![],
        max_file_size: 1024 * 1024,
    };
    client.index_codebase(index_req).await.unwrap();

    // Search filtering by Rust language only
    let request = AdvancedSearchRequest {
        query: "main function".to_string(),
        path: None,
        project: Some("lang-filter-test".to_string()),
        limit: 10,
        min_score: 0.3,
        file_extensions: vec![],
        languages: vec!["Rust".to_string()],
        path_patterns: vec![],
    };

    let result = client.search_with_filters(request).await;
    assert!(result.is_ok());

    let response = result.unwrap();
    // All results should be Rust
    for result in &response.results {
        assert_eq!(
            result.language, "Rust",
            "Expected Rust language, got: {}",
            result.language
        );
    }
}

#[tokio::test]
async fn test_search_with_filters_path_pattern() {
    let (client, temp_dir) = create_test_client().await;

    // Create nested directory structure
    let data_dir = temp_dir.path().join("data");
    let src_dir = data_dir.join("src");
    let tests_dir = data_dir.join("tests");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::create_dir_all(&tests_dir).unwrap();

    std::fs::write(src_dir.join("lib.rs"), "pub fn add(a: i32, b: i32) -> i32 { a + b }").unwrap();
    std::fs::write(
        tests_dir.join("test_lib.rs"),
        "fn test_add() { assert_eq!(add(1, 2), 3); }",
    )
    .unwrap();

    let index_req = IndexRequest {
        path: data_dir.to_string_lossy().to_string(),
        project: Some("path-pattern-test".to_string()),
        include_patterns: vec![],
        exclude_patterns: vec![],
        max_file_size: 1024 * 1024,
    };
    client.index_codebase(index_req).await.unwrap();

    // Search with path pattern for src only
    let request = AdvancedSearchRequest {
        query: "add function".to_string(),
        path: None,
        project: Some("path-pattern-test".to_string()),
        limit: 10,
        min_score: 0.3,
        file_extensions: vec![],
        languages: vec![],
        path_patterns: vec!["**/src/**".to_string()],
    };

    let result = client.search_with_filters(request).await;
    assert!(result.is_ok());

    let response = result.unwrap();
    // All results should be from src directory
    for result in &response.results {
        assert!(
            result.file_path.contains("src/") || result.file_path.starts_with("src/"),
            "Expected path to contain src/, got: {}",
            result.file_path
        );
    }
}

#[tokio::test]
async fn test_search_with_filters_combined_filters() {
    let (client, temp_dir) = create_test_client().await;

    // Create a complex directory structure
    let data_dir = temp_dir.path().join("data");
    let src_dir = data_dir.join("src");
    let lib_dir = data_dir.join("lib");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::create_dir_all(&lib_dir).unwrap();

    // Rust files in src
    std::fs::write(
        src_dir.join("handler.rs"),
        "pub fn handle_request() -> Response { Response::ok() }",
    )
    .unwrap();

    // Python files in src
    std::fs::write(
        src_dir.join("handler.py"),
        "def handle_request(): return Response.ok()",
    )
    .unwrap();

    // Rust files in lib
    std::fs::write(
        lib_dir.join("utils.rs"),
        "pub fn handle_utils() -> String { String::new() }",
    )
    .unwrap();

    let index_req = IndexRequest {
        path: data_dir.to_string_lossy().to_string(),
        project: Some("combined-filter-test".to_string()),
        include_patterns: vec![],
        exclude_patterns: vec![],
        max_file_size: 1024 * 1024,
    };
    client.index_codebase(index_req).await.unwrap();

    // Search with combined filters: Rust files in src directory only
    let request = AdvancedSearchRequest {
        query: "handle request".to_string(),
        path: None,
        project: Some("combined-filter-test".to_string()),
        limit: 10,
        min_score: 0.3,
        file_extensions: vec!["rs".to_string()],
        languages: vec!["Rust".to_string()],
        path_patterns: vec!["**/src/**".to_string()],
    };

    let result = client.search_with_filters(request).await;
    assert!(result.is_ok());

    let response = result.unwrap();
    // All results should be Rust files in src directory
    for result in &response.results {
        assert!(
            result.file_path.ends_with(".rs"),
            "Expected .rs file, got: {}",
            result.file_path
        );
        assert!(
            result.file_path.contains("src/") || result.file_path.starts_with("src/"),
            "Expected path to contain src/, got: {}",
            result.file_path
        );
        assert_eq!(
            result.language, "Rust",
            "Expected Rust language, got: {}",
            result.language
        );
    }
}

#[tokio::test]
async fn test_search_with_filters_empty_query_validation() {
    let (client, _temp_dir) = create_test_client().await;

    let request = AdvancedSearchRequest {
        query: "   ".to_string(), // Empty/whitespace query
        path: None,
        project: None,
        limit: 10,
        min_score: 0.7,
        file_extensions: vec![],
        languages: vec![],
        path_patterns: vec![],
    };

    let result = client.search_with_filters(request).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("cannot be empty"));
}

#[tokio::test]
async fn test_search_with_filters_threshold_boundary_at_0_3() {
    let (client, temp_dir) = create_test_client().await;

    // Index some data
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("test.rs"), "fn test_function() {}").unwrap();

    let index_req = IndexRequest {
        path: data_dir.to_string_lossy().to_string(),
        project: Some("boundary-test".to_string()),
        include_patterns: vec![],
        exclude_patterns: vec![],
        max_file_size: 1024 * 1024,
    };
    client.index_codebase(index_req).await.unwrap();

    // Search with threshold at 0.3 - should NOT trigger adaptive lowering
    let request = AdvancedSearchRequest {
        query: "completely unrelated xyz abc 123".to_string(),
        path: None,
        project: Some("boundary-test".to_string()),
        limit: 10,
        min_score: 0.3, // At the boundary, should not lower further
        file_extensions: vec![],
        languages: vec![],
        path_patterns: vec![],
    };

    let result = client.search_with_filters(request).await;
    assert!(result.is_ok());

    let response = result.unwrap();
    // Threshold should not be lowered below 0.3
    assert!(
        !response.threshold_lowered,
        "Threshold should not be lowered when already at 0.3"
    );
    assert_eq!(response.threshold_used, 0.3);
}

#[tokio::test]
async fn test_search_with_filters_multiple_extensions() {
    let (client, temp_dir) = create_test_client().await;

    // Create files with different extensions
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("code.rs"), "fn rust_code() {}").unwrap();
    std::fs::write(data_dir.join("code.ts"), "function tsCode() {}").unwrap();
    std::fs::write(data_dir.join("code.py"), "def python_code(): pass").unwrap();

    let index_req = IndexRequest {
        path: data_dir.to_string_lossy().to_string(),
        project: Some("multi-ext-test".to_string()),
        include_patterns: vec![],
        exclude_patterns: vec![],
        max_file_size: 1024 * 1024,
    };
    client.index_codebase(index_req).await.unwrap();

    // Search filtering by multiple extensions
    let request = AdvancedSearchRequest {
        query: "code function".to_string(),
        path: None,
        project: Some("multi-ext-test".to_string()),
        limit: 10,
        min_score: 0.3,
        file_extensions: vec!["rs".to_string(), "ts".to_string()],
        languages: vec![],
        path_patterns: vec![],
    };

    let result = client.search_with_filters(request).await;
    assert!(result.is_ok());

    let response = result.unwrap();
    // All results should be .rs or .ts files (not .py)
    for result in &response.results {
        assert!(
            result.file_path.ends_with(".rs") || result.file_path.ends_with(".ts"),
            "Expected .rs or .ts file, got: {}",
            result.file_path
        );
    }
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
