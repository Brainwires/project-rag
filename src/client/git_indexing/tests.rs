//! Tests for git history searching

use super::*;
use crate::client::RagClient;
use tempfile::TempDir;

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

#[test]
fn test_parse_date_filter_unix_timestamp() {
    let result = parse_date_filter("1704067200").unwrap();
    assert_eq!(result, 1704067200);
}

#[test]
fn test_parse_date_filter_iso8601() {
    let result = parse_date_filter("2024-01-01T00:00:00Z").unwrap();
    assert_eq!(result, 1704067200);
}

#[test]
fn test_parse_date_filter_invalid() {
    let result = parse_date_filter("invalid");
    assert!(result.is_err());
}

#[test]
fn test_parse_author_line() {
    let (name, email) = parse_author_line("Author: John Doe <john@example.com>");
    assert_eq!(name, "John Doe");
    assert_eq!(email, "john@example.com");
}

#[test]
fn test_parse_author_line_no_email() {
    let (name, email) = parse_author_line("Author: John Doe");
    assert_eq!(name, "John Doe");
    assert_eq!(email, "");
}

#[tokio::test]
async fn test_search_git_history_first_time() {
    // First search should index commits
    let (client, temp_dir) = create_test_client().await;
    let cache_path = temp_dir.path().join("git_cache.json");

    let req = SearchGitHistoryRequest {
        query: "test coverage".to_string(),
        path: ".".to_string(),
        project: None,
        branch: None,
        since: None,
        until: None,
        author: None,
        file_pattern: None,
        max_commits: 5,
        limit: 10,
        min_score: 0.0,
    };

    let result = do_search_git_history(
        client.embedding_provider.clone(),
        client.vector_db.clone(),
        client.git_cache.clone(),
        &cache_path,
        req,
    )
    .await;

    assert!(result.is_ok(), "Git history search should succeed");
    let response = result.unwrap();

    // Should have indexed commits
    assert!(
        response.commits_indexed > 0,
        "Should have indexed commits on first search"
    );
    assert_eq!(
        response.total_cached_commits, response.commits_indexed,
        "Total cached should match indexed on first search"
    );
}

#[tokio::test]
async fn test_search_git_history_second_time_uses_cache() {
    // Second search should use cache and not re-index
    let (client, temp_dir) = create_test_client().await;
    let cache_path = temp_dir.path().join("git_cache.json");

    let req = SearchGitHistoryRequest {
        query: "indexing".to_string(),
        path: ".".to_string(),
        project: None,
        branch: None,
        since: None,
        until: None,
        author: None,
        file_pattern: None,
        max_commits: 5,
        limit: 10,
        min_score: 0.0,
    };

    // First search
    let response1 = do_search_git_history(
        client.embedding_provider.clone(),
        client.vector_db.clone(),
        client.git_cache.clone(),
        &cache_path,
        req.clone(),
    )
    .await
    .unwrap();

    let first_indexed = response1.commits_indexed;
    assert!(first_indexed > 0, "First search should index commits");

    // Second search with same parameters
    let response2 = do_search_git_history(
        client.embedding_provider.clone(),
        client.vector_db.clone(),
        client.git_cache.clone(),
        &cache_path,
        req,
    )
    .await
    .unwrap();

    // Should use cache, not re-index
    assert_eq!(
        response2.commits_indexed, 0,
        "Second search should not re-index (use cache)"
    );
    assert_eq!(
        response2.total_cached_commits, first_indexed,
        "Cache should have commits from first search"
    );
}

#[tokio::test]
async fn test_search_git_history_with_author_filter() {
    let (client, temp_dir) = create_test_client().await;
    let cache_path = temp_dir.path().join("git_cache.json");

    let req = SearchGitHistoryRequest {
        query: "commit".to_string(),
        path: ".".to_string(),
        project: None,
        branch: None,
        since: None,
        until: None,
        author: Some(".*".to_string()), // Match all authors (regex)
        file_pattern: None,
        max_commits: 5,
        limit: 10,
        min_score: 0.0,
    };

    let result = do_search_git_history(
        client.embedding_provider.clone(),
        client.vector_db.clone(),
        client.git_cache.clone(),
        &cache_path,
        req,
    )
    .await;

    assert!(result.is_ok(), "Search with author filter should succeed");
}

#[tokio::test]
async fn test_search_git_history_with_file_pattern_filter() {
    let (client, temp_dir) = create_test_client().await;
    let cache_path = temp_dir.path().join("git_cache.json");

    let req = SearchGitHistoryRequest {
        query: "rust".to_string(),
        path: ".".to_string(),
        project: None,
        branch: None,
        since: None,
        until: None,
        author: None,
        file_pattern: Some(".*\\.rs$".to_string()), // Match .rs files
        max_commits: 5,
        limit: 10,
        min_score: 0.0,
    };

    let result = do_search_git_history(
        client.embedding_provider.clone(),
        client.vector_db.clone(),
        client.git_cache.clone(),
        &cache_path,
        req,
    )
    .await;

    assert!(
        result.is_ok(),
        "Search with file_pattern filter should succeed"
    );
}

#[tokio::test]
async fn test_search_git_history_with_date_filters() {
    let (client, temp_dir) = create_test_client().await;
    let cache_path = temp_dir.path().join("git_cache.json");

    // Use a date range that should include recent commits
    let req = SearchGitHistoryRequest {
        query: "update".to_string(),
        path: ".".to_string(),
        project: None,
        branch: None,
        since: Some("2024-01-01T00:00:00Z".to_string()),
        until: Some("2025-12-31T23:59:59Z".to_string()),
        author: None,
        file_pattern: None,
        max_commits: 5,
        limit: 10,
        min_score: 0.0,
    };

    let result = do_search_git_history(
        client.embedding_provider.clone(),
        client.vector_db.clone(),
        client.git_cache.clone(),
        &cache_path,
        req,
    )
    .await;

    assert!(result.is_ok(), "Search with date filters should succeed");
}

#[tokio::test]
async fn test_search_git_history_with_project_isolation() {
    let (client, temp_dir) = create_test_client().await;
    let cache_path = temp_dir.path().join("git_cache.json");

    let req = SearchGitHistoryRequest {
        query: "feature".to_string(),
        path: ".".to_string(),
        project: Some("test-project".to_string()),
        branch: None,
        since: None,
        until: None,
        author: None,
        file_pattern: None,
        max_commits: 3,
        limit: 5,
        min_score: 0.0,
    };

    let result = do_search_git_history(
        client.embedding_provider.clone(),
        client.vector_db.clone(),
        client.git_cache.clone(),
        &cache_path,
        req,
    )
    .await;

    assert!(
        result.is_ok(),
        "Search with project isolation should succeed"
    );
}

#[tokio::test]
async fn test_search_git_history_incremental_indexing() {
    // Test that requesting more commits triggers incremental indexing
    let (client, temp_dir) = create_test_client().await;
    let cache_path = temp_dir.path().join("git_cache.json");

    // First search with max_commits=2
    let req1 = SearchGitHistoryRequest {
        query: "test".to_string(),
        path: ".".to_string(),
        project: None,
        branch: None,
        since: None,
        until: None,
        author: None,
        file_pattern: None,
        max_commits: 2,
        limit: 10,
        min_score: 0.0,
    };

    let response1 = do_search_git_history(
        client.embedding_provider.clone(),
        client.vector_db.clone(),
        client.git_cache.clone(),
        &cache_path,
        req1,
    )
    .await
    .unwrap();

    let first_cached = response1.total_cached_commits;
    assert!(first_cached <= 2, "Should cache at most 2 commits");

    // Second search with max_commits=5 (more than cached)
    let req2 = SearchGitHistoryRequest {
        query: "test".to_string(),
        path: ".".to_string(),
        project: None,
        branch: None,
        since: None,
        until: None,
        author: None,
        file_pattern: None,
        max_commits: 5,
        limit: 10,
        min_score: 0.0,
    };

    let response2 = do_search_git_history(
        client.embedding_provider.clone(),
        client.vector_db.clone(),
        client.git_cache.clone(),
        &cache_path,
        req2,
    )
    .await
    .unwrap();

    // Should have indexed more commits
    assert!(
        response2.commits_indexed > 0,
        "Should index additional commits when max_commits increases"
    );
    assert!(
        response2.total_cached_commits > first_cached,
        "Total cached should increase"
    );
}

#[tokio::test]
async fn test_search_git_history_response_structure() {
    let (client, temp_dir) = create_test_client().await;
    let cache_path = temp_dir.path().join("git_cache.json");

    let req = SearchGitHistoryRequest {
        query: "refactor".to_string(),
        path: ".".to_string(),
        project: None,
        branch: None,
        since: None,
        until: None,
        author: None,
        file_pattern: None,
        max_commits: 5,
        limit: 10,
        min_score: 0.0,
    };

    let response = do_search_git_history(
        client.embedding_provider.clone(),
        client.vector_db.clone(),
        client.git_cache.clone(),
        &cache_path,
        req,
    )
    .await
    .unwrap();

    // Verify response structure
    assert!(response.duration_ms > 0, "Should have non-zero duration");
    assert!(
        response.total_cached_commits > 0,
        "Should have cached commits"
    );

    // Verify result structure if any results found
    for result in &response.results {
        assert!(!result.commit_hash.is_empty(), "Hash should not be empty");
        assert!(result.score >= 0.0, "Score should be non-negative");
    }
}

#[tokio::test]
async fn test_search_git_history_invalid_path() {
    let (client, temp_dir) = create_test_client().await;
    let cache_path = temp_dir.path().join("git_cache.json");

    let req = SearchGitHistoryRequest {
        query: "test".to_string(),
        path: "/nonexistent/path/to/repo".to_string(),
        project: None,
        branch: None,
        since: None,
        until: None,
        author: None,
        file_pattern: None,
        max_commits: 5,
        limit: 10,
        min_score: 0.0,
    };

    let result = do_search_git_history(
        client.embedding_provider.clone(),
        client.vector_db.clone(),
        client.git_cache.clone(),
        &cache_path,
        req,
    )
    .await;

    // Should error for non-existent path
    assert!(result.is_err(), "Should fail for invalid git repository");
}

#[tokio::test]
async fn test_search_git_history_limit_respected() {
    let (client, temp_dir) = create_test_client().await;
    let cache_path = temp_dir.path().join("git_cache.json");

    let req = SearchGitHistoryRequest {
        query: "commit".to_string(),
        path: ".".to_string(),
        project: None,
        branch: None,
        since: None,
        until: None,
        author: None,
        file_pattern: None,
        max_commits: 10,
        limit: 3, // Limit to 3 results
        min_score: 0.0,
    };

    let response = do_search_git_history(
        client.embedding_provider.clone(),
        client.vector_db.clone(),
        client.git_cache.clone(),
        &cache_path,
        req,
    )
    .await
    .unwrap();

    // Results should not exceed limit
    assert!(
        response.results.len() <= 3,
        "Results should respect limit parameter"
    );
}
