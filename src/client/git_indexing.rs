//! Git history searching functionality
//!
//! This module provides semantic search over git commit history with on-demand indexing.

use crate::embedding::EmbeddingProvider;
use crate::git::{CommitChunker, GitWalker};
use crate::git_cache::GitCache;
use crate::types::{GitSearchResult, SearchGitHistoryRequest, SearchGitHistoryResponse};
use crate::vector_db::VectorDatabase;
use anyhow::{Context, Result};
use chrono::DateTime;
use regex::Regex;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

/// Helper to search git history with on-demand indexing
pub async fn do_search_git_history<E, V>(
    embedding_provider: Arc<E>,
    vector_db: Arc<V>,
    git_cache: Arc<RwLock<GitCache>>,
    cache_path: &Path,
    req: SearchGitHistoryRequest,
) -> Result<SearchGitHistoryResponse>
where
    E: EmbeddingProvider + Send + Sync,
    V: VectorDatabase + Send + Sync,
{
    let start_time = Instant::now();

    tracing::info!(
        "Git history search: query='{}', path='{}', max_commits={}",
        req.query,
        req.path,
        req.max_commits
    );

    // Discover git repository
    let walker = tokio::task::spawn_blocking({
        let path = req.path.clone();
        move || GitWalker::discover(&path)
    })
    .await
    .context("Failed to spawn blocking task for git discovery")??;

    let repo_path = walker
        .repo_path()
        .to_str()
        .context("Invalid repository path")?
        .to_string();

    tracing::info!("Discovered git repository at: {}", repo_path);

    // Parse date filters if provided
    let since_timestamp = req.since.as_ref().and_then(|s| parse_date_filter(s).ok());

    let until_timestamp = req.until.as_ref().and_then(|s| parse_date_filter(s).ok());

    // Determine which commits to index (on-demand strategy)
    let mut git_cache_guard = git_cache.write().await;
    let cached_commits = git_cache_guard
        .get_repo(&repo_path)
        .cloned()
        .unwrap_or_default();

    let cached_count = cached_commits.len();
    tracing::info!("Found {} cached commits for this repo", cached_count);

    // Decide if we need to index more commits
    let commits_to_index = if cached_count >= req.max_commits {
        tracing::info!("Cache has enough commits, skipping indexing");
        0
    } else {
        req.max_commits - cached_count
    };

    let mut newly_indexed = 0;

    if commits_to_index > 0 {
        tracing::info!("Need to index {} more commits", commits_to_index);

        // Walk git history and extract new commits
        let commits = tokio::task::spawn_blocking({
            let branch = req.branch.clone();
            let max = Some(req.max_commits); // Walk up to max_commits
            move || {
                walker.iter_commits(
                    branch.as_deref(),
                    max,
                    since_timestamp,
                    until_timestamp,
                    &cached_commits,
                )
            }
        })
        .await
        .context("Failed to spawn blocking task for commit iteration")??;

        newly_indexed = commits.len();
        tracing::info!("Extracted {} new commits from git history", newly_indexed);

        if newly_indexed > 0 {
            // Convert commits to chunks
            let chunker = CommitChunker::new();
            let chunks = chunker.commits_to_chunks(&commits, &repo_path, req.project.clone())?;

            tracing::info!("Created {} chunks from commits", chunks.len());

            // Generate embeddings in batches
            let contents: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
            let metadatas = chunks.iter().map(|c| c.metadata.clone()).collect();

            let embeddings = embedding_provider
                .embed_batch(contents.clone())
                .context("Failed to generate embeddings for commits")?;

            tracing::info!("Generated {} embeddings", embeddings.len());

            // Store in vector database (use repo_path for per-project BM25)
            let stored = vector_db
                .store_embeddings(embeddings, metadatas, contents, &repo_path)
                .await
                .context("Failed to store commit embeddings")?;

            tracing::info!("Stored {} commit embeddings in vector database", stored);

            // Update cache with new commit hashes
            let new_hashes: HashSet<String> = commits.iter().map(|c| c.hash.clone()).collect();
            git_cache_guard.add_commits(repo_path.clone(), new_hashes);

            // Persist cache to disk
            git_cache_guard
                .save(cache_path)
                .context("Failed to save git cache")?;

            tracing::info!("Updated git cache with {} new commits", newly_indexed);
        }
    }

    drop(git_cache_guard); // Release write lock before search

    // Generate query embedding
    let query_embeddings = embedding_provider
        .embed_batch(vec![req.query.clone()])
        .context("Failed to generate query embedding")?;

    let query_vector = query_embeddings
        .into_iter()
        .next()
        .context("No query embedding generated")?;

    // Search vector database for git commits
    // Filter by language="git-commit" to only get commits
    let search_results = vector_db
        .search_filtered(
            query_vector,
            &req.query,
            req.limit * 2, // Get more results for post-filtering
            req.min_score,
            req.project.clone(),
            None,                           // root_path
            true,                           // hybrid search
            vec![],                         // no extension filter
            vec!["git-commit".to_string()], // filter by git-commit language
            vec![],                         // no path pattern
        )
        .await
        .context("Failed to search vector database")?;

    tracing::info!("Found {} search results", search_results.len());

    // Post-process results and apply regex filters
    let author_regex = req
        .author
        .as_ref()
        .and_then(|pattern| Regex::new(pattern).ok());

    let file_pattern_regex = req
        .file_pattern
        .as_ref()
        .and_then(|pattern| Regex::new(pattern).ok());

    let mut filtered_results = Vec::new();

    for result in search_results {
        // Parse commit info from file_path (format: git://{repo_path})
        if !result.file_path.starts_with("git://") {
            continue;
        }

        // Extract commit hash from file_hash field
        let commit_hash = result
            .file_path
            .split('/')
            .next_back()
            .unwrap_or(&result.file_path);

        // Parse content to extract commit details
        // Content format: "Commit Message:\n{message}\n\nAuthor: {name} <{email}>\n\nFiles Changed:\n..."
        let parts: Vec<&str> = result.content.splitn(5, "\n\n").collect();

        let commit_message = parts
            .first()
            .and_then(|s| s.strip_prefix("Commit Message:\n"))
            .unwrap_or("")
            .to_string();

        let author_line = parts.get(1).unwrap_or(&"");
        let (author, author_email) = parse_author_line(author_line);

        // Apply author regex filter
        if let Some(ref regex) = author_regex {
            let author_match = regex.is_match(&author) || regex.is_match(&author_email);
            if !author_match {
                continue;
            }
        }

        let files_changed: Vec<String> = if let Some(files_section) = parts.get(2) {
            if files_section.starts_with("Files Changed:") {
                files_section
                    .lines()
                    .skip(1) // Skip "Files Changed:" header
                    .filter_map(|line| line.strip_prefix("- "))
                    .map(|s| s.to_string())
                    .collect()
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        // Apply file pattern regex filter
        if let Some(ref regex) = file_pattern_regex {
            let file_match = files_changed.iter().any(|f| regex.is_match(f));
            if !file_match {
                continue;
            }
        }

        // Extract diff snippet (first ~500 chars of diff)
        let diff_snippet = if let Some(diff_section) = parts.get(3).or(parts.get(4)) {
            if diff_section.starts_with("Diff:") {
                let diff_content = diff_section.strip_prefix("Diff:\n").unwrap_or(diff_section);
                if diff_content.len() > 500 {
                    format!("{}...", &diff_content[..500])
                } else {
                    diff_content.to_string()
                }
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        // Parse commit date from start_line (we stored it there as a hack)
        // Actually, we should get it from the vector DB metadata
        let commit_date = 0; // TODO: Extract from proper metadata

        filtered_results.push(GitSearchResult {
            commit_hash: commit_hash.to_string(),
            commit_message,
            author,
            author_email,
            commit_date,
            score: result.score,
            vector_score: result.vector_score,
            keyword_score: result.keyword_score,
            files_changed,
            diff_snippet,
        });

        if filtered_results.len() >= req.limit {
            break;
        }
    }

    let duration_ms = start_time.elapsed().as_millis() as u64;
    let git_cache_guard = git_cache.read().await;
    let total_cached = git_cache_guard.commit_count(&repo_path);

    Ok(SearchGitHistoryResponse {
        results: filtered_results,
        commits_indexed: newly_indexed,
        total_cached_commits: total_cached,
        duration_ms,
    })
}

/// Parse a date filter string (ISO 8601 or Unix timestamp)
fn parse_date_filter(date_str: &str) -> Result<i64> {
    // Try parsing as Unix timestamp first
    if let Ok(timestamp) = date_str.parse::<i64>() {
        return Ok(timestamp);
    }

    // Try parsing as ISO 8601
    if let Ok(dt) = DateTime::parse_from_rfc3339(date_str) {
        return Ok(dt.timestamp());
    }

    // Try parsing common formats
    if let Ok(dt) = DateTime::parse_from_str(date_str, "%Y-%m-%d") {
        return Ok(dt.timestamp());
    }

    anyhow::bail!("Invalid date format: {}", date_str)
}

/// Parse author line: "Author: Name <email>"
fn parse_author_line(line: &str) -> (String, String) {
    let author_part = line.strip_prefix("Author: ").unwrap_or(line);

    if let Some(email_start) = author_part.find('<')
        && let Some(email_end) = author_part.find('>')
    {
        let name = author_part[..email_start].trim().to_string();
        let email = author_part[email_start + 1..email_end].to_string();
        return (name, email);
    }

    (author_part.trim().to_string(), String::new())
}

#[cfg(test)]
mod tests {
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
}
