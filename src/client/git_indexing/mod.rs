//! Git history searching functionality
//!
//! This module provides semantic search over git commit history with on-demand indexing.

use crate::embedding::EmbeddingProvider;
use crate::git::walker::CommitInfo;
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

    let repo_root = std::fs::canonicalize(walker.repo_path())
        .context("Failed to canonicalize repository root")?;
    let repo_path = repo_root.to_string_lossy().to_string();
    let project_root =
        std::fs::canonicalize(crate::project_path::normalize_transport_path(&req.path))
            .with_context(|| {
                format!("Failed to canonicalize history project root: {}", req.path)
            })?;
    let project_prefix = project_root
        .strip_prefix(&repo_root)
        .context("History path is outside the discovered repository")?
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    let history_cache_key = format!("{}::{}", repo_path, project_prefix);

    tracing::info!("Discovered git repository at: {}", repo_path);

    // Parse date filters if provided
    let since_timestamp = req.since.as_ref().and_then(|s| parse_date_filter(s).ok());

    let until_timestamp = req.until.as_ref().and_then(|s| parse_date_filter(s).ok());

    // Determine which commits to index (on-demand strategy)
    let mut git_cache_guard = git_cache.write().await;
    let cached_commits = git_cache_guard
        .get_repo(&history_cache_key)
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
    let mut invalid_records = 0usize;
    let mut diagnostics = Vec::new();

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

        let mut valid_commits = Vec::new();
        for commit in commits {
            if let Err(error) = validate_commit(&commit) {
                invalid_records += 1;
                diagnostics.push(error.to_string());
                continue;
            }
            if let Some(commit) = map_commit_to_project(commit, &project_prefix) {
                valid_commits.push(commit);
            }
        }
        newly_indexed = valid_commits.len();
        tracing::info!("Extracted {} new commits from git history", newly_indexed);

        if !diagnostics.is_empty() {
            git_cache_guard.record_invalid(&diagnostics);
        }

        if newly_indexed > 0 {
            // Convert commits to chunks
            let chunker = CommitChunker::new();
            let project_root_string = project_root.to_string_lossy().to_string();
            let chunks = chunker.commits_to_chunks(
                &valid_commits,
                &project_root_string,
                req.project.clone(),
            )?;

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
                .store_embeddings(embeddings, metadatas, contents, &project_root_string)
                .await
                .context("Failed to store commit embeddings")?;

            tracing::info!("Stored {} commit embeddings in vector database", stored);

            // Update cache with new commit hashes
            let new_hashes: HashSet<String> =
                valid_commits.iter().map(|c| c.hash.clone()).collect();
            git_cache_guard.add_commits(history_cache_key.clone(), new_hashes);

            // Persist cache to disk
            git_cache_guard
                .save(cache_path)
                .context("Failed to save git cache")?;

            tracing::info!("Updated git cache with {} new commits", newly_indexed);
        } else if !diagnostics.is_empty() {
            git_cache_guard
                .save(cache_path)
                .context("Failed to save Git indexing diagnostics")?;
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
            Some(project_root.to_string_lossy().to_string()),
            true,                           // hybrid search
            vec![],                         // no extension filter
            vec!["git-commit".to_string()], // filter by git-commit language
            vec![],                         // no path pattern
            crate::types::RecordOrigin::History,
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
        if result.origin != crate::types::RecordOrigin::History {
            continue;
        }

        let Some(commit_hash) = result.source_id.clone() else {
            invalid_records += 1;
            diagnostics.push("History result is missing a commit object id".to_string());
            continue;
        };
        if git2::Oid::from_str(&commit_hash).is_err() || result.indexed_at == 0 {
            invalid_records += 1;
            diagnostics.push(format!(
                "Excluded invalid history result: commit_hash='{}', commit_date={}",
                commit_hash, result.indexed_at
            ));
            continue;
        }

        // Parse content to extract commit details
        // Content format: "Commit Message:\n{message}\n\nAuthor: {name} <{email}>\n\nFiles Changed:\n..."
        let parts: Vec<&str> = result.content.splitn(2, "\n\n").collect();

        let commit_message = parts
            .first()
            .and_then(|s| s.strip_prefix("Commit Message:\n"))
            .unwrap_or("")
            .to_string();

        let author_line = result
            .content
            .lines()
            .find(|line| line.starts_with("Author: "))
            .unwrap_or("");
        let (author, author_email) = parse_author_line(author_line);
        let author_date = content_line_value(&result.content, "Author Date: ")
            .and_then(|value| value.parse().ok())
            .unwrap_or_default();
        if author_date == 0 {
            invalid_records += 1;
            diagnostics.push(format!(
                "Excluded history result {} with invalid author_date",
                commit_hash
            ));
            continue;
        }
        let subject = content_line_value(&result.content, "Subject: ")
            .unwrap_or_else(|| commit_message.lines().next().unwrap_or("").to_string());

        // Apply author regex filter
        if let Some(ref regex) = author_regex {
            let author_match = regex.is_match(&author) || regex.is_match(&author_email);
            if !author_match {
                continue;
            }
        }

        let files_changed: Vec<String> = result
            .content
            .lines()
            .skip_while(|line| *line != "Files Changed:")
            .skip(1)
            .take_while(|line| line.starts_with("- "))
            .filter_map(|line| line.strip_prefix("- "))
            .map(str::to_string)
            .collect();

        // Apply file pattern regex filter
        if let Some(ref regex) = file_pattern_regex {
            let file_match = files_changed.iter().any(|f| regex.is_match(f));
            if !file_match {
                continue;
            }
        }

        // Extract diff snippet (first ~500 chars of diff)
        let diff_snippet = if let Some((_, diff_content)) = result.content.split_once("\nDiff:\n") {
            if diff_content.len() > 500 {
                // Byte cap: slicing a str at a non-boundary offset panics.
                let end = crate::git::floor_char_boundary(diff_content, 500);
                format!("{}...", &diff_content[..end])
            } else {
                diff_content.to_string()
            }
        } else {
            String::new()
        };

        filtered_results.push(GitSearchResult {
            commit_hash,
            subject,
            commit_message,
            author,
            author_email,
            author_date,
            commit_date: result.indexed_at,
            score: result.score,
            vector_score: result.vector_score,
            keyword_score: result.keyword_score,
            path_at_commit: files_changed.clone(),
            files_changed,
            diff_snippet,
            origin: crate::types::RecordOrigin::History,
        });

        if filtered_results.len() >= req.limit {
            break;
        }
    }

    let duration_ms = start_time.elapsed().as_millis() as u64;
    let git_cache_guard = git_cache.read().await;
    let total_cached = git_cache_guard.commit_count(&history_cache_key);

    Ok(SearchGitHistoryResponse {
        results: filtered_results,
        commits_indexed: newly_indexed,
        total_cached_commits: total_cached,
        duration_ms,
        invalid_records,
        diagnostics,
    })
}

/// Parse a date filter string (ISO 8601 or Unix timestamp)
pub(crate) fn parse_date_filter(date_str: &str) -> Result<i64> {
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
pub(crate) fn parse_author_line(line: &str) -> (String, String) {
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
mod tests;
fn validate_commit(commit: &CommitInfo) -> Result<()> {
    git2::Oid::from_str(&commit.hash)
        .with_context(|| format!("Invalid Git object id: {}", commit.hash))?;
    if commit.author_date == 0 {
        anyhow::bail!("Commit {} has invalid author_date", commit.hash);
    }
    if commit.commit_date == 0 {
        anyhow::bail!("Commit {} has invalid commit_date", commit.hash);
    }
    Ok(())
}

fn map_commit_to_project(mut commit: CommitInfo, project_prefix: &str) -> Option<CommitInfo> {
    if project_prefix.is_empty() {
        commit.files_changed = commit
            .files_changed
            .into_iter()
            .map(|path| path.replace('\\', "/"))
            .collect();
        return Some(commit);
    }

    let prefix = format!("{}/", project_prefix.trim_matches('/'));
    commit.files_changed = commit
        .files_changed
        .into_iter()
        .filter_map(|path| {
            let normalized = path.replace('\\', "/");
            normalized.strip_prefix(&prefix).map(str::to_string)
        })
        .collect();
    (!commit.files_changed.is_empty()).then_some(commit)
}

fn content_line_value(content: &str, prefix: &str) -> Option<String> {
    content
        .lines()
        .find_map(|line| line.strip_prefix(prefix).map(str::to_string))
}
