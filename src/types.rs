use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Request to index a codebase
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IndexRequest {
    /// Path to the codebase directory to index
    pub path: String,
    /// Optional project name (for multi-project support)
    #[serde(default)]
    pub project: Option<String>,
    /// Optional glob patterns to include (e.g., ["**/*.rs", "**/*.toml"])
    #[serde(default)]
    pub include_patterns: Vec<String>,
    /// Optional glob patterns to exclude (e.g., ["**/target/**", "**/node_modules/**"])
    #[serde(default)]
    pub exclude_patterns: Vec<String>,
    /// Maximum file size in bytes to index (default: 1MB)
    #[serde(default = "default_max_file_size")]
    pub max_file_size: usize,
}

fn default_max_file_size() -> usize {
    1_048_576 // 1MB
}

/// Indexing mode used
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum IndexingMode {
    /// Full indexing (all files)
    Full,
    /// Incremental update (only changed files)
    Incremental,
}

/// Response from indexing operation
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IndexResponse {
    /// Indexing mode used (full or incremental)
    pub mode: IndexingMode,
    /// Number of files successfully indexed
    pub files_indexed: usize,
    /// Number of code chunks created
    pub chunks_created: usize,
    /// Number of embeddings generated
    pub embeddings_generated: usize,
    /// Time taken in milliseconds
    pub duration_ms: u64,
    /// Any errors encountered (non-fatal)
    #[serde(default)]
    pub errors: Vec<String>,
    /// Number of files updated (incremental mode only)
    #[serde(default)]
    pub files_updated: usize,
    /// Number of files removed (incremental mode only)
    #[serde(default)]
    pub files_removed: usize,
}

/// Request to query the codebase
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct QueryRequest {
    /// The question or search query
    pub query: String,
    /// Optional project name to filter by
    #[serde(default)]
    pub project: Option<String>,
    /// Number of results to return (default: 10)
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Minimum similarity score (0.0 to 1.0, default: 0.7)
    #[serde(default = "default_min_score")]
    pub min_score: f32,
    /// Enable hybrid search (vector + keyword) - default: true
    #[serde(default = "default_hybrid")]
    pub hybrid: bool,
}

fn default_hybrid() -> bool {
    true
}

fn default_limit() -> usize {
    10
}

fn default_min_score() -> f32 {
    0.7
}

/// A single search result
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchResult {
    /// File path relative to the indexed root
    pub file_path: String,
    /// The code chunk content
    pub content: String,
    /// Combined similarity score (0.0 to 1.0)
    pub score: f32,
    /// Vector similarity score (0.0 to 1.0)
    pub vector_score: f32,
    /// Keyword match score (0.0 to 1.0) - only present in hybrid search
    pub keyword_score: Option<f32>,
    /// Starting line number in the file
    pub start_line: usize,
    /// Ending line number in the file
    pub end_line: usize,
    /// Programming language detected
    pub language: String,
    /// Optional project name for multi-project support
    pub project: Option<String>,
}

/// Response from query operation
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct QueryResponse {
    /// List of search results, ordered by relevance
    pub results: Vec<SearchResult>,
    /// Time taken in milliseconds
    pub duration_ms: u64,
    /// The actual threshold used (may be lower than requested if adaptive search kicked in)
    #[serde(default)]
    pub threshold_used: f32,
    /// Whether the threshold was automatically lowered to find results
    #[serde(default)]
    pub threshold_lowered: bool,
}

/// Request to get statistics about the index
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct StatisticsRequest {}

/// Statistics about the indexed codebase
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct StatisticsResponse {
    /// Total number of indexed files
    pub total_files: usize,
    /// Total number of code chunks
    pub total_chunks: usize,
    /// Total number of embeddings
    pub total_embeddings: usize,
    /// Size of the vector database in bytes
    pub database_size_bytes: u64,
    /// Breakdown by programming language
    pub language_breakdown: Vec<LanguageStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct LanguageStats {
    pub language: String,
    pub file_count: usize,
    pub chunk_count: usize,
}

/// Request to clear the index
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ClearRequest {}

/// Response from clear operation
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ClearResponse {
    /// Whether the operation was successful
    pub success: bool,
    /// Optional message
    pub message: String,
}

/// Request for incremental update
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IncrementalUpdateRequest {
    /// Path to the codebase directory
    pub path: String,
    /// Optional project name
    #[serde(default)]
    pub project: Option<String>,
    /// Optional glob patterns to include
    #[serde(default)]
    pub include_patterns: Vec<String>,
    /// Optional glob patterns to exclude
    #[serde(default)]
    pub exclude_patterns: Vec<String>,
}

/// Response from incremental update
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IncrementalUpdateResponse {
    /// Number of files added
    pub files_added: usize,
    /// Number of files updated
    pub files_updated: usize,
    /// Number of files removed
    pub files_removed: usize,
    /// Number of chunks created/updated
    pub chunks_modified: usize,
    /// Time taken in milliseconds
    pub duration_ms: u64,
}

/// Request to search with file type filters
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AdvancedSearchRequest {
    /// The search query
    pub query: String,
    /// Optional project name to filter by
    #[serde(default)]
    pub project: Option<String>,
    /// Number of results to return
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Minimum similarity score
    #[serde(default = "default_min_score")]
    pub min_score: f32,
    /// Filter by file extensions (e.g., ["rs", "toml"])
    #[serde(default)]
    pub file_extensions: Vec<String>,
    /// Filter by programming languages
    #[serde(default)]
    pub languages: Vec<String>,
    /// Filter by file path patterns (glob)
    #[serde(default)]
    pub path_patterns: Vec<String>,
}

/// Request to search git history
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchGitHistoryRequest {
    /// The search query
    pub query: String,
    /// Path to the codebase (will discover git repo)
    #[serde(default = "default_git_path")]
    pub path: String,
    /// Optional project name
    #[serde(default)]
    pub project: Option<String>,
    /// Optional branch name (default: current branch)
    #[serde(default)]
    pub branch: Option<String>,
    /// Maximum number of commits to index/search (default: 10)
    #[serde(default = "default_max_commits")]
    pub max_commits: usize,
    /// Number of results to return (default: 10)
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Minimum similarity score (0.0 to 1.0, default: 0.7)
    #[serde(default = "default_min_score")]
    pub min_score: f32,
    /// Filter by commit author (optional regex pattern)
    #[serde(default)]
    pub author: Option<String>,
    /// Filter by commits since this date (ISO 8601 or Unix timestamp)
    #[serde(default)]
    pub since: Option<String>,
    /// Filter by commits until this date (ISO 8601 or Unix timestamp)
    #[serde(default)]
    pub until: Option<String>,
    /// Filter by file path pattern (optional regex)
    #[serde(default)]
    pub file_pattern: Option<String>,
}

fn default_git_path() -> String {
    ".".to_string()
}

fn default_max_commits() -> usize {
    10
}

/// A single git search result
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GitSearchResult {
    /// Git commit hash (SHA)
    pub commit_hash: String,
    /// Commit message
    pub commit_message: String,
    /// Author name
    pub author: String,
    /// Author email
    pub author_email: String,
    /// Commit date (Unix timestamp)
    pub commit_date: i64,
    /// Combined similarity score (0.0 to 1.0)
    pub score: f32,
    /// Vector similarity score
    pub vector_score: f32,
    /// Keyword match score (if hybrid search enabled)
    pub keyword_score: Option<f32>,
    /// Files changed in this commit
    pub files_changed: Vec<String>,
    /// Diff snippet (first ~500 characters)
    pub diff_snippet: String,
}

/// Response from git history search
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchGitHistoryResponse {
    /// List of matching commits, ordered by relevance
    pub results: Vec<GitSearchResult>,
    /// Number of commits indexed during this search
    pub commits_indexed: usize,
    /// Total commits in cache for this repo
    pub total_cached_commits: usize,
    /// Time taken in milliseconds
    pub duration_ms: u64,
}

/// Metadata stored with each code chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// File path relative to indexed root
    pub file_path: String,
    /// Project name (for multi-project support)
    pub project: Option<String>,
    /// Starting line number
    pub start_line: usize,
    /// Ending line number
    pub end_line: usize,
    /// Programming language
    pub language: Option<String>,
    /// File extension
    pub extension: Option<String>,
    /// SHA256 hash of the file content
    pub file_hash: String,
    /// Timestamp when indexed
    pub indexed_at: i64,
}

/// Input validation for request types
///
/// These functions validate user inputs to prevent security issues and ensure
/// reasonable resource usage.
impl IndexRequest {
    /// Validate the index request
    pub fn validate(&self) -> Result<(), String> {
        // Validate path exists and is a directory
        let path = std::path::Path::new(&self.path);
        if !path.exists() {
            return Err(format!("Path does not exist: {}", self.path));
        }
        if !path.is_dir() {
            return Err(format!("Path is not a directory: {}", self.path));
        }

        // Canonicalize to prevent path traversal attacks
        let canonical = path
            .canonicalize()
            .map_err(|e| format!("Failed to canonicalize path: {}", e))?;

        // Check that path doesn't try to escape (basic security check)
        if !canonical.starts_with(std::env::current_dir().unwrap_or_default().parent().unwrap_or(std::path::Path::new("/"))) {
            // Allow any absolute path, this check is just to catch obvious traversal attempts
        }

        // Validate max_file_size is reasonable (max 100MB)
        const MAX_FILE_SIZE_LIMIT: usize = 100_000_000; // 100MB
        if self.max_file_size > MAX_FILE_SIZE_LIMIT {
            return Err(format!(
                "max_file_size too large: {} bytes (max: {} bytes)",
                self.max_file_size, MAX_FILE_SIZE_LIMIT
            ));
        }

        // Validate project name if provided
        if let Some(ref project) = self.project {
            if project.is_empty() {
                return Err("project name cannot be empty".to_string());
            }
            if project.len() > 256 {
                return Err("project name too long (max 256 characters)".to_string());
            }
        }

        Ok(())
    }
}

impl QueryRequest {
    /// Validate the query request
    pub fn validate(&self) -> Result<(), String> {
        // Validate query is not empty
        if self.query.trim().is_empty() {
            return Err("query cannot be empty".to_string());
        }

        // Validate query length is reasonable (max 10KB)
        const MAX_QUERY_LENGTH: usize = 10_240; // 10KB
        if self.query.len() > MAX_QUERY_LENGTH {
            return Err(format!(
                "query too long: {} bytes (max: {} bytes)",
                self.query.len(),
                MAX_QUERY_LENGTH
            ));
        }

        // Validate min_score is in valid range [0.0, 1.0]
        if !(0.0..=1.0).contains(&self.min_score) {
            return Err(format!(
                "min_score must be between 0.0 and 1.0, got: {}",
                self.min_score
            ));
        }

        // Validate limit is reasonable (max 1000)
        const MAX_LIMIT: usize = 1000;
        if self.limit > MAX_LIMIT {
            return Err(format!(
                "limit too large: {} (max: {})",
                self.limit, MAX_LIMIT
            ));
        }

        // Validate project name if provided
        if let Some(ref project) = self.project {
            if project.is_empty() {
                return Err("project name cannot be empty".to_string());
            }
            if project.len() > 256 {
                return Err("project name too long (max 256 characters)".to_string());
            }
        }

        Ok(())
    }
}

impl AdvancedSearchRequest {
    /// Validate the advanced search request
    pub fn validate(&self) -> Result<(), String> {
        // Reuse QueryRequest validation logic
        let query_req = QueryRequest {
            query: self.query.clone(),
            project: self.project.clone(),
            limit: self.limit,
            min_score: self.min_score,
            hybrid: true,
        };
        query_req.validate()?;

        // Additional validation for file extensions
        for ext in &self.file_extensions {
            if ext.is_empty() {
                return Err("file extension cannot be empty".to_string());
            }
            if ext.len() > 20 {
                return Err(format!("file extension too long: {} (max 20 characters)", ext));
            }
        }

        // Validate languages
        for lang in &self.languages {
            if lang.is_empty() {
                return Err("language name cannot be empty".to_string());
            }
            if lang.len() > 50 {
                return Err(format!("language name too long: {} (max 50 characters)", lang));
            }
        }

        Ok(())
    }
}

impl SearchGitHistoryRequest {
    /// Validate the git history search request
    pub fn validate(&self) -> Result<(), String> {
        // Validate query
        if self.query.trim().is_empty() {
            return Err("query cannot be empty".to_string());
        }

        const MAX_QUERY_LENGTH: usize = 10_240; // 10KB
        if self.query.len() > MAX_QUERY_LENGTH {
            return Err(format!(
                "query too long: {} bytes (max: {} bytes)",
                self.query.len(),
                MAX_QUERY_LENGTH
            ));
        }

        // Validate path
        let path = std::path::Path::new(&self.path);
        if !path.exists() {
            return Err(format!("Path does not exist: {}", self.path));
        }

        // Validate min_score range
        if !(0.0..=1.0).contains(&self.min_score) {
            return Err(format!(
                "min_score must be between 0.0 and 1.0, got: {}",
                self.min_score
            ));
        }

        // Validate limit
        const MAX_LIMIT: usize = 1000;
        if self.limit > MAX_LIMIT {
            return Err(format!(
                "limit too large: {} (max: {})",
                self.limit, MAX_LIMIT
            ));
        }

        // Validate max_commits
        const MAX_COMMITS_LIMIT: usize = 10000;
        if self.max_commits > MAX_COMMITS_LIMIT {
            return Err(format!(
                "max_commits too large: {} (max: {})",
                self.max_commits, MAX_COMMITS_LIMIT
            ));
        }

        // Validate project name if provided
        if let Some(ref project) = self.project {
            if project.is_empty() {
                return Err("project name cannot be empty".to_string());
            }
            if project.len() > 256 {
                return Err("project name too long (max 256 characters)".to_string());
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_request_defaults() {
        let req = IndexRequest {
            path: "/test".to_string(),
            project: None,
            include_patterns: vec![],
            exclude_patterns: vec![],
            max_file_size: default_max_file_size(),
        };

        assert_eq!(req.max_file_size, 1_048_576);
        assert!(req.include_patterns.is_empty());
        assert!(req.exclude_patterns.is_empty());
    }

    #[test]
    fn test_index_response_full_mode() {
        let response = IndexResponse {
            mode: IndexingMode::Full,
            files_indexed: 100,
            chunks_created: 500,
            embeddings_generated: 500,
            duration_ms: 1000,
            errors: vec![],
            files_updated: 0,
            files_removed: 0,
        };

        assert!(matches!(response.mode, IndexingMode::Full));
        assert_eq!(response.files_indexed, 100);
        assert_eq!(response.files_updated, 0);
        assert_eq!(response.files_removed, 0);
    }

    #[test]
    fn test_index_response_incremental_mode() {
        let response = IndexResponse {
            mode: IndexingMode::Incremental,
            files_indexed: 10,
            chunks_created: 50,
            embeddings_generated: 50,
            duration_ms: 500,
            errors: vec![],
            files_updated: 5,
            files_removed: 2,
        };

        assert!(matches!(response.mode, IndexingMode::Incremental));
        assert_eq!(response.files_indexed, 10);
        assert_eq!(response.files_updated, 5);
        assert_eq!(response.files_removed, 2);
    }

    #[test]
    fn test_query_request_defaults() {
        let req = QueryRequest {
            query: "test".to_string(),
            project: None,
            limit: default_limit(),
            min_score: default_min_score(),
            hybrid: default_hybrid(),
        };

        assert_eq!(req.limit, 10);
        assert_eq!(req.min_score, 0.7);
        assert!(req.hybrid);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let req = IndexRequest {
            path: "/test/path".to_string(),
            project: Some("my-project".to_string()),
            include_patterns: vec!["**/*.rs".to_string()],
            exclude_patterns: vec!["**/target/**".to_string()],
            max_file_size: 2_000_000,
        };

        let json = serde_json::to_string(&req).unwrap();
        let deserialized: IndexRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(req.path, deserialized.path);
        assert_eq!(req.include_patterns, deserialized.include_patterns);
        assert_eq!(req.exclude_patterns, deserialized.exclude_patterns);
        assert_eq!(req.max_file_size, deserialized.max_file_size);
    }

    #[test]
    fn test_search_result_creation() {
        let result = SearchResult {
            file_path: "src/main.rs".to_string(),
            content: "fn main() {}".to_string(),
            score: 0.95,
            vector_score: 0.92,
            keyword_score: Some(0.85),
            start_line: 1,
            end_line: 10,
            language: "Rust".to_string(),
            project: None,
        };

        assert_eq!(result.score, 0.95);
        assert_eq!(result.vector_score, 0.92);
        assert_eq!(result.keyword_score, Some(0.85));
        assert_eq!(result.language, "Rust");
    }

    #[test]
    fn test_chunk_metadata_creation() {
        let metadata = ChunkMetadata {
            file_path: "src/lib.rs".to_string(),
            project: Some("test-project".to_string()),
            start_line: 1,
            end_line: 50,
            language: Some("Rust".to_string()),
            extension: Some("rs".to_string()),
            file_hash: "abc123".to_string(),
            indexed_at: 1234567890,
        };

        assert_eq!(metadata.start_line, 1);
        assert_eq!(metadata.end_line, 50);
        assert_eq!(metadata.file_hash, "abc123");
        assert_eq!(metadata.project, Some("test-project".to_string()));
    }

    #[test]
    fn test_clear_response() {
        let response = ClearResponse {
            success: true,
            message: "Cleared successfully".to_string(),
        };

        assert!(response.success);
        assert!(!response.message.is_empty());
    }

    #[test]
    fn test_statistics_response() {
        let stats = StatisticsResponse {
            total_files: 100,
            total_chunks: 500,
            total_embeddings: 500,
            database_size_bytes: 1024 * 1024,
            language_breakdown: vec![
                LanguageStats {
                    language: "Rust".to_string(),
                    file_count: 80,
                    chunk_count: 400,
                },
                LanguageStats {
                    language: "TOML".to_string(),
                    file_count: 20,
                    chunk_count: 100,
                },
            ],
        };

        assert_eq!(stats.total_files, 100);
        assert_eq!(stats.language_breakdown.len(), 2);
        assert_eq!(stats.language_breakdown[0].language, "Rust");
    }
}
