use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod file_ops;
pub use file_ops::{EditFileRequest, EditFileResponse, ReadFileRequest, ReadFileResponse};
mod patching;
pub use patching::{
    ApplyPatchRequest, ApplyPatchResponse, FilePatch, PatchConflict, PatchedFileResult,
};
mod removal;
pub use removal::{
    RemovalAnalysisScope, RemovalEvidence, RemovalVerdict, ValidateRemovalRequest,
    ValidateRemovalResponse,
};
mod find_unused;
pub use find_unused::{
    AnalysisCompleteness, FindUnusedRequest, FindUnusedResponse, SymbolRejections, UnusedCandidate,
    UnusedStatus, UnverifiableImport,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum RecordOrigin {
    #[default]
    Current,
    History,
}

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
    /// Optional path to filter by specific indexed codebase
    #[serde(default)]
    pub path: Option<String>,
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
    /// Absolute path to the indexed root directory
    #[serde(default, skip_serializing)]
    pub root_path: Option<String>,
    /// The code chunk content
    pub content: String,
    /// Original chunk bounds before snippet extraction.
    #[serde(default)]
    pub full_start_line: usize,
    #[serde(default)]
    pub full_end_line: usize,
    /// True when the retrieval response contains a snippet rather than the full chunk.
    #[serde(default)]
    pub content_truncated: bool,
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
    /// Retrieval plane containing this record.
    #[serde(default)]
    pub origin: RecordOrigin,
    /// Source record id (for history this is the validated commit object id).
    #[serde(default)]
    pub source_id: Option<String>,
    /// Source timestamp recorded by the index.
    #[serde(default)]
    pub indexed_at: i64,
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
    /// Matches observed before the response result budget was applied.
    #[serde(default)]
    pub total_matches: usize,
    #[serde(default)]
    pub returned_matches: usize,
    #[serde(default)]
    pub results_truncated: bool,
    /// Reserved for cursor-based pagination. None means this backend cannot
    /// currently continue the bounded candidate set.
    #[serde(default)]
    pub next_cursor: Option<String>,
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
    /// Total symbol definitions in the relations store (populated during indexing)
    #[serde(default)]
    pub total_definitions: usize,
    /// Total symbol references in the relations store (populated on demand,
    /// e.g. by tools that verify usage; 0 unless references were stored)
    #[serde(default)]
    pub total_references: usize,
    /// Persisted code references, excluding documentation/comments/strings.
    #[serde(default)]
    pub code_reference_count: usize,
    /// Number of files with at least one stored definition
    #[serde(default)]
    pub files_with_definitions: usize,
    #[serde(default)]
    pub index_schema_version: u32,
    #[serde(default)]
    pub invalid_history_records: usize,
    #[serde(default)]
    pub index_diagnostics: Vec<String>,
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
    /// Optional path to filter by specific indexed codebase
    #[serde(default)]
    pub path: Option<String>,
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
    /// First line of the commit message.
    pub subject: String,
    /// Commit message
    pub commit_message: String,
    /// Author name
    pub author: String,
    /// Author email
    pub author_email: String,
    /// Author timestamp (Unix timestamp).
    pub author_date: i64,
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
    /// Canonical project-relative paths as they appeared in this commit.
    pub path_at_commit: Vec<String>,
    /// Diff snippet (first ~500 characters)
    pub diff_snippet: String,
    pub origin: RecordOrigin,
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
    /// Invalid Git records excluded from the authoritative history index.
    #[serde(default)]
    pub invalid_records: usize,
    #[serde(default)]
    pub diagnostics: Vec<String>,
}

// ============================================================================
// Code Relations Types (find_definition, find_references, get_call_graph)
// ============================================================================

/// Request to find the definition of a symbol at a given location
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FindDefinitionRequest {
    /// File path (relative or absolute)
    pub file_path: String,
    /// Line number (1-based)
    pub line: usize,
    /// Column number (0-based)
    pub column: usize,
    /// Optional project name to filter by
    #[serde(default)]
    pub project: Option<String>,
}

impl FindDefinitionRequest {
    /// Validate the find definition request
    pub fn validate(&self) -> Result<(), String> {
        if self.file_path.is_empty() {
            return Err("file_path cannot be empty".to_string());
        }
        if self.line == 0 {
            return Err("line must be 1-based (cannot be 0)".to_string());
        }
        Ok(())
    }
}

/// Response from find_definition
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FindDefinitionResponse {
    /// The found definition, if any
    pub definition: Option<crate::relations::DefinitionResult>,
    /// Resolution state of the symbol occurrence at the requested position.
    pub resolution_status: crate::relations::ResolutionStatus,
    pub evidence_kind: crate::relations::EvidenceKind,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub candidates: Vec<crate::relations::ReferenceCandidate>,
    /// Precision level of the result
    pub precision: String,
    /// Time taken in milliseconds
    pub duration_ms: u64,
}

/// Request to find all references to a symbol
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FindReferencesRequest {
    /// File path (relative or absolute)
    pub file_path: String,
    /// Line number (1-based)
    pub line: usize,
    /// Column number (0-based)
    pub column: usize,
    /// Maximum number of references to return
    #[serde(default = "default_references_limit")]
    pub limit: usize,
    /// Optional project name to filter by
    #[serde(default)]
    pub project: Option<String>,
    /// Include the definition itself in results
    #[serde(default = "default_include_definition")]
    pub include_definition: bool,
    /// Filter references by parser language (case-insensitive).
    #[serde(default)]
    pub language: Option<String>,
    /// Filter by canonical project-relative reference path.
    #[serde(default)]
    pub path_filter: Option<String>,
    /// Restrict reference kinds. Empty uses the default code-only set.
    #[serde(default)]
    pub reference_kinds: Vec<crate::relations::ReferenceKind>,
    /// Restrict resolution states. Empty includes all states.
    #[serde(default)]
    pub resolution_statuses: Vec<crate::relations::ResolutionStatus>,
    /// Restrict evidence kinds. Empty includes all evidence kinds.
    #[serde(default)]
    pub evidence_kinds: Vec<crate::relations::EvidenceKind>,
    /// Restrict occurrences to active/unknown scope in these build configurations.
    #[serde(default)]
    pub configurations: Vec<String>,
    /// Include documentation, comment, and string matches when no kind filter is supplied.
    #[serde(default)]
    pub include_non_code: bool,
    /// Zero-based result offset for deterministic pagination.
    #[serde(default)]
    pub cursor: usize,
}

fn default_references_limit() -> usize {
    100
}

fn default_include_definition() -> bool {
    true
}

impl FindReferencesRequest {
    /// Validate the find references request
    pub fn validate(&self) -> Result<(), String> {
        if self.file_path.is_empty() {
            return Err("file_path cannot be empty".to_string());
        }
        if self.line == 0 {
            return Err("line must be 1-based (cannot be 0)".to_string());
        }
        const MAX_LIMIT: usize = 10000;
        if self.limit > MAX_LIMIT {
            return Err(format!(
                "limit too large: {} (max: {})",
                self.limit, MAX_LIMIT
            ));
        }
        if let Some(path) = &self.path_filter
            && path.trim().is_empty()
        {
            return Err("path_filter cannot be empty".to_string());
        }
        if self
            .configurations
            .iter()
            .any(|configuration| configuration.trim().is_empty())
        {
            return Err("configurations cannot contain an empty config_id".to_string());
        }
        Ok(())
    }
}

/// Response from find_references
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FindReferencesResponse {
    /// The symbol being referenced
    pub symbol_name: Option<String>,
    pub target_resolution_status: crate::relations::ResolutionStatus,
    pub target_evidence_kind: crate::relations::EvidenceKind,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub target_candidates: Vec<crate::relations::ReferenceCandidate>,
    /// List of found references
    pub references: Vec<crate::relations::ReferenceResult>,
    /// Total count (may be higher than returned if limit applied)
    pub total_count: usize,
    /// Total matches after all explicit/default filters, before pagination.
    pub total_matches: usize,
    /// Number of rows returned in this page.
    pub returned_matches: usize,
    pub results_truncated: bool,
    pub next_cursor: Option<usize>,
    /// Counts computed from the same filtered reference set as this response.
    pub statistics: ReferenceStatistics,
    /// Precision level of the results
    pub precision: String,
    /// Time taken in milliseconds
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ReferenceStatistics {
    /// Exact count after the active filters; never the current page length.
    pub code_reference_count: usize,
    pub resolved_count: usize,
    pub ambiguous_count: usize,
    pub unresolved_count: usize,
}

/// Request to get call graph for a function
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GetCallGraphRequest {
    /// File path (relative or absolute)
    pub file_path: String,
    /// Line number (1-based)
    pub line: usize,
    /// Column number (0-based)
    pub column: usize,
    /// Maximum depth to traverse (default: 2)
    #[serde(default = "default_call_graph_depth")]
    pub depth: usize,
    /// Optional project name to filter by
    #[serde(default)]
    pub project: Option<String>,
    /// Include callers (functions that call this function)
    #[serde(default = "default_true")]
    pub include_callers: bool,
    /// Include callees (functions this function calls)
    #[serde(default = "default_true")]
    pub include_callees: bool,
    /// Maximum unique nodes in the returned graph, including the root.
    #[serde(default = "default_graph_max_nodes")]
    pub max_nodes: usize,
    /// Maximum source occurrences in the returned graph.
    #[serde(default = "default_graph_max_edges")]
    pub max_edges: usize,
    /// Edge/reference kinds to traverse. Empty defaults to calls and constructor calls.
    #[serde(default)]
    pub edge_kinds: Vec<crate::relations::ReferenceKind>,
    /// Resolution states to include. Empty defaults to resolved edges only.
    #[serde(default)]
    pub resolution_statuses: Vec<crate::relations::ResolutionStatus>,
    /// Exact, case-insensitive parser-language filters.
    #[serde(default)]
    pub language_filters: Vec<String>,
    /// Canonical project-relative path substring filters.
    #[serde(default)]
    pub path_filters: Vec<String>,
    /// Restrict graph observations to active/unknown scope in these configurations.
    #[serde(default)]
    pub configurations: Vec<String>,
}

fn default_call_graph_depth() -> usize {
    2
}

fn default_true() -> bool {
    true
}

fn default_graph_max_nodes() -> usize {
    200
}

fn default_graph_max_edges() -> usize {
    500
}

impl GetCallGraphRequest {
    /// Validate the get call graph request
    pub fn validate(&self) -> Result<(), String> {
        if self.file_path.is_empty() {
            return Err("file_path cannot be empty".to_string());
        }
        if self.line == 0 {
            return Err("line must be 1-based (cannot be 0)".to_string());
        }
        const MAX_DEPTH: usize = 10;
        if self.depth > MAX_DEPTH {
            return Err(format!(
                "depth too large: {} (max: {})",
                self.depth, MAX_DEPTH
            ));
        }
        if self.max_nodes == 0 {
            return Err("max_nodes must be at least 1".to_string());
        }
        if self.max_edges == 0 {
            return Err("max_edges must be at least 1".to_string());
        }
        const HARD_MAX_NODES: usize = 5_000;
        const HARD_MAX_EDGES: usize = 20_000;
        if self.max_nodes > HARD_MAX_NODES {
            return Err(format!(
                "max_nodes too large: {} (max: {})",
                self.max_nodes, HARD_MAX_NODES
            ));
        }
        if self.max_edges > HARD_MAX_EDGES {
            return Err(format!(
                "max_edges too large: {} (max: {})",
                self.max_edges, HARD_MAX_EDGES
            ));
        }
        if self.path_filters.iter().any(|path| path.trim().is_empty()) {
            return Err("path_filters cannot contain an empty value".to_string());
        }
        if self
            .language_filters
            .iter()
            .any(|language| language.trim().is_empty())
        {
            return Err("language_filters cannot contain an empty value".to_string());
        }
        if self
            .configurations
            .iter()
            .any(|configuration| configuration.trim().is_empty())
        {
            return Err("configurations cannot contain an empty config_id".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GraphTotals {
    /// Discovered total. This is a lower bound when `exact` is false.
    pub nodes: usize,
    /// Discovered total. This is a lower bound when `exact` is false.
    pub edges: usize,
    /// True only when traversal completed within the requested depth and filters.
    pub exact: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GraphContinuation {
    /// Stable symbols at which a capped traversal can continue.
    pub pending_symbol_ids: Vec<String>,
    pub next_depth: usize,
    pub remaining_depth: usize,
    pub reason: String,
}

/// Response from get_call_graph
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GetCallGraphResponse {
    /// The root symbol (function/method at the requested location)
    pub root_symbol: Option<crate::relations::SymbolInfo>,
    /// Unique logical symbols, including the root at distance zero.
    pub nodes: Vec<crate::relations::CallGraphNode>,
    /// Deduplicated source occurrences between nodes, plus explicitly requested
    /// ambiguous/unresolved outgoing observations with no authoritative target.
    pub edges: Vec<crate::relations::GraphEdge>,
    pub requested_depth: usize,
    pub graph_truncated: bool,
    pub returned_nodes: usize,
    pub returned_edges: usize,
    pub estimated_or_known_total: GraphTotals,
    pub continuation: Option<GraphContinuation>,
    /// Filters actually applied after defaults were expanded.
    pub applied_edge_kinds: Vec<crate::relations::ReferenceKind>,
    pub applied_resolution_statuses: Vec<crate::relations::ResolutionStatus>,
    /// Successfully published index generation used for this graph.
    #[serde(default)]
    pub index_generation: u64,
    /// True when returned from the generation-scoped in-memory cache.
    #[serde(default)]
    pub cache_hit: bool,
    /// Precision level of the results
    pub precision: String,
    /// Time taken in milliseconds
    pub duration_ms: u64,
}

/// Request to list every symbol defined in one file
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ListSymbolsRequest {
    /// File path (relative or absolute)
    pub file_path: String,
    /// Optional project name to filter by
    #[serde(default)]
    pub project: Option<String>,
    /// Restrict to these symbol kinds (e.g. ["function", "method"]); empty means all
    #[serde(default)]
    pub kinds: Vec<String>,
}

impl ListSymbolsRequest {
    /// Validate the list symbols request
    pub fn validate(&self) -> Result<(), String> {
        if self.file_path.is_empty() {
            return Err("file_path cannot be empty".to_string());
        }
        Ok(())
    }
}

/// Response from list_symbols
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ListSymbolsResponse {
    /// File the symbols were extracted from, relative to the indexed root
    pub file_path: String,
    /// Every definition found, ordered by start_line. Never includes file content.
    pub symbols: Vec<crate::relations::SymbolInfo>,
    /// Number of symbols returned
    pub total_count: usize,
    /// Precision level of the extractor for this language
    pub precision: String,
    /// Definition nodes that were recognised but could not be named, and are therefore
    /// MISSING from `symbols`. Empty is the normal case; anything here means this
    /// listing is incomplete and must not be treated as an inventory.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skipped: Vec<crate::relations::SkippedDefinition>,
    /// Time taken in milliseconds
    pub duration_ms: u64,
}

/// Metadata stored with each code chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// File path relative to indexed root
    pub file_path: String,
    /// Absolute path to the indexed root directory
    #[serde(default)]
    pub root_path: Option<String>,
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
    /// Current-tree and history records are never searched together implicitly.
    #[serde(default)]
    pub origin: RecordOrigin,
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
        if !canonical.starts_with(
            std::env::current_dir()
                .unwrap_or_default()
                .parent()
                .unwrap_or(std::path::Path::new("/")),
        ) {
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
            path: None,
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
                return Err(format!(
                    "file extension too long: {} (max 20 characters)",
                    ext
                ));
            }
        }

        // Validate languages
        for lang in &self.languages {
            if lang.is_empty() {
                return Err("language name cannot be empty".to_string());
            }
            if lang.len() > 50 {
                return Err(format!(
                    "language name too long: {} (max 50 characters)",
                    lang
                ));
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
mod tests;
