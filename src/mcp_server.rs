use crate::client::RagClient;
use crate::types::*;

use anyhow::{Context, Result};
use rmcp::{
    ErrorData as McpError, Peer, RoleServer, ServerHandler, ServiceExt,
    handler::server::{router::prompt::PromptRouter, tool::ToolRouter, wrapper::Parameters},
    model::*,
    prompt, prompt_handler, prompt_router,
    service::RequestContext,
    tool, tool_handler, tool_router,
};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Guard that cancels a CancellationToken when dropped.
/// This ensures that if the async handler's future is dropped (e.g., due to client disconnect),
/// the cancellation token is triggered, allowing cooperative cancellation of long-running operations.
struct CancelOnDropGuard {
    token: CancellationToken,
}

impl CancelOnDropGuard {
    fn new(token: CancellationToken) -> Self {
        Self { token }
    }
}

impl Drop for CancelOnDropGuard {
    fn drop(&mut self) {
        if !self.token.is_cancelled() {
            tracing::info!("Tool handler dropped, triggering cancellation");
            self.token.cancel();
        }
    }
}

#[derive(Clone)]
pub struct RagMcpServer {
    client: Arc<RagClient>,
    tool_router: ToolRouter<Self>,
    prompt_router: PromptRouter<Self>,
}

impl RagMcpServer {
    /// Create a new RAG MCP server with default configuration
    pub async fn new() -> Result<Self> {
        let client = RagClient::new().await?;
        Self::with_client(Arc::new(client))
    }

    /// Create a new RAG MCP server with an existing client
    pub fn with_client(client: Arc<RagClient>) -> Result<Self> {
        Ok(Self {
            client,
            tool_router: Self::tool_router(),
            prompt_router: Self::prompt_router(),
        })
    }

    /// Get the underlying client
    pub fn client(&self) -> &RagClient {
        &self.client
    }

    /// Create a new RAG MCP server with custom configuration
    pub async fn with_config(config: crate::config::Config) -> Result<Self> {
        let client = RagClient::with_config(config).await?;
        Self::with_client(Arc::new(client))
    }

    /// Normalize a path to a canonical absolute form for consistent cache lookups
    pub fn normalize_path(path: &str) -> Result<String> {
        RagClient::normalize_path(path)
    }

    /// Index a codebase directory (convenience method for testing)
    #[allow(clippy::too_many_arguments)]
    pub async fn do_index(
        &self,
        path: String,
        project: Option<String>,
        include_patterns: Vec<String>,
        exclude_patterns: Vec<String>,
        max_file_size: usize,
        peer: Option<Peer<RoleServer>>,
        progress_token: Option<ProgressToken>,
        cancel_token: Option<CancellationToken>,
    ) -> Result<IndexResponse> {
        let cancel_token = cancel_token.unwrap_or_default();
        crate::client::indexing::do_index_smart(
            &self.client,
            path,
            project,
            include_patterns,
            exclude_patterns,
            max_file_size,
            peer,
            progress_token,
            cancel_token,
        )
        .await
    }
}

#[tool_router(router = tool_router)]
impl RagMcpServer {
    #[tool(
        description = "Index a codebase directory, creating embeddings for semantic search. Automatically performs full indexing for new codebases or incremental updates for previously indexed codebases."
    )]
    async fn index_codebase(
        &self,
        meta: Meta,
        peer: Peer<RoleServer>,
        Parameters(req): Parameters<IndexRequest>,
    ) -> Result<String, String> {
        // Validate request inputs
        req.validate()?;

        // Get progress token if provided
        let progress_token = meta.get_progress_token();

        // Create a cancellation token for this indexing operation
        // When this handler's future is dropped (e.g., client disconnects),
        // the CancellationToken will be dropped and signal cancellation
        let cancel_token = CancellationToken::new();
        let cancel_token_for_index = cancel_token.clone();

        // Use a guard to cancel on drop
        let _cancel_guard = CancelOnDropGuard::new(cancel_token);

        let response = crate::client::indexing::do_index_smart(
            &self.client,
            req.path,
            req.project,
            req.include_patterns,
            req.exclude_patterns,
            req.max_file_size,
            Some(peer),
            progress_token,
            cancel_token_for_index,
        )
        .await
        .map_err(|e| format!("{:#}", e))?; // Use alternate display to show full error chain

        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(description = "Query the indexed codebase using semantic search")]
    async fn query_codebase(
        &self,
        Parameters(req): Parameters<QueryRequest>,
    ) -> Result<String, String> {
        // Validate request inputs
        req.validate()?;

        let response = self
            .client
            .query_codebase(req)
            .await
            .map_err(|e| format!("{:#}", e))?;

        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(description = "Get statistics about the indexed codebase")]
    async fn get_statistics(
        &self,
        Parameters(_req): Parameters<StatisticsRequest>,
    ) -> Result<String, String> {
        let response = self
            .client
            .get_statistics()
            .await
            .map_err(|e| format!("{:#}", e))?;

        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(description = "Clear all indexed data from the vector database")]
    async fn clear_index(
        &self,
        Parameters(_req): Parameters<ClearRequest>,
    ) -> Result<String, String> {
        let response = self
            .client
            .clear_index()
            .await
            .map_err(|e| format!("{:#}", e))?;

        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(description = "Advanced search with filters for file type, language, and path patterns")]
    async fn search_by_filters(
        &self,
        Parameters(req): Parameters<AdvancedSearchRequest>,
    ) -> Result<String, String> {
        // Validate request inputs
        req.validate()?;

        let response = self
            .client
            .search_with_filters(req)
            .await
            .map_err(|e| format!("{:#}", e))?;

        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(description = "Search git commit history using semantic search with on-demand indexing")]
    async fn search_git_history(
        &self,
        Parameters(req): Parameters<SearchGitHistoryRequest>,
    ) -> Result<String, String> {
        // Validate request inputs
        req.validate()?;

        let response = self
            .client
            .search_git_history(req)
            .await
            .map_err(|e| format!("{:#}", e))?;

        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(
        description = "Find the definition of a symbol occurrence at a file location. Returns explicit resolution status, evidence kind, and candidates; ambiguous or unresolved occurrences do not fall back to the enclosing function."
    )]
    async fn find_definition(
        &self,
        Parameters(req): Parameters<FindDefinitionRequest>,
    ) -> Result<String, String> {
        // Validate request inputs
        req.validate()?;

        let response = self
            .client
            .find_definition(req)
            .await
            .map_err(|e| format!("{:#}", e))?;

        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(
        description = "Find persisted, classified references to the logical symbol at a file location. Defaults to code-only matches (documentation/comments/strings excluded); supports language, canonical path, reference-kind, resolution-status, and evidence-kind filters. Unverified textual candidates remain ambiguous or unresolved and expose candidates instead of a false target. Totals are computed before pagination."
    )]
    async fn find_references(
        &self,
        Parameters(req): Parameters<FindReferencesRequest>,
    ) -> Result<String, String> {
        // Validate request inputs
        req.validate()?;

        let response = self
            .client
            .find_references(req)
            .await
            .map_err(|e| format!("{:#}", e))?;

        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(
        description = "Get a bounded dependency graph for a function at a file location. Depth is exact (0=root, 1=direct neighbors), nodes are unique stable symbols, cycles are safe, and edges include source provenance plus explicit resolution/evidence. Defaults to resolved call and constructor-call edges; supports direction, kind, status, language, path, node, and edge filters/budgets."
    )]
    async fn get_call_graph(
        &self,
        Parameters(req): Parameters<GetCallGraphRequest>,
    ) -> Result<String, String> {
        // Validate request inputs
        req.validate()?;

        let response = self
            .client
            .get_call_graph(req)
            .await
            .map_err(|e| format!("{:#}", e))?;

        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(
        description = "List every symbol defined in one file (name, kind, line span, signature) with no file content. Use this to enumerate a file: query_codebase and search_by_filters are relevance-ranked with a limit and cannot enumerate, while find_definition and find_references need a position you already have."
    )]
    async fn list_symbols(
        &self,
        Parameters(req): Parameters<ListSymbolsRequest>,
    ) -> Result<String, String> {
        // Validate request inputs
        req.validate()?;

        let response = self
            .client
            .list_symbols(req)
            .await
            .map_err(|e| format!("{:#}", e))?;

        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(
        description = "Read a bounded slice of a file inside an indexed project root. Defaults to start_line=1 and line_count=30. Explicit larger reads are allowed up to the server hard limit. The legacy inclusive end_line field remains supported but cannot be combined with line_count. range_clamped reports EOF clamping; content_truncated reports a server cap. Returns a SHA256 file_hash for guarded edits."
    )]
    async fn read_file(
        &self,
        Parameters(req): Parameters<ReadFileRequest>,
    ) -> Result<String, String> {
        req.validate()?;

        let response = self
            .client
            .read_file(req)
            .await
            .map_err(|e| format!("{:#}", e))?;

        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(
        description = "Replace a line range (or the whole file) in a file inside an already-indexed project root, then automatically reindex that root so search results stay current. Pass expected_hash (from read_file) to guard against overwriting a change you haven't seen - a mismatch returns status: \"hash_conflict\" instead of applying the edit. Omit start_line/end_line to replace or create the whole file. Set start_line to end_line + 1 to insert content without deleting anything."
    )]
    async fn edit_file(
        &self,
        Parameters(req): Parameters<EditFileRequest>,
    ) -> Result<String, String> {
        req.validate()?;

        let response = self
            .client
            .edit_file(req)
            .await
            .map_err(|e| format!("{:#}", e))?;

        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(
        description = "Conservatively analyze unused imports and symbols within explicit build-configuration scope. Auto-discovers compile_commands.json and accepts configured alternatives. Findings report referenced/unused_in_analyzed_configurations/inconclusive semantics, analyzed config IDs, completeness, unresolved dependency kinds, limitations, and safe_for_destructive_edit=false when build, generated, preprocessor, reflection, or dynamic wiring evidence is incomplete. Treat findings as evidence, never as safe to auto-delete."
    )]
    async fn find_unused(
        &self,
        Parameters(req): Parameters<FindUnusedRequest>,
    ) -> Result<String, String> {
        req.validate()?;

        let response = self
            .client
            .find_unused(req)
            .await
            .map_err(|e| format!("{:#}", e))?;

        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }
}

// Prompts for slash commands
#[prompt_router]
impl RagMcpServer {
    #[prompt(
        name = "index",
        description = "Index a codebase directory to enable semantic search (automatically performs full or incremental based on existing index)"
    )]
    async fn index_prompt(
        &self,
        Parameters(args): Parameters<serde_json::Value>,
    ) -> Result<GetPromptResult, McpError> {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        let messages = vec![PromptMessage::new_text(
            PromptMessageRole::User,
            format!(
                "Please index the codebase at path: '{}'. This will automatically perform a full index if this is the first time, or an incremental update if the codebase has been indexed before.",
                path
            ),
        )];

        Ok(GetPromptResult {
            description: Some(format!(
                "Index codebase at {} (auto-detects full/incremental)",
                path
            )),
            messages,
        })
    }

    #[prompt(
        name = "query",
        description = "Search the indexed codebase using semantic search"
    )]
    async fn query_prompt(
        &self,
        Parameters(args): Parameters<serde_json::Value>,
    ) -> Result<Vec<PromptMessage>, McpError> {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");

        Ok(vec![PromptMessage::new_text(
            PromptMessageRole::User,
            format!("Please search the codebase for: {}", query),
        )])
    }

    #[prompt(
        name = "stats",
        description = "Get statistics about the indexed codebase"
    )]
    async fn stats_prompt(&self) -> Vec<PromptMessage> {
        vec![PromptMessage::new_text(
            PromptMessageRole::User,
            "Please get statistics about the indexed codebase.",
        )]
    }

    #[prompt(
        name = "clear",
        description = "Clear all indexed data from the vector database"
    )]
    async fn clear_prompt(&self) -> Vec<PromptMessage> {
        vec![PromptMessage::new_text(
            PromptMessageRole::User,
            "Please clear all indexed data from the vector database.",
        )]
    }

    #[prompt(
        name = "search",
        description = "Advanced search with filters (file type, language, path)"
    )]
    async fn search_prompt(
        &self,
        Parameters(args): Parameters<serde_json::Value>,
    ) -> Result<Vec<PromptMessage>, McpError> {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");

        Ok(vec![PromptMessage::new_text(
            PromptMessageRole::User,
            format!("Please perform an advanced search for: {}", query),
        )])
    }

    #[prompt(
        name = "git-search",
        description = "Search git commit history using semantic search (automatically indexes commits on-demand)"
    )]
    async fn git_search_prompt(
        &self,
        Parameters(args): Parameters<serde_json::Value>,
    ) -> Result<Vec<PromptMessage>, McpError> {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        Ok(vec![PromptMessage::new_text(
            PromptMessageRole::User,
            format!(
                "Please search git commit history at path '{}' for: {}. This will automatically index commits as needed.",
                path, query
            ),
        )])
    }

    #[prompt(
        name = "definition",
        description = "Find where a symbol is defined at a given file location"
    )]
    async fn definition_prompt(
        &self,
        Parameters(args): Parameters<serde_json::Value>,
    ) -> Result<Vec<PromptMessage>, McpError> {
        let file = args.get("file").and_then(|v| v.as_str()).unwrap_or("");
        let line = args.get("line").and_then(|v| v.as_u64()).unwrap_or(1);
        let column = args.get("column").and_then(|v| v.as_u64()).unwrap_or(0);

        Ok(vec![PromptMessage::new_text(
            PromptMessageRole::User,
            format!(
                "Please find the definition of the symbol at file '{}', line {}, column {}.",
                file, line, column
            ),
        )])
    }

    #[prompt(
        name = "references",
        description = "Find all references to a symbol at a given file location"
    )]
    async fn references_prompt(
        &self,
        Parameters(args): Parameters<serde_json::Value>,
    ) -> Result<Vec<PromptMessage>, McpError> {
        let file = args.get("file").and_then(|v| v.as_str()).unwrap_or("");
        let line = args.get("line").and_then(|v| v.as_u64()).unwrap_or(1);
        let column = args.get("column").and_then(|v| v.as_u64()).unwrap_or(0);

        Ok(vec![PromptMessage::new_text(
            PromptMessageRole::User,
            format!(
                "Please find all references to the symbol at file '{}', line {}, column {}.",
                file, line, column
            ),
        )])
    }

    #[prompt(
        name = "callgraph",
        description = "Get the call graph (callers and callees) for a function at a given location"
    )]
    async fn callgraph_prompt(
        &self,
        Parameters(args): Parameters<serde_json::Value>,
    ) -> Result<Vec<PromptMessage>, McpError> {
        let file = args.get("file").and_then(|v| v.as_str()).unwrap_or("");
        let line = args.get("line").and_then(|v| v.as_u64()).unwrap_or(1);
        let column = args.get("column").and_then(|v| v.as_u64()).unwrap_or(0);

        Ok(vec![PromptMessage::new_text(
            PromptMessageRole::User,
            format!(
                "Please get the call graph for the function at file '{}', line {}, column {}. Show what calls this function and what it calls.",
                file, line, column
            ),
        )])
    }

    #[prompt(
        name = "read",
        description = "Read a slice (or all) of a file's current content"
    )]
    async fn read_prompt(
        &self,
        Parameters(args): Parameters<serde_json::Value>,
    ) -> Result<Vec<PromptMessage>, McpError> {
        let file = args.get("file").and_then(|v| v.as_str()).unwrap_or("");

        Ok(vec![PromptMessage::new_text(
            PromptMessageRole::User,
            format!("Please read the file '{}'.", file),
        )])
    }

    #[prompt(
        name = "edit",
        description = "Edit a file inside an indexed project and reindex it automatically"
    )]
    async fn edit_prompt(
        &self,
        Parameters(args): Parameters<serde_json::Value>,
    ) -> Result<Vec<PromptMessage>, McpError> {
        let file = args.get("file").and_then(|v| v.as_str()).unwrap_or("");

        Ok(vec![PromptMessage::new_text(
            PromptMessageRole::User,
            format!(
                "Please edit the file '{}'. First read it with read_file to get its current content and file_hash, then call edit_file with the new content and that expected_hash.",
                file
            ),
        )])
    }

    #[prompt(
        name = "unused",
        description = "Find unused imports and dead-code candidates in a file or directory"
    )]
    async fn unused_prompt(
        &self,
        Parameters(args): Parameters<serde_json::Value>,
    ) -> Result<Vec<PromptMessage>, McpError> {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        Ok(vec![PromptMessage::new_text(
            PromptMessageRole::User,
            format!(
                "Please run the find_unused tool on '{}' and summarize findings by status and analyzed build configuration. \
                 Highlight inconclusive limitations and never imply destructive-edit safety when it is false.",
                path
            ),
        )])
    }
}

#[tool_handler(router = self.tool_router)]
#[prompt_handler]
impl ServerHandler for RagMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::default(),
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .enable_prompts()
                .build(),
            server_info: Implementation {
                name: "project".into(),
                title: Some("Project RAG - Code Understanding with Semantic Search".into()),
                version: env!("CARGO_PKG_VERSION").into(),
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "RAG-based codebase indexing and semantic search. \
                Use index_codebase to create embeddings (automatically performs full or incremental indexing), \
                query_codebase to search, and search_by_filters for advanced queries. \
                Use read_file and edit_file to read and modify files inside an indexed project; \
                edit_file automatically reindexes the affected file. \
                Use find_unused to surface conservative unused-code evidence; findings are not deletion authorization."
                    .into(),
            ),
        }
    }
}

impl RagMcpServer {
    pub async fn serve_stdio() -> Result<()> {
        tracing::info!("Starting RAG MCP server");

        let server = Self::new().await.context("Failed to create MCP server")?;

        let transport = rmcp::transport::io::stdio();

        server.serve(transport).await?.waiting().await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests;
