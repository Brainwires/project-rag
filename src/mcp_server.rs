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

        let response = crate::client::indexing::do_index_smart(
                &self.client,
                req.path,
                req.project,
                req.include_patterns,
                req.exclude_patterns,
                req.max_file_size,
                Some(peer),
                progress_token,
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
                query_codebase to search, and search_by_filters for advanced queries."
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
