use anyhow::Result;
use project_rag::mcp_server::RagMcpServer;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Start the RAG MCP server over stdio
    RagMcpServer::serve_stdio().await?;

    Ok(())
}
