use anyhow::Result;
use project_rag::mcp_test_minimal::TestServer;
use rmcp::service::ServiceExt;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Start the test MCP server over stdio
    let server = TestServer::new();
    let transport = rmcp::transport::io::stdio();
    server.serve(transport).await?.waiting().await?;

    Ok(())
}
