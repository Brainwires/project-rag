//! MCP Server integration example
//!
//! This example demonstrates how to:
//! 1. Create a RagClient with custom configuration
//! 2. Wrap it in an MCP server
//! 3. Start the MCP server over stdio
//!
//! This is useful when you want to embed the MCP server in your own application
//! or customize its behavior.
//!
//! Run with: cargo run --example mcp_client
//! (The server will wait for MCP protocol messages on stdin)

use project_rag::{mcp_server::RagMcpServer, Config, RagClient};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging - useful for debugging MCP communication
    tracing_subscriber::fmt::init();

    println!("=== Project RAG - MCP Server Example ===");
    println!();
    println!("This example shows how to programmatically create and run the MCP server.");
    println!();

    // Option 1: Create server with default configuration
    // This is the simplest approach
    println!("Option 1: Creating MCP server with default configuration...");
    let _simple_server = RagMcpServer::new().await?;
    println!("  Server created successfully!");
    println!();

    // Option 2: Create server with custom client
    // This gives you more control over the configuration
    println!("Option 2: Creating MCP server with custom client...");

    // Load configuration (from environment, config file, or defaults)
    let config = Config::new()?;
    println!("  Loaded configuration:");
    println!("    Vector DB backend: {}", config.vector_db.backend);
    println!("    Min search score: {}", config.search.min_score);
    println!("    Default search limit: {}", config.search.limit);

    // Create client with custom configuration
    let client = RagClient::with_config(config).await?;
    println!("  RAG client created!");

    // Wrap client in MCP server
    let server = RagMcpServer::with_client(Arc::new(client))?;
    println!("  MCP server wrapper created!");
    println!();

    // The server exposes these MCP tools:
    println!("Available MCP Tools:");
    println!("  1. index_codebase     - Index a directory for semantic search");
    println!("  2. query_codebase     - Semantic search across indexed code");
    println!("  3. search_by_filters  - Advanced search with filters");
    println!("  4. get_statistics     - Get index statistics");
    println!("  5. clear_index        - Clear all indexed data");
    println!("  6. search_git_history - Search git commit history");
    println!("  7. find_definition    - Find where a symbol is defined");
    println!("  8. find_references    - Find all references to a symbol");
    println!("  9. get_call_graph     - Get function call graph");
    println!();

    println!("Available MCP Prompts (Slash Commands):");
    println!("  /project:index      /project:query       /project:search");
    println!("  /project:stats      /project:clear       /project:git-search");
    println!("  /project:definition /project:references  /project:callgraph");
    println!();

    // To actually run the server over stdio, uncomment the following:
    // println!("Starting MCP server on stdio...");
    // println!("(Send JSON-RPC messages to interact with the server)");
    // server.serve_stdio().await?;

    // For this example, we'll just demonstrate the setup
    println!("To run the server, use:");
    println!("  cargo run --release");
    println!("Or:");
    println!("  ./target/release/project-rag");
    println!();

    // Example: Using the client directly (without MCP protocol)
    println!("--- Direct Client Usage Example ---");
    println!();

    // Access the underlying client from the server
    // Note: In real usage, you'd typically use either the MCP protocol
    // OR the direct client API, not both
    let client = server.client();

    // Check if anything is indexed
    let stats = client.get_statistics().await?;

    println!("Current index statistics:");
    println!("  Total files: {}", stats.total_files);
    println!("  Total chunks: {}", stats.total_chunks);
    if !stats.language_breakdown.is_empty() {
        println!("  Languages indexed:");
        for lang in stats.language_breakdown.iter().take(5) {
            println!("    - {}: {} files", lang.language, lang.file_count);
        }
    }

    println!();
    println!("=== MCP Server Example Complete ===");

    Ok(())
}
