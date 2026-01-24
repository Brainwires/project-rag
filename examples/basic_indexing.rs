//! Basic library usage example
//!
//! This example demonstrates how to use project-rag as a library to:
//! 1. Index a codebase directory
//! 2. Run semantic queries
//! 3. Print search results
//!
//! Run with: cargo run --example basic_indexing -- /path/to/codebase

use project_rag::{IndexRequest, QueryRequest, RagClient};
use std::env;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Get codebase path from command line or use current directory
    let codebase_path = env::args()
        .nth(1)
        .unwrap_or_else(|| ".".to_string());

    println!("=== Project RAG - Basic Indexing Example ===\n");

    // Create client with default configuration (uses LanceDB embedded database)
    println!("Creating RAG client...");
    let client = RagClient::new().await?;
    println!("Client created successfully!\n");

    // Index the codebase
    println!("Indexing codebase at: {}", codebase_path);
    let index_req = IndexRequest {
        path: codebase_path.clone(),
        project: Some("example-project".to_string()),
        include_patterns: vec![], // Empty = include all supported file types
        exclude_patterns: vec!["**/target/**".to_string(), "**/node_modules/**".to_string()],
        max_file_size: 1_048_576, // 1 MB
    };

    let index_response = client.index_codebase(index_req).await?;
    println!("\nIndexing complete!");
    println!("  Mode: {:?}", index_response.mode);
    println!("  Files indexed: {}", index_response.files_indexed);
    println!("  Chunks created: {}", index_response.chunks_created);
    println!("  Duration: {}ms", index_response.duration_ms);
    if !index_response.errors.is_empty() {
        println!("  Errors: {:?}", index_response.errors);
    }

    // Run a semantic query
    println!("\n--- Running Semantic Query ---\n");
    let queries = vec![
        "main function entry point",
        "error handling",
        "configuration settings",
    ];

    for query_text in queries {
        println!("Query: \"{}\"", query_text);
        let query_req = QueryRequest {
            query: query_text.to_string(),
            project: Some("example-project".to_string()),
            path: None,
            limit: 3,
            min_score: 0.5,
            hybrid: true, // Enable hybrid search (vector + keyword)
        };

        let query_response = client.query_codebase(query_req).await?;
        println!(
            "  Found {} results in {}ms",
            query_response.results.len(),
            query_response.duration_ms
        );

        for (i, result) in query_response.results.iter().enumerate() {
            println!(
                "  {}. {} (lines {}-{}) - score: {:.3}",
                i + 1,
                result.file_path,
                result.start_line,
                result.end_line,
                result.score
            );
            // Show first line of content as preview
            if let Some(first_line) = result.content.lines().next() {
                let preview = if first_line.len() > 60 {
                    format!("{}...", &first_line[..60])
                } else {
                    first_line.to_string()
                };
                println!("     Preview: {}", preview.trim());
            }
        }
        println!();
    }

    println!("=== Example Complete ===");
    Ok(())
}
