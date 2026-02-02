//! Advanced search example with filters
//!
//! This example demonstrates how to use the advanced search features:
//! 1. Filter by file extensions
//! 2. Filter by programming languages
//! 3. Filter by path patterns
//! 4. Use hybrid search (vector + BM25 keyword)
//!
//! Run with: cargo run --example advanced_search -- /path/to/codebase

use project_rag::{AdvancedSearchRequest, IndexRequest, RagClient};
use std::env;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Get codebase path from command line
    let codebase_path = env::args().nth(1).unwrap_or_else(|| ".".to_string());

    println!("=== Project RAG - Advanced Search Example ===\n");

    // Create client
    let client = RagClient::new().await?;

    // First, ensure the codebase is indexed
    println!("Ensuring codebase is indexed...");
    let index_req = IndexRequest {
        path: codebase_path.clone(),
        project: Some("advanced-demo".to_string()),
        include_patterns: vec![],
        exclude_patterns: vec!["**/target/**".to_string()],
        max_file_size: 1_048_576,
    };
    let _ = client.index_codebase(index_req).await?;

    // Get statistics to see what we have
    println!("\n--- Index Statistics ---\n");
    let stats = client.get_statistics().await?;
    println!("Total files: {}", stats.total_files);
    println!("Total chunks: {}", stats.total_chunks);
    println!("Languages:");
    for lang in &stats.language_breakdown {
        println!(
            "  - {}: {} files ({} chunks)",
            lang.language, lang.file_count, lang.chunk_count
        );
    }

    // Example 1: Search only in Rust files
    println!("\n--- Search 1: Only Rust Files ---\n");
    let search_req = AdvancedSearchRequest {
        query: "error handling".to_string(),
        project: Some("advanced-demo".to_string()),
        path: None,
        limit: 5,
        min_score: 0.5,
        file_extensions: vec!["rs".to_string()],
        languages: vec![],
        path_patterns: vec![],
    };
    let results = client.search_with_filters(search_req).await?;
    println!("Query: \"error handling\" in .rs files");
    println!("Found {} results:", results.results.len());
    for (i, result) in results.results.iter().take(3).enumerate() {
        println!(
            "  {}. {} (lines {}-{}) - score: {:.3}",
            i + 1,
            result.file_path,
            result.start_line,
            result.end_line,
            result.score
        );
    }

    // Example 2: Search by programming language
    println!("\n--- Search 2: By Language (Rust) ---\n");
    let search_req = AdvancedSearchRequest {
        query: "async function".to_string(),
        project: Some("advanced-demo".to_string()),
        path: None,
        limit: 5,
        min_score: 0.5,
        file_extensions: vec![],
        languages: vec!["Rust".to_string()],
        path_patterns: vec![],
    };
    let results = client.search_with_filters(search_req).await?;
    println!("Query: \"async function\" in Rust language");
    println!("Found {} results:", results.results.len());
    for (i, result) in results.results.iter().take(3).enumerate() {
        println!(
            "  {}. {} (lines {}-{}) - score: {:.3}",
            i + 1,
            result.file_path,
            result.start_line,
            result.end_line,
            result.score
        );
    }

    // Example 3: Search by path pattern
    println!("\n--- Search 3: By Path Pattern ---\n");
    let search_req = AdvancedSearchRequest {
        query: "database".to_string(),
        project: Some("advanced-demo".to_string()),
        path: None,
        limit: 5,
        min_score: 0.4,
        file_extensions: vec![],
        languages: vec![],
        path_patterns: vec!["src/".to_string()],
    };
    let results = client.search_with_filters(search_req).await?;
    println!("Query: \"database\" in src/ directory");
    println!("Found {} results:", results.results.len());
    for (i, result) in results.results.iter().take(3).enumerate() {
        println!(
            "  {}. {} (lines {}-{}) - score: {:.3}",
            i + 1,
            result.file_path,
            result.start_line,
            result.end_line,
            result.score
        );
    }

    // Example 4: Combined filters
    println!("\n--- Search 4: Combined Filters ---\n");
    let search_req = AdvancedSearchRequest {
        query: "configuration".to_string(),
        project: Some("advanced-demo".to_string()),
        path: None,
        limit: 5,
        min_score: 0.4,
        file_extensions: vec!["rs".to_string(), "toml".to_string()],
        languages: vec![],
        path_patterns: vec![],
    };
    let results = client.search_with_filters(search_req).await?;
    println!("Query: \"configuration\" in .rs and .toml files");
    println!("Found {} results:", results.results.len());
    for (i, result) in results.results.iter().take(5).enumerate() {
        println!(
            "  {}. {} [{}] - score: {:.3}",
            i + 1,
            result.file_path,
            result.language,
            result.score
        );
    }

    println!("\n=== Advanced Search Example Complete ===");
    Ok(())
}
