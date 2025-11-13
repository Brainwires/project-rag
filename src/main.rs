use anyhow::Result;
use clap::{Parser, Subcommand};
use project_rag::mcp_server::RagMcpServer;
use std::panic;

/// Project-RAG: RAG-based codebase indexing and semantic search MCP server
#[derive(Parser)]
#[command(name = "project-rag")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "MCP server for semantic code search with RAG capabilities", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the MCP server over stdio (default mode)
    Serve,

    /// Show version and system information
    Version,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Parse CLI arguments
    let cli = Cli::parse();

    // Handle commands
    match cli.command {
        Some(Commands::Version) => {
            show_version_info();
            return Ok(());
        }
        Some(Commands::Serve) | None => {
            // Set up global panic handler
            setup_panic_handler();

            // Start the RAG MCP server over stdio with error handling
            if let Err(e) = RagMcpServer::serve_stdio().await {
                tracing::error!("Fatal error in MCP server: {:#}", e);
                eprintln!("Fatal error: {:#}", e);
                std::process::exit(1);
            }
        }
    }

    Ok(())
}

/// Display comprehensive version and system information
fn show_version_info() {
    // Basic version info
    println!("project-rag v{}", env!("CARGO_PKG_VERSION"));
    println!();

    // System information
    println!("System Information:");
    println!("  Build Date:      {}", env!("BUILD_TIMESTAMP"));
    println!("  Git Commit:      {}", env!("GIT_COMMIT_HASH"));
    println!("  Rust Version:    {}", env!("CARGO_PKG_RUST_VERSION"));
    println!();

    // Vector database configuration
    println!("Vector Database:");
    let backend = env!("VECTOR_DB_BACKEND");
    println!("  Backend:         {}", backend);

    #[cfg(not(feature = "qdrant-backend"))]
    {
        use project_rag::vector_db::lance_client::LanceVectorDB;
        let default_path = LanceVectorDB::default_lancedb_path();
        println!("  Default Path:    {}", default_path);
        println!("  Type:            Embedded (no external server required)");
    }

    #[cfg(feature = "qdrant-backend")]
    {
        use project_rag::vector_db::qdrant_client::QdrantVectorDB;
        let default_url = QdrantVectorDB::default_url();
        println!("  Default URL:     {}", default_url);
        println!("  Type:            External server (requires Qdrant running)");
    }
    println!();

    // Embedding model information
    println!("Embedding Model:");
    println!("  Model:           all-MiniLM-L6-v2");
    println!("  Dimensions:      384");
    println!("  Provider:        FastEmbed (local, no API calls)");
    println!();

    // Configuration
    println!("Configuration:");
    use project_rag::paths::PlatformPaths;
    let config_path = PlatformPaths::default_config_path();
    println!("  Config File:     {}", config_path.display());
    println!("  Config Priority: CLI args > Env vars > Config file > Defaults");
    println!("  Env Prefix:      PROJECT_RAG_*");
    println!();

    // Additional features
    println!("Features:");
    println!("  Hybrid Search:   Enabled (Vector + BM25 keyword search)");
    println!("  AST Chunking:    12 languages supported");
    println!("  Git History:     Semantic search across commits");
    println!("  Incremental:     Smart indexing (auto-detects changes)");
    println!();

    // Supported languages
    println!("Supported Languages:");
    println!("  Programming:     Rust, Python, JavaScript, TypeScript, Go, Java,");
    println!("                   Swift, C, C++, C#, Ruby, PHP, Kotlin, Scala");
    println!("  Configuration:   JSON, YAML, TOML, XML");
    println!("  Markup:          HTML, CSS, SCSS, Markdown");
    println!("  Other:           Shell, SQL, Text");
}

/// Set up a global panic handler that logs panic information
fn setup_panic_handler() {
    panic::set_hook(Box::new(|panic_info| {
        let backtrace = std::backtrace::Backtrace::capture();

        let location = panic_info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "unknown location".to_string());

        let message = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic message".to_string()
        };

        // Log to tracing system
        tracing::error!(
            "PANIC at {}: {}\nBacktrace:\n{:?}",
            location,
            message,
            backtrace
        );

        // Also log to stderr for immediate visibility
        eprintln!("\n!!! PANIC !!!");
        eprintln!("Location: {}", location);
        eprintln!("Message: {}", message);
        eprintln!("Backtrace:\n{:?}", backtrace);
        eprintln!("!!! END PANIC !!!\n");
    }));

    tracing::info!("Global panic handler initialized");
}
