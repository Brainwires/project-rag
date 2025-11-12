use anyhow::Result;
use project_rag::mcp_server::RagMcpServer;
use std::panic;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Set up global panic handler
    setup_panic_handler();

    // Start the RAG MCP server over stdio with error handling
    if let Err(e) = RagMcpServer::serve_stdio().await {
        tracing::error!("Fatal error in MCP server: {:#}", e);
        eprintln!("Fatal error: {:#}", e);
        std::process::exit(1);
    }

    Ok(())
}

/// Set up a global panic handler that logs panic information
fn setup_panic_handler() {
    panic::set_hook(Box::new(|panic_info| {
        let backtrace = std::backtrace::Backtrace::capture();

        let location = panic_info.location()
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
