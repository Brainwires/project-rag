//! # Project RAG - RAG-based Codebase Indexing and Semantic Search
//!
//! A dual-purpose Rust library and MCP server for semantic code search using RAG
//! (Retrieval-Augmented Generation).
//!
//! ## Overview
//!
//! Project RAG combines vector embeddings with BM25 keyword search to enable semantic
//! code search across large projects. It supports incremental indexing, git history search,
//! and provides both a Rust library API and an MCP server for AI assistant integration.
//!
//! ## Architecture
//!
//! - **RagClient**: Core library containing all functionality (embeddings, vector DB, indexing, search)
//! - **RagMcpServer**: Thin wrapper around RagClient that exposes functionality via MCP protocol
//! - Both library and MCP server are always built together - no feature flags needed
//!
//! ## Key Features
//!
//! - **Semantic Search**: FastEmbed (all-MiniLM-L6-v2) for local embeddings
//! - **Hybrid Search**: Combines vector similarity with BM25 keyword matching (RRF)
//! - **Dual Database Support**: LanceDB (embedded, default) or Qdrant (external server)
//! - **Smart Indexing**: Auto-detects full vs incremental updates with persistent caching
//! - **AST-Based Chunking**: Tree-sitter parsing for 12 programming languages
//! - **Git History Search**: Semantic search over commit history with on-demand indexing
//! - **Dual API**: Use as a Rust library or as an MCP server for AI assistants
//!
//! ## Library Usage Example
//!
//! ```no_run
//! use project_rag::{RagClient, IndexRequest, QueryRequest};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create client with default configuration
//!     let client = RagClient::new().await?;
//!
//!     // Index a codebase
//!     let index_req = IndexRequest {
//!         path: "/path/to/codebase".to_string(),
//!         project: Some("my-project".to_string()),
//!         include_patterns: vec!["**/*.rs".to_string()],
//!         exclude_patterns: vec!["**/target/**".to_string()],
//!         max_file_size: 1_048_576,
//!     };
//!     let index_response = client.index_codebase(index_req).await?;
//!     println!("Indexed {} files", index_response.files_indexed);
//!
//!     // Query the codebase
//!     let query_req = QueryRequest {
//!         query: "authentication logic".to_string(),
//!         project: Some("my-project".to_string()),
//!         limit: 10,
//!         min_score: 0.7,
//!         hybrid: true,
//!     };
//!     let query_response = client.query_codebase(query_req).await?;
//!     for result in query_response.results {
//!         println!("Found in {}: score {}", result.file_path, result.score);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## MCP Server Usage Example
//!
//! The MCP server wraps RagClient and exposes it via the MCP protocol:
//!
//! ```no_run
//! use project_rag::mcp_server::RagMcpServer;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create server (internally creates a RagClient)
//!     let server = RagMcpServer::new().await?;
//!
//!     // Serve over stdio (MCP protocol)
//!     server.serve_stdio().await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! Or you can create a server with an existing client:
//!
//! ```no_run
//! use project_rag::{RagClient, mcp_server::RagMcpServer};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create client with custom configuration
//!     let client = RagClient::new().await?;
//!
//!     // Wrap client in MCP server
//!     let server = RagMcpServer::with_client(Arc::new(client))?;
//!
//!     server.serve_stdio().await?;
//!     Ok(())
//! }
//! ```
//!
//! ## Modules
//!
//! - [`client`]: Core library client API with all functionality
//! - [`mcp_server`]: MCP protocol server implementation that wraps the client
//! - [`embedding`]: Embedding generation using FastEmbed
//! - [`vector_db`]: Vector database abstraction (LanceDB and Qdrant)
//! - [`bm25_search`]: BM25 keyword search using Tantivy
//! - [`indexer`]: File walking, AST parsing, and code chunking
//! - [`git`]: Git history walking and commit chunking
//! - [`cache`]: Persistent hash cache for incremental updates
//! - [`git_cache`]: Git commit tracking cache
//! - [`config`]: Configuration management with environment variable support
//! - [`types`]: Request/response types with validation
//! - [`error`]: Error types and result aliases
//! - [`paths`]: Path normalization utilities

// Core modules (always available)
/// BM25 keyword search using Tantivy for hybrid search
pub mod bm25_search;

/// Persistent hash cache for tracking file changes across restarts
pub mod cache;

/// Configuration management with environment variable overrides
pub mod config;

/// Embedding generation using FastEmbed (all-MiniLM-L6-v2)
pub mod embedding;

/// Error types and utilities
pub mod error;

/// Git repository walking and commit extraction
pub mod git;

/// Git commit tracking cache for incremental git history indexing
pub mod git_cache;

/// Glob pattern matching utilities for path filtering
pub mod glob_utils;

/// File walking, code chunking, and AST parsing
pub mod indexer;

/// Path normalization and utility functions
pub mod paths;

/// Code relationships: definitions, references, call graphs
pub mod relations;

/// Request/response types with validation
pub mod types;

/// Vector database abstraction supporting LanceDB and Qdrant
pub mod vector_db;

// Library client API (core functionality)
pub mod client;
pub use client::RagClient;

// MCP server (wraps the client and exposes via MCP protocol)
pub mod mcp_server;

// Re-export commonly used types for convenience
pub use types::{
    AdvancedSearchRequest, ClearRequest, ClearResponse, FindDefinitionRequest,
    FindDefinitionResponse, FindReferencesRequest, FindReferencesResponse, GetCallGraphRequest,
    GetCallGraphResponse, GitSearchResult, IndexRequest, IndexResponse, IndexingMode,
    LanguageStats, QueryRequest, QueryResponse, SearchGitHistoryRequest, SearchGitHistoryResponse,
    SearchResult, StatisticsRequest, StatisticsResponse,
};

pub use config::Config;
pub use error::RagError;
