//! # Project RAG - RAG-based Codebase Indexing and Semantic Search
//!
//! A Rust-based Model Context Protocol (MCP) server that provides AI assistants with
//! RAG (Retrieval-Augmented Generation) capabilities for understanding massive codebases.
//!
//! ## Overview
//!
//! Project RAG combines vector embeddings with BM25 keyword search to enable semantic
//! code search across large projects. It supports incremental indexing, git history search,
//! and provides an MCP server interface for AI assistants like Claude.
//!
//! ## Key Features
//!
//! - **Semantic Search**: FastEmbed (all-MiniLM-L6-v2) for local embeddings
//! - **Hybrid Search**: Combines vector similarity with BM25 keyword matching (RRF)
//! - **Dual Database Support**: LanceDB (embedded, default) or Qdrant (external server)
//! - **Smart Indexing**: Auto-detects full vs incremental updates with persistent caching
//! - **AST-Based Chunking**: Tree-sitter parsing for 12 programming languages
//! - **Git History Search**: Semantic search over commit history with on-demand indexing
//! - **MCP Protocol**: 5 tools and 5 slash commands for AI assistant integration
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐
//! │   MCP Client    │  (Claude, VS Code, etc.)
//! └────────┬────────┘
//!          │ stdio
//! ┌────────▼────────┐
//! │  RagMcpServer   │  (5 tools, 5 prompts)
//! └────────┬────────┘
//!          │
//!    ┌─────┴─────┬──────────┬─────────────┐
//!    │           │          │             │
//! ┌──▼──┐  ┌────▼────┐  ┌──▼──┐   ┌─────▼──────┐
//! │FastE│  │LanceDB/ │  │BM25 │   │ HashCache  │
//! │mbed │  │Qdrant   │  │(Tant│   │(Persist)   │
//! └─────┘  └─────────┘  │ivy) │   └────────────┘
//!                       └─────┘
//! ```
//!
//! ## Modules
//!
//! - [`mcp_server`]: MCP protocol server implementation with tools and prompts
//! - [`embedding`]: Embedding generation using FastEmbed
//! - [`vector_db`]: Vector database abstraction (LanceDB and Qdrant)
//! - [`bm25_search`]: BM25 keyword search using Tantivy
//! - [`indexer`]: File walking, AST parsing, and code chunking
//! - [`git`]: Git history walking and commit chunking
//! - [`cache`]: Persistent hash cache for incremental updates
//! - [`git_cache`]: Git commit tracking cache
//! - [`config`]: Configuration management with environment variable support
//! - [`types`]: MCP request/response types with JSON schema
//! - [`error`]: Error types and result aliases
//! - [`paths`]: Path normalization utilities
//!
//! ## Usage Example
//!
//! ```no_run
//! use project_rag::mcp_server::RagMcpServer;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create server with default configuration
//!     let server = RagMcpServer::new().await?;
//!
//!     // Serve over stdio (MCP protocol)
//!     server.serve_stdio().await?;
//!
//!     Ok(())
//! }
//! ```

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

/// File walking, code chunking, and AST parsing
pub mod indexer;

/// MCP server implementation with tools and prompts
pub mod mcp_server;

/// Path normalization and utility functions
pub mod paths;

/// MCP request/response types with JSON schema definitions
pub mod types;

/// Vector database abstraction supporting LanceDB and Qdrant
pub mod vector_db;
