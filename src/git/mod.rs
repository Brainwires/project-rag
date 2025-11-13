//! Git repository operations for semantic search over commit history
//!
//! Provides functionality to walk git repositories, extract commit information,
//! and chunk commits into searchable units for vector indexing.

/// Commit chunking for converting git commits into searchable text chunks
pub mod chunker;
/// Git repository walking and commit extraction
pub mod walker;

pub use chunker::CommitChunker;
pub use walker::GitWalker;
