mod ast_parser;
mod chunker;
mod file_walker;

pub use ast_parser::AstParser;
pub use chunker::{ChunkStrategy, CodeChunker};
pub use file_walker::FileWalker;

use crate::types::ChunkMetadata;

/// Represents a code chunk ready for embedding
#[derive(Debug, Clone)]
pub struct CodeChunk {
    pub content: String,
    pub metadata: ChunkMetadata,
}
