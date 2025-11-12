use super::CodeChunk;
use crate::indexer::ast_parser::AstParser;
use crate::indexer::file_walker::FileInfo;
use crate::types::ChunkMetadata;
use std::time::{SystemTime, UNIX_EPOCH};

/// Strategy for chunking code
pub enum ChunkStrategy {
    /// Fixed number of lines per chunk
    FixedLines(usize),
    /// Sliding window with overlap
    SlidingWindow { size: usize, overlap: usize },
    /// AST-based chunking (functions, classes, methods)
    AstBased,
    /// Hybrid: AST-based with fallback to fixed lines
    Hybrid { fallback_lines: usize },
}

pub struct CodeChunker {
    strategy: ChunkStrategy,
}

impl CodeChunker {
    pub fn new(strategy: ChunkStrategy) -> Self {
        Self { strategy }
    }

    /// Create a chunker with default strategy (Hybrid AST with 50 line fallback)
    pub fn default_strategy() -> Self {
        Self::new(ChunkStrategy::Hybrid { fallback_lines: 50 })
    }

    /// Chunk a file into multiple code chunks
    pub fn chunk_file(&self, file_info: &FileInfo) -> Vec<CodeChunk> {
        match &self.strategy {
            ChunkStrategy::FixedLines(lines_per_chunk) => {
                self.chunk_fixed_lines(file_info, *lines_per_chunk)
            }
            ChunkStrategy::SlidingWindow { size, overlap } => {
                self.chunk_sliding_window(file_info, *size, *overlap)
            }
            ChunkStrategy::AstBased => self.chunk_ast_based(file_info),
            ChunkStrategy::Hybrid { fallback_lines } => {
                // Try AST-based first, fallback to fixed lines if it fails
                let ast_chunks = self.chunk_ast_based(file_info);
                if ast_chunks.is_empty() {
                    self.chunk_fixed_lines(file_info, *fallback_lines)
                } else {
                    ast_chunks
                }
            }
        }
    }

    /// Chunk using fixed number of lines
    fn chunk_fixed_lines(&self, file_info: &FileInfo, lines_per_chunk: usize) -> Vec<CodeChunk> {
        let lines: Vec<&str> = file_info.content.lines().collect();
        let mut chunks = Vec::new();

        if lines.is_empty() {
            return chunks;
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        for (chunk_idx, chunk_lines) in lines.chunks(lines_per_chunk).enumerate() {
            let start_line = chunk_idx * lines_per_chunk + 1;
            let end_line = start_line + chunk_lines.len() - 1;
            let content = chunk_lines.join("\n");

            // Skip empty chunks
            if content.trim().is_empty() {
                continue;
            }

            let metadata = ChunkMetadata {
                file_path: file_info.relative_path.clone(),
                project: file_info.project.clone(),
                start_line,
                end_line,
                language: file_info.language.clone(),
                extension: file_info.extension.clone(),
                file_hash: file_info.hash.clone(),
                indexed_at: timestamp,
            };

            chunks.push(CodeChunk { content, metadata });
        }

        chunks
    }

    /// Chunk using sliding window with overlap
    fn chunk_sliding_window(
        &self,
        file_info: &FileInfo,
        size: usize,
        overlap: usize,
    ) -> Vec<CodeChunk> {
        let lines: Vec<&str> = file_info.content.lines().collect();
        let mut chunks = Vec::new();

        if lines.is_empty() {
            return chunks;
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let step = if overlap < size { size - overlap } else { 1 };
        let mut start_idx = 0;

        while start_idx < lines.len() {
            let end_idx = (start_idx + size).min(lines.len());
            let chunk_lines = &lines[start_idx..end_idx];
            let content = chunk_lines.join("\n");

            // Skip empty chunks
            if content.trim().is_empty() {
                start_idx += step;
                continue;
            }

            let start_line = start_idx + 1;
            let end_line = end_idx;

            let metadata = ChunkMetadata {
                file_path: file_info.relative_path.clone(),
                project: file_info.project.clone(),
                start_line,
                end_line,
                language: file_info.language.clone(),
                extension: file_info.extension.clone(),
                file_hash: file_info.hash.clone(),
                indexed_at: timestamp,
            };

            chunks.push(CodeChunk { content, metadata });

            // Break if we've reached the end
            if end_idx >= lines.len() {
                break;
            }

            start_idx += step;
        }

        chunks
    }

    /// Chunk using AST-based parsing (functions, classes, methods)
    fn chunk_ast_based(&self, file_info: &FileInfo) -> Vec<CodeChunk> {
        // Check if we have an extension and can parse it
        let extension = match &file_info.extension {
            Some(ext) => ext,
            None => {
                tracing::debug!("No extension for AST parsing: {:?}", file_info.path);
                return Vec::new();
            }
        };

        // Try to create parser for this language
        let mut parser = match AstParser::new(extension) {
            Ok(p) => p,
            Err(_) => {
                tracing::debug!("Unsupported language for AST parsing: {}", extension);
                return Vec::new();
            }
        };

        // Parse the file
        let ast_nodes = match parser.parse(&file_info.content) {
            Ok(nodes) => nodes,
            Err(e) => {
                tracing::warn!("Failed to parse file {:?}: {}", file_info.path, e);
                return Vec::new();
            }
        };

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let mut chunks = Vec::new();
        let lines: Vec<&str> = file_info.content.lines().collect();

        for ast_node in ast_nodes {
            // Extract the content for this node
            let start_idx = ast_node.start_line.saturating_sub(1);
            let end_idx = ast_node.end_line.min(lines.len());

            if start_idx >= end_idx {
                continue;
            }

            let chunk_lines = &lines[start_idx..end_idx];
            let content = chunk_lines.join("\n");

            // Skip empty chunks
            if content.trim().is_empty() {
                continue;
            }

            let metadata = ChunkMetadata {
                file_path: file_info.relative_path.clone(),
                project: file_info.project.clone(),
                start_line: ast_node.start_line,
                end_line: ast_node.end_line,
                language: file_info.language.clone(),
                extension: file_info.extension.clone(),
                file_hash: file_info.hash.clone(),
                indexed_at: timestamp,
            };

            chunks.push(CodeChunk { content, metadata });
        }

        // If no chunks were created, log it
        if chunks.is_empty() {
            tracing::debug!("No AST chunks created for {:?}", file_info.path);
        }

        chunks
    }
}

impl Default for CodeChunker {
    fn default() -> Self {
        Self::default_strategy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn create_test_file_info(content: &str) -> FileInfo {
        FileInfo {
            path: PathBuf::from("test.rs"),
            relative_path: "test.rs".to_string(),
            project: None,
            extension: Some("rs".to_string()),
            language: Some("Rust".to_string()),
            content: content.to_string(),
            hash: "test_hash".to_string(),
        }
    }

    #[test]
    fn test_fixed_lines_chunking() {
        let content = (1..=100)
            .map(|i| format!("line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        let file_info = create_test_file_info(&content);

        let chunker = CodeChunker::new(ChunkStrategy::FixedLines(10));
        let chunks = chunker.chunk_file(&file_info);

        assert_eq!(chunks.len(), 10);
        assert_eq!(chunks[0].metadata.start_line, 1);
        assert_eq!(chunks[0].metadata.end_line, 10);
        assert_eq!(chunks[9].metadata.start_line, 91);
        assert_eq!(chunks[9].metadata.end_line, 100);
    }

    #[test]
    fn test_sliding_window_chunking() {
        let content = (1..=20)
            .map(|i| format!("line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        let file_info = create_test_file_info(&content);

        let chunker = CodeChunker::new(ChunkStrategy::SlidingWindow {
            size: 10,
            overlap: 5,
        });
        let chunks = chunker.chunk_file(&file_info);

        // With size=10 and overlap=5, step=5
        // Chunks: [1-10], [6-15], [11-20]
        assert!(chunks.len() >= 3);
        assert_eq!(chunks[0].metadata.start_line, 1);
    }

    #[test]
    fn test_default_strategy() {
        let chunker = CodeChunker::default_strategy();
        assert!(matches!(chunker.strategy, ChunkStrategy::Hybrid { .. }));
    }

    #[test]
    fn test_default() {
        let chunker = CodeChunker::default();
        assert!(matches!(chunker.strategy, ChunkStrategy::Hybrid { .. }));
    }

    #[test]
    fn test_empty_file() {
        let file_info = create_test_file_info("");
        let chunker = CodeChunker::new(ChunkStrategy::FixedLines(10));
        let chunks = chunker.chunk_file(&file_info);
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_whitespace_only_file() {
        let file_info = create_test_file_info("   \n\t\n   ");
        let chunker = CodeChunker::new(ChunkStrategy::FixedLines(10));
        let chunks = chunker.chunk_file(&file_info);
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_single_line_file() {
        let file_info = create_test_file_info("fn main() {}");
        let chunker = CodeChunker::new(ChunkStrategy::FixedLines(10));
        let chunks = chunker.chunk_file(&file_info);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].metadata.start_line, 1);
        assert_eq!(chunks[0].metadata.end_line, 1);
    }

    #[test]
    fn test_sliding_window_overlap_equal_size() {
        let content = (1..=20).map(|i| format!("line {}", i)).collect::<Vec<_>>().join("\n");
        let file_info = create_test_file_info(&content);

        let chunker = CodeChunker::new(ChunkStrategy::SlidingWindow {
            size: 10,
            overlap: 10,
        });
        let chunks = chunker.chunk_file(&file_info);
        // When overlap equals size, step should be 1
        assert!(chunks.len() > 10);
    }

    #[test]
    fn test_sliding_window_overlap_greater_than_size() {
        let content = (1..=20).map(|i| format!("line {}", i)).collect::<Vec<_>>().join("\n");
        let file_info = create_test_file_info(&content);

        let chunker = CodeChunker::new(ChunkStrategy::SlidingWindow {
            size: 10,
            overlap: 15,
        });
        let chunks = chunker.chunk_file(&file_info);
        // When overlap > size, step should be 1
        assert!(chunks.len() > 10);
    }

    #[test]
    fn test_ast_based_rust() {
        let content = r#"
fn hello() {
    println!("Hello");
}

fn world() {
    println!("World");
}
"#;
        let file_info = create_test_file_info(content);
        let chunker = CodeChunker::new(ChunkStrategy::AstBased);
        let chunks = chunker.chunk_file(&file_info);
        // Should extract two functions
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_ast_based_no_extension() {
        let mut file_info = create_test_file_info("fn main() {}");
        file_info.extension = None;
        let chunker = CodeChunker::new(ChunkStrategy::AstBased);
        let chunks = chunker.chunk_file(&file_info);
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_ast_based_unsupported_language() {
        let mut file_info = create_test_file_info("some content");
        file_info.extension = Some("txt".to_string());
        let chunker = CodeChunker::new(ChunkStrategy::AstBased);
        let chunks = chunker.chunk_file(&file_info);
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_hybrid_with_ast_success() {
        let content = r#"
fn hello() {
    println!("Hello");
}
"#;
        let file_info = create_test_file_info(content);
        let chunker = CodeChunker::new(ChunkStrategy::Hybrid { fallback_lines: 50 });
        let chunks = chunker.chunk_file(&file_info);
        // Should use AST parsing
        assert!(chunks.len() >= 1);
    }

    #[test]
    fn test_hybrid_fallback_to_fixed() {
        let mut file_info = create_test_file_info("line 1\nline 2\nline 3");
        file_info.extension = Some("txt".to_string());
        let chunker = CodeChunker::new(ChunkStrategy::Hybrid { fallback_lines: 2 });
        let chunks = chunker.chunk_file(&file_info);
        // Should fallback to fixed lines since .txt is not supported by AST
        assert!(chunks.len() >= 1);
    }

    #[test]
    fn test_metadata_fields() {
        let mut file_info = create_test_file_info("fn main() {}");
        file_info.project = Some("test-project".to_string());
        file_info.hash = "abc123".to_string();

        let chunker = CodeChunker::new(ChunkStrategy::FixedLines(10));
        let chunks = chunker.chunk_file(&file_info);

        assert_eq!(chunks.len(), 1);
        let chunk = &chunks[0];
        assert_eq!(chunk.metadata.file_path, "test.rs");
        assert_eq!(chunk.metadata.project, Some("test-project".to_string()));
        assert_eq!(chunk.metadata.language, Some("Rust".to_string()));
        assert_eq!(chunk.metadata.extension, Some("rs".to_string()));
        assert_eq!(chunk.metadata.file_hash, "abc123");
        assert!(chunk.metadata.indexed_at > 0);
    }

    #[test]
    fn test_sliding_window_empty_chunks_skipped() {
        let content = "line 1\n\n\n\nline 5";
        let file_info = create_test_file_info(content);
        let chunker = CodeChunker::new(ChunkStrategy::SlidingWindow {
            size: 2,
            overlap: 0,
        });
        let chunks = chunker.chunk_file(&file_info);
        // Should skip chunks with only whitespace
        assert!(chunks.len() >= 1);
        for chunk in chunks {
            assert!(!chunk.content.trim().is_empty());
        }
    }

    #[test]
    fn test_fixed_lines_empty_chunks_skipped() {
        let content = "line 1\n\n\nline 4";
        let file_info = create_test_file_info(content);
        let chunker = CodeChunker::new(ChunkStrategy::FixedLines(2));
        let chunks = chunker.chunk_file(&file_info);
        // Should have chunks but skip empty ones
        for chunk in chunks {
            assert!(!chunk.content.trim().is_empty());
        }
    }

    #[test]
    fn test_ast_based_invalid_syntax() {
        let content = "fn incomplete {"; // Invalid Rust
        let file_info = create_test_file_info(content);
        let chunker = CodeChunker::new(ChunkStrategy::AstBased);
        let chunks = chunker.chunk_file(&file_info);
        // Should handle parse errors gracefully
        assert_eq!(chunks.len(), 0);
    }
}
