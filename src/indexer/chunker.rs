use super::CodeChunk;
use crate::indexer::file_walker::FileInfo;
use crate::types::ChunkMetadata;
use std::time::{SystemTime, UNIX_EPOCH};

/// Strategy for chunking code
pub enum ChunkStrategy {
    /// Fixed number of lines per chunk
    FixedLines(usize),
    /// Sliding window with overlap
    SlidingWindow { size: usize, overlap: usize },
}

pub struct CodeChunker {
    strategy: ChunkStrategy,
}

impl CodeChunker {
    pub fn new(strategy: ChunkStrategy) -> Self {
        Self { strategy }
    }

    /// Create a chunker with default strategy (50 lines per chunk)
    pub fn default_strategy() -> Self {
        Self::new(ChunkStrategy::FixedLines(50))
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
            extension: Some("rs".to_string()),
            language: Some("Rust".to_string()),
            content: content.to_string(),
            hash: "test_hash".to_string(),
        }
    }

    #[test]
    fn test_fixed_lines_chunking() {
        let content = (1..=100).map(|i| format!("line {}", i)).collect::<Vec<_>>().join("\n");
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
        let content = (1..=20).map(|i| format!("line {}", i)).collect::<Vec<_>>().join("\n");
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
}
