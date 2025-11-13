use crate::git::walker::CommitInfo;
use crate::indexer::CodeChunk;
use crate::types::ChunkMetadata;
use anyhow::Result;

/// Converts git commits into chunks suitable for embedding
pub struct CommitChunker {
    /// Maximum content length before truncation
    max_content_length: usize,
}

impl CommitChunker {
    /// Create a new commit chunker with default settings
    pub fn new() -> Self {
        Self {
            max_content_length: 6000, // ~1500 tokens for all-MiniLM-L6-v2
        }
    }

    /// Create with custom max content length
    pub fn with_max_length(max_content_length: usize) -> Self {
        Self { max_content_length }
    }

    /// Convert a commit into a chunk for embedding
    pub fn commit_to_chunk(
        &self,
        commit: &CommitInfo,
        repo_path: &str,
        project: Option<String>,
    ) -> Result<CodeChunk> {
        // Build searchable content: message + diff
        let mut content = String::new();

        // Add commit message
        content.push_str("Commit Message:\n");
        content.push_str(&commit.message);
        content.push_str("\n\n");

        // Add author info
        content.push_str("Author: ");
        content.push_str(&commit.author_name);
        if !commit.author_email.is_empty() {
            content.push_str(" <");
            content.push_str(&commit.author_email);
            content.push('>');
        }
        content.push_str("\n\n");

        // Add files changed
        if !commit.files_changed.is_empty() {
            content.push_str("Files Changed:\n");
            for file in &commit.files_changed {
                content.push_str("- ");
                content.push_str(file);
                content.push('\n');
            }
            content.push('\n');
        }

        // Add diff content
        if !commit.diff_content.is_empty() {
            content.push_str("Diff:\n");
            content.push_str(&commit.diff_content);
        }

        // Truncate if too long
        if content.len() > self.max_content_length {
            content.truncate(self.max_content_length);
            content.push_str("\n\n[... content truncated for embedding ...]");
        }

        // Create chunk metadata
        // Note: Git commits don't have line numbers, so we use 0
        let metadata = ChunkMetadata {
            file_path: format!("git://{}", repo_path),
            project,
            start_line: 0,
            end_line: 0,
            language: Some("git-commit".to_string()),
            extension: Some("commit".to_string()),
            file_hash: commit.hash.clone(),
            indexed_at: commit.commit_date,
        };

        Ok(CodeChunk { content, metadata })
    }

    /// Batch convert commits to chunks
    pub fn commits_to_chunks(
        &self,
        commits: &[CommitInfo],
        repo_path: &str,
        project: Option<String>,
    ) -> Result<Vec<CodeChunk>> {
        commits
            .iter()
            .map(|commit| self.commit_to_chunk(commit, repo_path, project.clone()))
            .collect()
    }
}

impl Default for CommitChunker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_commit() -> CommitInfo {
        CommitInfo {
            hash: "abc123def456".to_string(),
            message:
                "Fix authentication bug\n\nThis commit fixes a critical bug in the auth module."
                    .to_string(),
            author_name: "John Doe".to_string(),
            author_email: "john@example.com".to_string(),
            commit_date: 1704067200, // 2024-01-01
            files_changed: vec!["src/auth.rs".to_string(), "tests/auth_tests.rs".to_string()],
            diff_content: "@@ -10,7 +10,7 @@\n-    old_line\n+    new_line\n".to_string(),
            parent_hashes: vec!["parent123".to_string()],
        }
    }

    #[test]
    fn test_commit_to_chunk() {
        let chunker = CommitChunker::new();
        let commit = create_test_commit();

        let chunk = chunker
            .commit_to_chunk(&commit, "/repo/path", None)
            .expect("Should convert commit to chunk");

        assert_eq!(chunk.metadata.file_path, "git:///repo/path");
        assert_eq!(chunk.metadata.language, Some("git-commit".to_string()));
        assert_eq!(chunk.metadata.file_hash, "abc123def456");
        assert!(chunk.content.contains("Fix authentication bug"));
        assert!(chunk.content.contains("John Doe"));
        assert!(chunk.content.contains("src/auth.rs"));
        assert!(chunk.content.contains("new_line"));
    }

    #[test]
    fn test_content_truncation() {
        let chunker = CommitChunker::with_max_length(100);
        let mut commit = create_test_commit();
        commit.diff_content = "x".repeat(10000); // Very large diff

        let chunk = chunker
            .commit_to_chunk(&commit, "/repo/path", None)
            .expect("Should convert commit");

        assert!(chunk.content.len() <= 150); // 100 + truncation message
        assert!(chunk.content.contains("[... content truncated"));
    }

    #[test]
    fn test_commits_to_chunks_batch() {
        let chunker = CommitChunker::new();
        let commits = vec![create_test_commit(), {
            let mut c = create_test_commit();
            c.hash = "different_hash".to_string();
            c
        }];

        let chunks = chunker
            .commits_to_chunks(&commits, "/repo/path", Some("my-project".to_string()))
            .expect("Should convert batch");

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].metadata.file_hash, "abc123def456");
        assert_eq!(chunks[1].metadata.file_hash, "different_hash");
        assert_eq!(chunks[0].metadata.project, Some("my-project".to_string()));
    }

    #[test]
    fn test_empty_author_email() {
        let chunker = CommitChunker::new();
        let mut commit = create_test_commit();
        commit.author_email = String::new();

        let chunk = chunker
            .commit_to_chunk(&commit, "/repo/path", None)
            .expect("Should handle empty email");

        assert!(chunk.content.contains("John Doe"));
        assert!(!chunk.content.contains("<>"));
    }

    #[test]
    fn test_no_files_changed() {
        let chunker = CommitChunker::new();
        let mut commit = create_test_commit();
        commit.files_changed = vec![];

        let chunk = chunker
            .commit_to_chunk(&commit, "/repo/path", None)
            .expect("Should handle no files");

        assert!(!chunk.content.contains("Files Changed:"));
    }

    #[test]
    fn test_no_diff_content() {
        let chunker = CommitChunker::new();
        let mut commit = create_test_commit();
        commit.diff_content = String::new();

        let chunk = chunker
            .commit_to_chunk(&commit, "/repo/path", None)
            .expect("Should handle no diff");

        assert!(!chunk.content.contains("Diff:"));
    }
}
