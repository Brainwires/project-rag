use anyhow::{Context, Result};
use git2::{DiffOptions, Repository, Sort};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Information about a git commit
#[derive(Debug, Clone)]
pub struct CommitInfo {
    /// Full commit SHA hash (40 characters)
    pub hash: String,
    /// Commit message (first line and body)
    pub message: String,
    /// Author's name
    pub author_name: String,
    /// Author's email address
    pub author_email: String,
    /// Commit timestamp (Unix epoch seconds)
    pub commit_date: i64,
    /// List of file paths changed in this commit
    pub files_changed: Vec<String>,
    /// Unified diff content (truncated if too large)
    pub diff_content: String,
    /// SHA hashes of parent commits
    pub parent_hashes: Vec<String>,
}

/// Git repository walker for extracting commit information
pub struct GitWalker {
    repo: Repository,
    repo_path: PathBuf,
}

impl GitWalker {
    /// Discover and open a git repository from any path within it
    pub fn discover<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        // Discover the repository (walks up directory tree)
        let repo_path = Repository::discover(path)
            .context("Failed to discover git repository")?
            .path()
            .parent()
            .context("Invalid repository path")?
            .to_path_buf();

        let repo = Repository::open(&repo_path).context("Failed to open git repository")?;

        tracing::info!("Opened git repository at: {}", repo_path.display());

        Ok(Self { repo, repo_path })
    }

    /// Get the repository root path
    pub fn repo_path(&self) -> &Path {
        &self.repo_path
    }

    /// Get the current branch name, or None if detached HEAD
    pub fn current_branch(&self) -> Option<String> {
        self.repo.head().ok()?.shorthand().map(|s| s.to_string())
    }

    /// Iterate commits with filters
    pub fn iter_commits(
        &self,
        branch: Option<&str>,
        max_count: Option<usize>,
        since_date: Option<i64>,
        until_date: Option<i64>,
        skip_hashes: &HashSet<String>,
    ) -> Result<Vec<CommitInfo>> {
        let mut revwalk = self.repo.revwalk()?;
        revwalk.set_sorting(Sort::TIME | Sort::TOPOLOGICAL)?;

        // Determine starting point
        if let Some(branch_name) = branch {
            let reference = self
                .repo
                .find_branch(branch_name, git2::BranchType::Local)
                .context("Failed to find branch")?;
            let oid = reference.get().target().context("Branch has no target")?;
            revwalk.push(oid)?;
        } else {
            // Use HEAD
            revwalk.push_head()?;
        }

        let mut commits = Vec::new();
        let mut count = 0;
        let max = max_count.unwrap_or(usize::MAX);

        for oid in revwalk {
            if count >= max {
                break;
            }

            let oid = oid?;
            let commit = self.repo.find_commit(oid)?;
            let commit_hash = format!("{}", commit.id());

            // Skip if already indexed
            if skip_hashes.contains(&commit_hash) {
                tracing::debug!("Skipping already indexed commit: {}", commit_hash);
                continue;
            }

            let commit_time = commit.time().seconds();

            // Apply date filters
            if let Some(since) = since_date
                && commit_time < since
            {
                break; // Commits are sorted, no need to continue
            }

            if let Some(until) = until_date
                && commit_time > until
            {
                continue;
            }

            // Extract commit info
            let commit_info = self.extract_commit_info(&commit)?;
            commits.push(commit_info);
            count += 1;

            if count % 50 == 0 {
                tracing::debug!("Processed {} commits", count);
            }
        }

        tracing::info!("Extracted {} new commits", commits.len());
        Ok(commits)
    }

    /// Extract detailed information from a commit
    fn extract_commit_info(&self, commit: &git2::Commit) -> Result<CommitInfo> {
        let hash = format!("{}", commit.id());
        let message = commit.message().unwrap_or("").to_string();
        let author = commit.author();
        let author_name = author.name().unwrap_or("Unknown").to_string();
        let author_email = author.email().unwrap_or("").to_string();
        let commit_date = commit.time().seconds();

        // Extract parent hashes
        let parent_hashes: Vec<String> = commit.parents().map(|p| format!("{}", p.id())).collect();

        // Get diff and changed files
        let (files_changed, diff_content) = self.extract_diff(commit)?;

        Ok(CommitInfo {
            hash,
            message,
            author_name,
            author_email,
            commit_date,
            files_changed,
            diff_content,
            parent_hashes,
        })
    }

    /// Extract diff and list of changed files
    fn extract_diff(&self, commit: &git2::Commit) -> Result<(Vec<String>, String)> {
        let mut files_changed = Vec::new();
        let mut diff_content = String::new();
        let mut diff_truncated = false;

        let tree = commit.tree()?;

        // Get parent tree (if exists)
        let parent_tree = if commit.parent_count() > 0 {
            Some(commit.parent(0)?.tree()?)
        } else {
            None
        };

        let mut diff_opts = DiffOptions::new();
        diff_opts
            .context_lines(3)
            .interhunk_lines(0)
            .ignore_whitespace(false);

        let diff = if let Some(parent) = parent_tree {
            self.repo
                .diff_tree_to_tree(Some(&parent), Some(&tree), Some(&mut diff_opts))?
        } else {
            // First commit - diff against empty tree
            self.repo
                .diff_tree_to_tree(None, Some(&tree), Some(&mut diff_opts))?
        };

        // Iterate through deltas (file changes)
        for delta in diff.deltas() {
            if let Some(path) = delta.new_file().path() {
                files_changed.push(path.display().to_string());
            }
        }

        // Generate diff text
        diff.print(git2::DiffFormat::Patch, |_delta, _hunk, line| {
            // Stop adding content if already truncated (but continue processing - return true)
            if diff_truncated {
                return true;
            }

            // Skip binary files
            if line.origin() == 'B' {
                return true;
            }

            // Check if we're approaching the size limit before processing
            if diff_content.len() >= 100_000 {
                diff_truncated = true;
                return true; // Continue processing, just stop adding content
            }

            // Build diff content string - only if valid UTF-8
            let origin = line.origin();
            if let Ok(content) = std::str::from_utf8(line.content()) {
                match origin {
                    '+' | '-' | ' ' => {
                        diff_content.push(origin);
                        diff_content.push_str(content);
                    }
                    'F' => {
                        // File header
                        diff_content.push_str("--- ");
                        diff_content.push_str(content);
                    }
                    'H' => {
                        // Hunk header
                        diff_content.push_str(content);
                    }
                    _ => {}
                }
            } else {
                // Invalid UTF-8 - skip this line but continue processing
                tracing::debug!("Skipping diff line with invalid UTF-8");
            }

            // Always return true to continue processing (don't signal error to git2)
            true
        })?;

        // Truncate if too large and add marker
        if diff_content.len() > 8000 {
            diff_content.truncate(8000);
            diff_content.push_str("\n\n[... diff truncated ...]");
            tracing::warn!("Truncated large diff for commit {}", commit.id());
        }

        Ok((files_changed, diff_content))
    }

    /// Check if repository has any commits
    pub fn has_commits(&self) -> bool {
        self.repo.head().is_ok()
    }

    /// Get total commit count (approximation)
    pub fn estimate_commit_count(&self) -> Result<usize> {
        let mut revwalk = self.repo.revwalk()?;
        revwalk.push_head()?;
        Ok(revwalk.count())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discover_current_repo() {
        // This test assumes we're running in the project-rag repository
        let walker = GitWalker::discover(".").expect("Should find git repo");
        assert!(walker.repo_path().exists());
        assert!(walker.has_commits());
    }

    #[test]
    fn test_current_branch() {
        let walker = GitWalker::discover(".").expect("Should find git repo");
        let branch = walker.current_branch();
        assert!(branch.is_some(), "Should have a current branch");
    }

    #[test]
    fn test_iter_commits_limited() {
        let walker = GitWalker::discover(".").expect("Should find git repo");
        let skip = HashSet::new();

        let commits = walker
            .iter_commits(None, Some(5), None, None, &skip)
            .expect("Should iterate commits");

        assert!(commits.len() <= 5, "Should respect max_count");

        for commit in &commits {
            assert!(!commit.hash.is_empty(), "Commit hash should not be empty");
            assert!(
                !commit.author_name.is_empty(),
                "Author name should not be empty"
            );
        }
    }

    #[test]
    fn test_commit_info_structure() {
        let walker = GitWalker::discover(".").expect("Should find git repo");
        let skip = HashSet::new();

        let commits = walker
            .iter_commits(None, Some(1), None, None, &skip)
            .expect("Should get commits");

        if let Some(commit) = commits.first() {
            assert_eq!(commit.hash.len(), 40, "Git SHA should be 40 chars");
            assert!(commit.commit_date > 0, "Commit date should be positive");
        }
    }

    #[test]
    fn test_skip_hashes() {
        let walker = GitWalker::discover(".").expect("Should find git repo");
        let skip = HashSet::new();

        // Get first commit
        let commits = walker
            .iter_commits(None, Some(1), None, None, &skip)
            .expect("Should get commits");

        if let Some(first_commit) = commits.first() {
            let mut skip_set = HashSet::new();
            skip_set.insert(first_commit.hash.clone());

            // Try again with that commit in skip set
            let commits2 = walker
                .iter_commits(None, Some(1), None, None, &skip_set)
                .expect("Should get commits");

            // Should get different commit (or fewer commits if only one exists)
            if let Some(second_commit) = commits2.first() {
                assert_ne!(
                    first_commit.hash, second_commit.hash,
                    "Should skip specified commit"
                );
            }
        }
    }
}
