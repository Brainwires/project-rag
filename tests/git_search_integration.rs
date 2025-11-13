use project_rag::git::{CommitChunker, GitWalker};
use project_rag::git_cache::GitCache;
use std::collections::HashSet;

#[test]
fn test_git_walker_discovers_repo() {
    // Test that we can discover the project-rag repository
    let result = GitWalker::discover(".");
    assert!(result.is_ok(), "Should discover git repository");

    let walker = result.unwrap();
    assert!(walker.repo_path().exists(), "Repository path should exist");
    assert!(walker.has_commits(), "Repository should have commits");
}

#[test]
fn test_git_walker_iter_commits() {
    let walker = GitWalker::discover(".").expect("Should discover repo");
    let skip = HashSet::new();

    // Get last 5 commits
    let commits = walker
        .iter_commits(None, Some(5), None, None, &skip)
        .expect("Should get commits");

    assert!(commits.len() <= 5, "Should get at most 5 commits");
    assert!(!commits.is_empty(), "Should get at least one commit");

    // Verify commit structure
    for commit in &commits {
        assert!(!commit.hash.is_empty(), "Commit hash should not be empty");
        assert!(
            !commit.author_name.is_empty(),
            "Author name should not be empty"
        );
        assert!(commit.commit_date > 0, "Commit date should be positive");
    }

    println!("✓ Found {} commits", commits.len());
    println!(
        "✓ Latest commit: {} by {}",
        commits[0].hash, commits[0].author_name
    );
}

#[test]
fn test_commit_chunker() {
    let walker = GitWalker::discover(".").expect("Should discover repo");
    let skip = HashSet::new();

    // Get one commit
    let commits = walker
        .iter_commits(None, Some(1), None, None, &skip)
        .expect("Should get commits");

    assert!(!commits.is_empty(), "Should have at least one commit");

    let commit = &commits[0];
    let chunker = CommitChunker::new();
    let chunk = chunker
        .commit_to_chunk(commit, "/test/repo", None)
        .expect("Should convert commit to chunk");

    // Verify chunk structure
    assert!(
        chunk.content.contains("Commit Message:"),
        "Should have commit message section"
    );
    assert!(
        chunk.content.contains("Author:"),
        "Should have author section"
    );
    assert_eq!(
        chunk.metadata.language,
        Some("git-commit".to_string()),
        "Should have git-commit language"
    );
    assert_eq!(
        chunk.metadata.file_hash, commit.hash,
        "File hash should match commit hash"
    );

    println!(
        "✓ Created chunk with {} bytes of content",
        chunk.content.len()
    );
}

#[test]
fn test_git_cache_operations() {
    let mut cache = GitCache::default();

    // Test adding commits
    let mut commits = HashSet::new();
    commits.insert("abc123".to_string());
    commits.insert("def456".to_string());

    cache.add_commits("/repo/path".to_string(), commits);

    assert_eq!(cache.commit_count("/repo/path"), 2, "Should have 2 commits");
    assert!(
        cache.has_commit("/repo/path", "abc123"),
        "Should have first commit"
    );
    assert!(
        cache.has_commit("/repo/path", "def456"),
        "Should have second commit"
    );
    assert!(
        !cache.has_commit("/repo/path", "xyz789"),
        "Should not have non-existent commit"
    );

    println!("✓ Git cache operations working correctly");
}

#[test]
fn test_git_search_with_recent_commits() {
    let walker = GitWalker::discover(".").expect("Should discover repo");
    let skip = HashSet::new();

    // Get last 3 commits
    let commits = walker
        .iter_commits(None, Some(3), None, None, &skip)
        .expect("Should get commits");

    println!("\n=== Recent Commits ===");
    for (i, commit) in commits.iter().enumerate() {
        println!(
            "{}. {} - {}",
            i + 1,
            &commit.hash[..8],
            commit.message.lines().next().unwrap_or("(no message)")
        );
        println!(
            "   Author: {} <{}>",
            commit.author_name, commit.author_email
        );
        println!("   Files: {}", commit.files_changed.len());
    }

    // Convert to chunks
    let chunker = CommitChunker::new();
    let chunks = chunker
        .commits_to_chunks(&commits, ".", None)
        .expect("Should create chunks");

    assert_eq!(
        chunks.len(),
        commits.len(),
        "Should have one chunk per commit"
    );

    println!(
        "\n✓ Successfully processed {} commits into searchable chunks",
        chunks.len()
    );
}
