//! Built-in exclusions for files that commonly hold secrets.
//!
//! These are applied by the file walker regardless of the user's include/exclude
//! patterns, so credentials never end up in the vector store or BM25 index.

use globset::{GlobBuilder, GlobSet, GlobSetBuilder};
use std::path::Path;
use std::sync::OnceLock;

/// Glob patterns, matched against the path relative to the walk root, that are always
/// excluded from indexing.
pub const DEFAULT_EXCLUDE_GLOBS: &[&str] = &[
    "**/.env",
    "**/.env.*",
    "**/*.pem",
    "**/*.key",
    "**/id_rsa*",
    "**/id_ed25519*",
    "**/.aws/**",
    "**/.ssh/**",
    "**/*.p12",
    "**/*.pfx",
    "**/.npmrc",
    "**/.netrc",
];

fn glob_set() -> &'static GlobSet {
    static SET: OnceLock<GlobSet> = OnceLock::new();
    SET.get_or_init(|| {
        let mut builder = GlobSetBuilder::new();
        for pattern in DEFAULT_EXCLUDE_GLOBS {
            let glob = GlobBuilder::new(pattern)
                .literal_separator(true)
                .build()
                .expect("built-in exclude pattern must be a valid glob");
            builder.add(glob);
        }
        builder
            .build()
            .expect("built-in exclude glob set must compile")
    })
}

/// Returns true if `relative_path` (relative to the walk root) is a file that must never
/// be indexed.
pub fn is_default_excluded(relative_path: &Path) -> bool {
    glob_set().is_match(relative_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn excluded(path: &str) -> bool {
        is_default_excluded(Path::new(path))
    }

    #[test]
    fn test_env_files() {
        assert!(excluded(".env"));
        assert!(excluded(".env.local"));
        assert!(excluded(".env.production"));
        assert!(excluded("backend/.env"));
        assert!(excluded("apps/web/.env.development"));
    }

    #[test]
    fn test_key_material() {
        assert!(excluded("server.pem"));
        assert!(excluded("certs/ca.pem"));
        assert!(excluded("private.key"));
        assert!(excluded("tls/server.key"));
        assert!(excluded("keystore.p12"));
        assert!(excluded("cert.pfx"));
        assert!(excluded("id_rsa"));
        assert!(excluded("id_rsa.pub"));
        assert!(excluded("keys/id_ed25519"));
        assert!(excluded("keys/id_ed25519.pub"));
    }

    #[test]
    fn test_credential_directories_and_rc_files() {
        assert!(excluded(".aws/credentials"));
        assert!(excluded("home/.aws/config"));
        assert!(excluded(".ssh/config"));
        assert!(excluded(".ssh/known_hosts"));
        assert!(excluded(".npmrc"));
        assert!(excluded("packages/app/.npmrc"));
        assert!(excluded(".netrc"));
    }

    #[test]
    fn test_ordinary_files_are_not_excluded() {
        assert!(!excluded("src/main.rs"));
        assert!(!excluded("README.md"));
        assert!(!excluded("src/environment.rs"));
        assert!(!excluded("env.example.toml"));
        assert!(!excluded("docs/keys.md"));
        assert!(!excluded("src/keyboard.rs"));
        assert!(!excluded("awesome/config.toml"));
        assert!(!excluded("Cargo.toml"));
    }
}
