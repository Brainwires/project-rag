//! Glob pattern matching utilities for path filtering

use globset::{Glob, GlobMatcher};

/// Check if a file path matches any of the given glob patterns
///
/// # Examples
///
/// ```
/// use project_rag::glob_utils::matches_any_pattern;
///
/// let patterns = vec!["lib/**".to_string(), "src/**/*.ts".to_string()];
/// assert!(matches_any_pattern("/project/lib/utils.ts", &patterns));
/// assert!(matches_any_pattern("/project/src/components/Button.ts", &patterns));
/// assert!(!matches_any_pattern("/project/tests/unit.rs", &patterns));
/// ```
pub fn matches_any_pattern(path: &str, patterns: &[String]) -> bool {
    if patterns.is_empty() {
        return true; // No patterns means match everything
    }

    patterns.iter().any(|pattern| {
        // Try to compile the glob pattern
        match Glob::new(pattern) {
            Ok(glob) => {
                let matcher = glob.compile_matcher();

                // Try matching against the full path
                if matcher.is_match(path) {
                    return true;
                }

                // Try without leading slash
                let path_no_slash = path.trim_start_matches('/');
                if matcher.is_match(path_no_slash) {
                    return true;
                }

                // For patterns like "lib/**", also try matching against path suffixes
                // This handles cases like "/absolute/path/to/lib/file.ts" matching "lib/**"
                if pattern.contains("**") || pattern.contains('*') {
                    // Split path into components and try matching from each component
                    let path_parts: Vec<&str> = path.split('/').collect();
                    for i in 0..path_parts.len() {
                        let suffix = path_parts[i..].join("/");
                        if matcher.is_match(&suffix) {
                            return true;
                        }
                    }
                }

                false
            }
            Err(e) => {
                // If glob compilation fails, fall back to simple substring matching
                tracing::warn!(
                    "Invalid glob pattern '{}', falling back to substring match: {}",
                    pattern,
                    e
                );
                path.contains(pattern)
            }
        }
    })
}

/// Compile multiple glob patterns into matchers for efficient repeated matching
///
/// Returns None if any pattern fails to compile
pub fn compile_patterns(patterns: &[String]) -> Option<Vec<GlobMatcher>> {
    patterns
        .iter()
        .map(|pattern| {
            Glob::new(pattern)
                .map(|g| g.compile_matcher())
                .map_err(|e| {
                    tracing::warn!("Failed to compile glob pattern '{}': {}", pattern, e);
                    e
                })
                .ok()
        })
        .collect()
}

/// Check if a path matches any of the precompiled glob matchers
pub fn matches_any_matcher(path: &str, matchers: &[GlobMatcher]) -> bool {
    if matchers.is_empty() {
        return true;
    }

    matchers.iter().any(|matcher| {
        // Try matching against the full path
        if matcher.is_match(path) {
            return true;
        }

        // Try without leading slash
        let path_no_slash = path.trim_start_matches('/');
        if matcher.is_match(path_no_slash) {
            return true;
        }

        // Try matching against path suffixes for glob patterns
        let path_parts: Vec<&str> = path.split('/').collect();
        for i in 0..path_parts.len() {
            let suffix = path_parts[i..].join("/");
            if matcher.is_match(&suffix) {
                return true;
            }
        }

        false
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matches_directory_glob() {
        let patterns = vec!["lib/**".to_string()];

        assert!(matches_any_pattern("/project/lib/utils.ts", &patterns));
        assert!(matches_any_pattern("lib/nested/file.rs", &patterns));
        assert!(!matches_any_pattern("/project/src/main.rs", &patterns));
    }

    #[test]
    fn test_matches_extension_glob() {
        let patterns = vec!["**/*.ts".to_string()];

        assert!(matches_any_pattern("/project/src/main.ts", &patterns));
        assert!(matches_any_pattern("lib/utils.ts", &patterns));
        assert!(!matches_any_pattern("/project/src/main.rs", &patterns));
    }

    #[test]
    fn test_matches_multiple_patterns() {
        let patterns = vec!["lib/**".to_string(), "**/*.tsx".to_string()];

        assert!(matches_any_pattern("/project/lib/utils.ts", &patterns));
        assert!(matches_any_pattern("/project/src/Component.tsx", &patterns));
        assert!(!matches_any_pattern("/project/src/main.rs", &patterns));
    }

    #[test]
    fn test_matches_complex_glob() {
        // globset doesn't support brace expansion {ts,tsx}
        // Use separate patterns instead
        let patterns = vec!["src/components/**/*.ts".to_string()];

        assert!(matches_any_pattern(
            "/project/src/components/Button.ts",
            &patterns
        ));
        assert!(!matches_any_pattern("/project/lib/utils.ts", &patterns));
    }

    #[test]
    fn test_empty_patterns() {
        let patterns = vec![];
        assert!(matches_any_pattern("/any/path.rs", &patterns));
    }

    #[test]
    fn test_invalid_pattern_fallback() {
        let patterns = vec!["[invalid".to_string()];

        // Should fall back to substring matching
        assert!(matches_any_pattern("/path/[invalid/file.rs", &patterns));
        assert!(!matches_any_pattern("/path/valid/file.rs", &patterns));
    }

    #[test]
    fn test_compile_patterns() {
        let patterns = vec!["lib/**".to_string(), "**/*.rs".to_string()];
        let matchers = compile_patterns(&patterns);

        assert!(matchers.is_some());
        let matchers = matchers.unwrap();
        assert_eq!(matchers.len(), 2);

        assert!(matches_any_matcher("/project/lib/utils.ts", &matchers));
        assert!(matches_any_matcher("/project/src/main.rs", &matchers));
        assert!(!matches_any_matcher("/project/test.txt", &matchers));
    }

    #[test]
    fn test_compile_invalid_patterns() {
        let patterns = vec!["lib/**".to_string(), "[invalid".to_string()];
        let matchers = compile_patterns(&patterns);

        // Should return None if any pattern fails to compile
        assert!(matchers.is_none());
    }

    #[test]
    fn test_matches_without_leading_slash() {
        let patterns = vec!["lib/**".to_string()];

        // Should match with or without leading slash
        assert!(matches_any_pattern("lib/file.rs", &patterns));
        assert!(matches_any_pattern("/lib/file.rs", &patterns));
    }

    #[test]
    fn test_specific_file_pattern() {
        let patterns = vec!["**/test.rs".to_string()];

        assert!(matches_any_pattern("/project/src/test.rs", &patterns));
        assert!(matches_any_pattern("test.rs", &patterns));
        assert!(!matches_any_pattern("/project/src/main.rs", &patterns));
    }
}
