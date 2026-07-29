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

/// Largest index `<= max` that lies on a UTF-8 character boundary in `s`.
///
/// `String::truncate` and `&s[..n]` take **byte** offsets and panic unless the
/// offset falls on a character boundary. Commit messages and diffs routinely
/// carry non-ASCII text, so every byte-length cap applied to them has to be
/// floored through here first.
pub fn floor_char_boundary(s: &str, max: usize) -> usize {
    if max >= s.len() {
        return s.len();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    end
}

#[cfg(test)]
mod tests {
    use super::floor_char_boundary;

    #[test]
    fn floors_offsets_onto_character_boundaries() {
        // Each Cyrillic character below is two bytes in UTF-8.
        let s = "аб";
        assert_eq!(s.len(), 4, "test string should be 4 bytes");
        assert_eq!(floor_char_boundary(s, 0), 0);
        assert_eq!(floor_char_boundary(s, 1), 0, "offset 1 splits the first char");
        assert_eq!(floor_char_boundary(s, 2), 2);
        assert_eq!(floor_char_boundary(s, 3), 2, "offset 3 splits the second char");
        assert_eq!(floor_char_boundary(s, 4), 4);
    }

    #[test]
    fn clamps_offsets_past_the_end() {
        assert_eq!(floor_char_boundary("abc", 99), 3);
        assert_eq!(floor_char_boundary("", 5), 0);
    }

    #[test]
    fn every_floored_offset_is_safe_to_truncate_at() {
        // Regression guard for the panic this helper exists to prevent:
        // String::truncate asserts is_char_boundary, so a raw byte cap
        // crashed on any text with multi-byte characters.
        let s = "программа mixed with ASCII — and more";
        for i in 0..=s.len() {
            let mut owned = s.to_string();
            let end = floor_char_boundary(s, i);
            owned.truncate(end);
            assert!(owned.len() <= i);
        }
    }
}
