//! UTF-8 safe string truncation helpers.
//!
//! Byte-indexed slicing (`&s[..n]`) and `String::truncate(n)` panic when `n` falls inside
//! a multi-byte character. These helpers round `n` down to the nearest char boundary.

/// Largest byte index `<= max_bytes` that lies on a char boundary of `s`.
///
/// Equivalent to the (still unstable) `str::floor_char_boundary`.
pub fn floor_char_boundary(s: &str, max_bytes: usize) -> usize {
    if max_bytes >= s.len() {
        return s.len();
    }
    let mut index = max_bytes;
    // index 0 is always a boundary, so this terminates
    while !s.is_char_boundary(index) {
        index -= 1;
    }
    index
}

/// Longest prefix of `s` that is at most `max_bytes` long without splitting a character.
pub fn truncate_str(s: &str, max_bytes: usize) -> &str {
    &s[..floor_char_boundary(s, max_bytes)]
}

/// Truncate `s` in place to at most `max_bytes` without splitting a character.
pub fn truncate_string(s: &mut String, max_bytes: usize) {
    let index = floor_char_boundary(s, max_bytes);
    s.truncate(index);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ascii_is_cut_exactly() {
        assert_eq!(truncate_str("hello world", 5), "hello");
        assert_eq!(floor_char_boundary("hello", 3), 3);
    }

    #[test]
    fn test_limit_beyond_length_returns_whole_string() {
        assert_eq!(truncate_str("héllo", 100), "héllo");
        assert_eq!(floor_char_boundary("héllo", 6), 6);
        assert_eq!(floor_char_boundary("", 10), 0);
    }

    #[test]
    fn test_zero_limit() {
        assert_eq!(truncate_str("héllo", 0), "");
    }

    #[test]
    fn test_two_byte_char_at_boundary() {
        // "héllo" is h(1) é(2) l l o: byte 2 is inside 'é'
        assert_eq!(truncate_str("héllo", 2), "h");
        assert_eq!(truncate_str("héllo", 3), "hé");
        assert_eq!(truncate_str("héllo", 1), "h");
    }

    #[test]
    fn test_four_byte_char_at_boundary() {
        // "a😀b" is a(1) 😀(4) b(1)
        let s = "a😀b";
        assert_eq!(truncate_str(s, 2), "a");
        assert_eq!(truncate_str(s, 3), "a");
        assert_eq!(truncate_str(s, 4), "a");
        assert_eq!(truncate_str(s, 5), "a😀");
        assert_eq!(truncate_str(s, 6), "a😀b");
    }

    #[test]
    fn test_truncate_string_in_place() {
        let mut s = "日本語テキスト".to_string(); // 3 bytes per char
        truncate_string(&mut s, 7);
        assert_eq!(s, "日本");
        truncate_string(&mut s, 100);
        assert_eq!(s, "日本");
    }

    #[test]
    fn test_every_cut_point_is_valid_utf8() {
        let s = "aé😀日b";
        for n in 0..=s.len() {
            let t = truncate_str(s, n);
            assert!(t.len() <= n);
            assert!(s.starts_with(t));
        }
    }
}
