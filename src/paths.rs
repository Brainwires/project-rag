/// Centralized platform-specific path computation
///
/// Provides consistent path handling across Windows, macOS, and Linux following
/// XDG Base Directory specification on Unix-like systems.
use std::path::PathBuf;

/// Platform-agnostic path utilities
pub struct PlatformPaths;

impl PlatformPaths {
    /// Get the appropriate data directory for the current platform
    ///
    /// - Windows: %LOCALAPPDATA%
    /// - macOS: ~/Library/Application Support
    /// - Linux/Unix: $XDG_DATA_HOME or ~/.local/share
    pub fn data_dir() -> PathBuf {
        if cfg!(target_os = "windows") {
            std::env::var("LOCALAPPDATA")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("."))
        } else if cfg!(target_os = "macos") {
            std::env::var("HOME")
                .map(|home| PathBuf::from(home).join("Library/Application Support"))
                .unwrap_or_else(|_| PathBuf::from("."))
        } else {
            // Linux/Unix - follow XDG Base Directory specification
            std::env::var("XDG_DATA_HOME")
                .map(PathBuf::from)
                .or_else(|_| {
                    std::env::var("HOME").map(|home| PathBuf::from(home).join(".local/share"))
                })
                .unwrap_or_else(|_| PathBuf::from("."))
        }
    }

    /// Get the appropriate cache directory for the current platform
    ///
    /// - Windows: %LOCALAPPDATA%
    /// - macOS: ~/Library/Caches
    /// - Linux/Unix: $XDG_CACHE_HOME or ~/.cache
    pub fn cache_dir() -> PathBuf {
        if cfg!(target_os = "windows") {
            std::env::var("LOCALAPPDATA")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("."))
        } else if cfg!(target_os = "macos") {
            std::env::var("HOME")
                .map(|home| PathBuf::from(home).join("Library/Caches"))
                .unwrap_or_else(|_| PathBuf::from("."))
        } else {
            // Linux/Unix - follow XDG Base Directory specification
            std::env::var("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .or_else(|_| std::env::var("HOME").map(|home| PathBuf::from(home).join(".cache")))
                .unwrap_or_else(|_| PathBuf::from("."))
        }
    }

    /// Get the appropriate config directory for the current platform
    ///
    /// - Windows: %APPDATA%
    /// - macOS: ~/Library/Application Support
    /// - Linux/Unix: $XDG_CONFIG_HOME or ~/.config
    pub fn config_dir() -> PathBuf {
        if cfg!(target_os = "windows") {
            std::env::var("APPDATA")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("."))
        } else if cfg!(target_os = "macos") {
            std::env::var("HOME")
                .map(|home| PathBuf::from(home).join("Library/Application Support"))
                .unwrap_or_else(|_| PathBuf::from("."))
        } else {
            // Linux/Unix - follow XDG Base Directory specification
            std::env::var("XDG_CONFIG_HOME")
                .map(PathBuf::from)
                .or_else(|_| std::env::var("HOME").map(|home| PathBuf::from(home).join(".config")))
                .unwrap_or_else(|_| PathBuf::from("."))
        }
    }

    /// Get default project-specific data directory
    ///
    /// Returns: {data_dir}/brainwires
    pub fn project_data_dir() -> PathBuf {
        Self::data_dir().join("brainwires")
    }

    /// Get default project-specific cache directory
    ///
    /// Returns: {cache_dir}/brainwires
    pub fn project_cache_dir() -> PathBuf {
        Self::cache_dir().join("brainwires")
    }

    /// Get default project-specific config directory
    ///
    /// Returns: {config_dir}/brainwires
    pub fn project_config_dir() -> PathBuf {
        Self::config_dir().join("brainwires")
    }

    /// Get default LanceDB database path
    ///
    /// Returns: {data_dir}/brainwires/lancedb
    pub fn default_lancedb_path() -> PathBuf {
        Self::project_data_dir().join("lancedb")
    }

    /// Get default hash cache path
    ///
    /// Returns: {cache_dir}/brainwires/hash_cache.json
    pub fn default_hash_cache_path() -> PathBuf {
        Self::project_cache_dir().join("hash_cache.json")
    }

    /// Get default git cache path
    ///
    /// Returns: {cache_dir}/brainwires/git_cache.json
    pub fn default_git_cache_path() -> PathBuf {
        Self::project_cache_dir().join("git_cache.json")
    }

    /// Get default config file path
    ///
    /// Returns: {config_dir}/brainwires/config.toml
    pub fn default_config_path() -> PathBuf {
        Self::project_config_dir().join("config.toml")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_data_dir_not_empty() {
        let dir = PlatformPaths::data_dir();
        assert!(!dir.as_os_str().is_empty());
    }

    #[test]
    fn test_cache_dir_not_empty() {
        let dir = PlatformPaths::cache_dir();
        assert!(!dir.as_os_str().is_empty());
    }

    #[test]
    fn test_config_dir_not_empty() {
        let dir = PlatformPaths::config_dir();
        assert!(!dir.as_os_str().is_empty());
    }

    #[test]
    fn test_project_paths_contain_project_name() {
        let data_dir = PlatformPaths::project_data_dir();
        let cache_dir = PlatformPaths::project_cache_dir();
        let config_dir = PlatformPaths::project_config_dir();

        assert!(data_dir.to_string_lossy().contains("brainwires"));
        assert!(cache_dir.to_string_lossy().contains("brainwires"));
        assert!(config_dir.to_string_lossy().contains("brainwires"));
    }

    #[test]
    fn test_default_lancedb_path() {
        let path = PlatformPaths::default_lancedb_path();
        assert!(path.to_string_lossy().contains("brainwires"));
        assert!(path.to_string_lossy().contains("lancedb"));
    }

    #[test]
    fn test_default_hash_cache_path() {
        let path = PlatformPaths::default_hash_cache_path();
        assert!(path.to_string_lossy().contains("brainwires"));
        assert!(path.to_string_lossy().contains("hash_cache.json"));
    }

    #[test]
    fn test_default_git_cache_path() {
        let path = PlatformPaths::default_git_cache_path();
        assert!(path.to_string_lossy().contains("brainwires"));
        assert!(path.to_string_lossy().contains("git_cache.json"));
    }

    #[test]
    fn test_default_config_path() {
        let path = PlatformPaths::default_config_path();
        assert!(path.to_string_lossy().contains("brainwires"));
        assert!(path.to_string_lossy().contains("config.toml"));
    }

    #[test]
    fn test_paths_are_absolute_or_relative() {
        // Paths should either be absolute or fallback to "."
        let data_dir = PlatformPaths::data_dir();
        assert!(data_dir.is_absolute() || data_dir == PathBuf::from("."));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_data_dir_with_xdg_data_home() {
        // Test that XDG_DATA_HOME is respected
        let original = env::var("XDG_DATA_HOME").ok();
        unsafe {
            env::set_var("XDG_DATA_HOME", "/custom/data");
        }

        let dir = PlatformPaths::data_dir();
        assert_eq!(dir, PathBuf::from("/custom/data"));

        // Restore original value
        unsafe {
            match original {
                Some(val) => env::set_var("XDG_DATA_HOME", val),
                None => env::remove_var("XDG_DATA_HOME"),
            }
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_data_dir_fallback_to_home() {
        // Test fallback to HOME/.local/share when XDG_DATA_HOME is not set
        let xdg_original = env::var("XDG_DATA_HOME").ok();
        let home_original = env::var("HOME").ok();

        unsafe {
            env::remove_var("XDG_DATA_HOME");
            env::set_var("HOME", "/home/testuser");
        }

        let dir = PlatformPaths::data_dir();
        assert_eq!(dir, PathBuf::from("/home/testuser/.local/share"));

        // Restore original values
        unsafe {
            match xdg_original {
                Some(val) => env::set_var("XDG_DATA_HOME", val),
                None => env::remove_var("XDG_DATA_HOME"),
            }
            match home_original {
                Some(val) => env::set_var("HOME", val),
                None => env::remove_var("HOME"),
            }
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_cache_dir_with_xdg_cache_home() {
        // Test that XDG_CACHE_HOME is respected
        let original = env::var("XDG_CACHE_HOME").ok();
        unsafe {
            env::set_var("XDG_CACHE_HOME", "/custom/cache");
        }

        let dir = PlatformPaths::cache_dir();
        assert_eq!(dir, PathBuf::from("/custom/cache"));

        // Restore original value
        unsafe {
            match original {
                Some(val) => env::set_var("XDG_CACHE_HOME", val),
                None => env::remove_var("XDG_CACHE_HOME"),
            }
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_cache_dir_fallback_to_home() {
        // Test fallback to HOME/.cache when XDG_CACHE_HOME is not set
        let xdg_original = env::var("XDG_CACHE_HOME").ok();
        let home_original = env::var("HOME").ok();

        unsafe {
            env::remove_var("XDG_CACHE_HOME");
            env::set_var("HOME", "/home/testuser");
        }

        let dir = PlatformPaths::cache_dir();
        assert_eq!(dir, PathBuf::from("/home/testuser/.cache"));

        // Restore original values
        unsafe {
            match xdg_original {
                Some(val) => env::set_var("XDG_CACHE_HOME", val),
                None => env::remove_var("XDG_CACHE_HOME"),
            }
            match home_original {
                Some(val) => env::set_var("HOME", val),
                None => env::remove_var("HOME"),
            }
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_config_dir_with_xdg_config_home() {
        // Test that XDG_CONFIG_HOME is respected
        let original = env::var("XDG_CONFIG_HOME").ok();
        unsafe {
            env::set_var("XDG_CONFIG_HOME", "/custom/config");
        }

        let dir = PlatformPaths::config_dir();
        assert_eq!(dir, PathBuf::from("/custom/config"));

        // Restore original value
        unsafe {
            match original {
                Some(val) => env::set_var("XDG_CONFIG_HOME", val),
                None => env::remove_var("XDG_CONFIG_HOME"),
            }
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_config_dir_fallback_to_home() {
        // Test fallback to HOME/.config when XDG_CONFIG_HOME is not set
        let xdg_original = env::var("XDG_CONFIG_HOME").ok();
        let home_original = env::var("HOME").ok();

        unsafe {
            env::remove_var("XDG_CONFIG_HOME");
            env::set_var("HOME", "/home/testuser");
        }

        let dir = PlatformPaths::config_dir();
        assert_eq!(dir, PathBuf::from("/home/testuser/.config"));

        // Restore original values
        unsafe {
            match xdg_original {
                Some(val) => env::set_var("XDG_CONFIG_HOME", val),
                None => env::remove_var("XDG_CONFIG_HOME"),
            }
            match home_original {
                Some(val) => env::set_var("HOME", val),
                None => env::remove_var("HOME"),
            }
        }
    }

    #[test]
    fn test_all_project_dirs_are_subdirectories() {
        // Verify that project-specific dirs are subdirectories of base dirs
        let data_dir = PlatformPaths::data_dir();
        let project_data = PlatformPaths::project_data_dir();

        assert!(
            project_data.starts_with(&data_dir) || data_dir == PathBuf::from("."),
            "project_data_dir should be subdirectory of data_dir"
        );

        let cache_dir = PlatformPaths::cache_dir();
        let project_cache = PlatformPaths::project_cache_dir();

        assert!(
            project_cache.starts_with(&cache_dir) || cache_dir == PathBuf::from("."),
            "project_cache_dir should be subdirectory of cache_dir"
        );
    }

    #[test]
    fn test_specific_file_paths() {
        // Test that specific file paths include expected components
        let lancedb_path = PlatformPaths::default_lancedb_path();
        let hash_cache_path = PlatformPaths::default_hash_cache_path();
        let git_cache_path = PlatformPaths::default_git_cache_path();
        let config_path = PlatformPaths::default_config_path();

        // All should contain project name
        for path in [&lancedb_path, &hash_cache_path, &git_cache_path, &config_path] {
            assert!(
                path.to_string_lossy().contains("brainwires"),
                "Path {:?} should contain 'brainwires'",
                path
            );
        }

        // Specific components
        assert!(lancedb_path.ends_with("lancedb"));
        assert!(hash_cache_path.ends_with("hash_cache.json"));
        assert!(git_cache_path.ends_with("git_cache.json"));
        assert!(config_path.ends_with("config.toml"));
    }
}
