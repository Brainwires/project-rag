//! Canonical project-root-relative file identity.
//!
//! Filesystem paths are accepted as transport aliases only. Once resolved, the
//! persistent and user-facing identity is always a `/`-separated path relative
//! to one explicit, canonical project root.

use anyhow::{Context, Result};
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedProjectPath {
    /// Canonical absolute path used only for filesystem access.
    pub absolute: PathBuf,
    /// Canonical project-relative identity used for storage and MCP output.
    pub relative: String,
}

#[derive(Debug, Clone)]
pub struct ProjectPathResolver {
    root: PathBuf,
}

impl ProjectPathResolver {
    pub fn new(project_root: impl AsRef<Path>) -> Result<Self> {
        let root_input =
            normalize_transport_path(project_root.as_ref().as_os_str().to_string_lossy().as_ref());
        let root = std::fs::canonicalize(&root_input).with_context(|| {
            format!(
                "Failed to canonicalize project root: {}",
                project_root.as_ref().display()
            )
        })?;
        if !root.is_dir() {
            anyhow::bail!("Project root is not a directory: {}", root.display());
        }
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve an existing input path and reject symlink/junction escapes.
    pub fn resolve_existing(&self, input: &str) -> Result<ResolvedProjectPath> {
        let candidate = self.candidate(input);
        let absolute = std::fs::canonicalize(&candidate)
            .with_context(|| format!("Failed to canonicalize path: {}", input))?;
        self.finish(absolute, input)
    }

    /// Resolve a not-yet-existing file through its canonical existing parent.
    pub fn resolve_for_write(&self, input: &str) -> Result<ResolvedProjectPath> {
        let candidate = self.candidate(input);
        if candidate.exists() {
            return self.resolve_existing(input);
        }

        let parent = candidate
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Invalid file path: {}", input))?;
        let canonical_parent = std::fs::canonicalize(parent)
            .with_context(|| format!("Parent directory does not exist: {}", parent.display()))?;
        let file_name = candidate
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("Invalid file path: {}", input))?;
        self.finish(canonical_parent.join(file_name), input)
    }

    fn candidate(&self, input: &str) -> PathBuf {
        let normalized = normalize_transport_path(input);
        if normalized.is_absolute() {
            normalized
        } else {
            self.root.join(normalized)
        }
    }

    fn finish(&self, absolute: PathBuf, original: &str) -> Result<ResolvedProjectPath> {
        let relative = relative_to_root(&self.root, &absolute).ok_or_else(|| {
            anyhow::anyhow!(
                "'{}' resolves outside project root '{}'",
                original,
                self.root.display()
            )
        })?;

        if relative.as_os_str().is_empty() {
            anyhow::bail!(
                "Expected a project file path, got project root: {}",
                original
            );
        }

        let relative = relative
            .components()
            .filter_map(|component| match component {
                Component::Normal(value) => Some(value.to_string_lossy().into_owned()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("/");

        if relative.is_empty() {
            anyhow::bail!(
                "Path has no canonical project-relative identity: {}",
                original
            );
        }

        Ok(ResolvedProjectPath { absolute, relative })
    }
}

/// Strip transport-only Windows extended-length prefixes and accept either
/// slash style on every host. The filesystem still decides case semantics.
pub fn normalize_transport_path(input: &str) -> PathBuf {
    let without_prefix = input
        .strip_prefix("\\\\?\\UNC\\")
        .map(|rest| format!("\\\\{}", rest))
        .or_else(|| input.strip_prefix("\\\\?\\").map(str::to_string))
        .unwrap_or_else(|| input.to_string());

    #[cfg(windows)]
    let normalized = without_prefix.replace('/', "\\");
    #[cfg(not(windows))]
    let normalized = without_prefix.replace('\\', "/");

    PathBuf::from(normalized)
}

fn relative_to_root(root: &Path, path: &Path) -> Option<PathBuf> {
    if let Ok(relative) = path.strip_prefix(root) {
        return Some(relative.to_path_buf());
    }

    // Windows filesystems are normally case-insensitive, while Path::strip_prefix
    // compares spelling. Compare components using host semantics, but preserve the
    // canonical path's spelling in the returned identity.
    #[cfg(windows)]
    {
        let root_components: Vec<_> = root.components().collect();
        let path_components: Vec<_> = path.components().collect();
        if root_components.len() > path_components.len() {
            return None;
        }
        let matches = root_components.iter().zip(&path_components).all(|(a, b)| {
            a.as_os_str()
                .to_string_lossy()
                .eq_ignore_ascii_case(&b.as_os_str().to_string_lossy())
        });
        if matches {
            return Some(path_components[root_components.len()..].iter().collect());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn relative_and_absolute_aliases_have_one_identity() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir(temp.path().join("src")).unwrap();
        let file = temp.path().join("src").join("a.cpp");
        std::fs::write(&file, "int a;").unwrap();
        let resolver = ProjectPathResolver::new(temp.path()).unwrap();

        let relative = resolver.resolve_existing(r".\src\a.cpp").unwrap();
        let absolute = resolver
            .resolve_existing(&file.to_string_lossy().replace('\\', "/"))
            .unwrap();

        assert_eq!(relative.relative, "src/a.cpp");
        assert_eq!(relative, absolute);
    }

    #[test]
    fn dot_dot_escape_is_rejected() {
        let parent = TempDir::new().unwrap();
        let root = parent.path().join("project");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(parent.path().join("outside.rs"), "fn outside() {}").unwrap();
        let resolver = ProjectPathResolver::new(&root).unwrap();

        assert!(resolver.resolve_existing("../outside.rs").is_err());
    }

    #[test]
    fn relocation_does_not_change_persistent_file_identity() {
        let temp = TempDir::new().unwrap();
        let first = temp.path().join("first");
        let moved = temp.path().join("moved");
        for root in [&first, &moved] {
            std::fs::create_dir_all(root.join("src")).unwrap();
            std::fs::write(root.join("src/lib.rs"), "pub fn value() {}").unwrap();
        }

        let first_identity = ProjectPathResolver::new(&first)
            .unwrap()
            .resolve_existing("src/lib.rs")
            .unwrap()
            .relative;
        let moved_identity = ProjectPathResolver::new(&moved)
            .unwrap()
            .resolve_existing("src/lib.rs")
            .unwrap()
            .relative;
        assert_eq!(first_identity, moved_identity);
    }

    #[cfg(windows)]
    #[test]
    fn case_aliases_resolve_to_one_canonical_identity() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir(temp.path().join("Src")).unwrap();
        std::fs::write(temp.path().join("Src/Module.rs"), "fn module() {}").unwrap();
        let resolver = ProjectPathResolver::new(temp.path()).unwrap();
        let canonical = resolver.resolve_existing("Src/Module.rs").unwrap();
        let alias = resolver.resolve_existing("src/module.rs").unwrap();
        assert_eq!(canonical.relative, alias.relative);
    }

    #[cfg(windows)]
    #[test]
    fn extended_length_and_ordinary_windows_paths_match() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("a.rs");
        std::fs::write(&file, "fn a() {}").unwrap();
        let resolver = ProjectPathResolver::new(temp.path()).unwrap();
        let ordinary = resolver.resolve_existing(&file.to_string_lossy()).unwrap();
        let extended = resolver
            .resolve_existing(&format!(r"\\?\{}", file.display()))
            .unwrap();
        assert_eq!(ordinary, extended);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escape_is_rejected() {
        use std::os::unix::fs::symlink;
        let parent = TempDir::new().unwrap();
        let root = parent.path().join("project");
        std::fs::create_dir(&root).unwrap();
        let outside = parent.path().join("outside.rs");
        std::fs::write(&outside, "fn outside() {}").unwrap();
        symlink(&outside, root.join("escape.rs")).unwrap();
        let resolver = ProjectPathResolver::new(&root).unwrap();
        assert!(resolver.resolve_existing("escape.rs").is_err());
    }
}
