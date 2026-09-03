//! Request/response types for the find_unused tool: report import bindings and
//! symbol definitions that nothing references.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::build_config::ConfigurationState;
use crate::relations::SymbolKind;

fn default_check() -> String {
    "all".to_string()
}

fn default_limit() -> usize {
    100
}

fn default_max_file_size() -> usize {
    1_048_576
}

/// Request to scan a file or directory for unused imports and symbols
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FindUnusedRequest {
    /// File or directory to scan. Directories are walked with .gitignore support.
    pub path: String,
    /// Optional project name (used when probing the index for cross-file usage)
    #[serde(default)]
    pub project: Option<String>,
    /// What to check: "imports" (unused import/use/include bindings),
    /// "symbols" (definitions nothing references), or "all" (default)
    #[serde(default = "default_check")]
    pub check: String,
    /// Maximum number of candidates to return (default 100)
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Maximum file size in bytes to scan (default 1 MB)
    #[serde(default = "default_max_file_size")]
    pub max_file_size: usize,
    /// Restrict analysis to these discovered/explicit config IDs. Empty means all.
    #[serde(default)]
    pub configurations: Vec<String>,
}

impl FindUnusedRequest {
    /// Validate the find unused request
    pub fn validate(&self) -> Result<(), String> {
        if self.path.is_empty() {
            return Err("path cannot be empty".to_string());
        }
        if !matches!(self.check.as_str(), "imports" | "symbols" | "all") {
            return Err(format!(
                "check must be one of 'imports', 'symbols', 'all' (got '{}')",
                self.check
            ));
        }
        if self.limit == 0 {
            return Err("limit must be >= 1".to_string());
        }
        if self
            .configurations
            .iter()
            .any(|configuration| configuration.trim().is_empty())
        {
            return Err("configurations cannot contain an empty config_id".to_string());
        }
        Ok(())
    }

    /// Whether unused imports should be checked
    pub fn check_imports(&self) -> bool {
        matches!(self.check.as_str(), "imports" | "all")
    }

    /// Whether unused symbols should be checked
    pub fn check_symbols(&self) -> bool {
        matches!(self.check.as_str(), "symbols" | "all")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum UnusedStatus {
    Referenced,
    UnusedInAnalyzedConfigurations,
    Inconclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisCompleteness {
    Complete,
    Partial,
}

/// A definition or import binding that appears to be unused
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct UnusedCandidate {
    /// File path (relative to the scanned root) containing the candidate
    pub file_path: String,
    /// The unused name (import binding, function, class, ...)
    pub name: String,
    /// Symbol kind; `import` for unused imports
    pub kind: SymbolKind,
    /// Line the definition or import starts on (1-based)
    pub line: usize,
    /// How much to trust this finding: "high", "medium", or "low".
    /// This tool is text-based: dynamic dispatch, macros, reflection and
    /// framework wiring are invisible to it, so treat candidates as leads to
    /// verify, not facts. Never delete code from this list automatically.
    pub confidence: String,
    /// Why this was flagged
    pub reason: String,
    /// The definition/import line, for eyeballing without opening the file
    pub signature: String,
    /// What the analysis actually searched for before flagging: the identifier
    /// probed, or the header symbols probed for a C/C++ include
    #[serde(default)]
    pub probe: String,
    pub status: UnusedStatus,
    #[serde(default)]
    pub analyzed_configurations: Vec<String>,
    pub analysis_completeness: AnalysisCompleteness,
    #[serde(default)]
    pub configuration_states: Vec<ConfigurationState>,
    #[serde(default)]
    pub unresolved_dependency_kinds: Vec<String>,
    #[serde(default)]
    pub limitations: Vec<String>,
    /// Never infer deletion safety from a textual unused scan.
    pub safe_for_destructive_edit: bool,
}

/// An import binding that could not be verified and was therefore NOT flagged
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct UnverifiableImport {
    /// File path (relative to the scanned root) containing the import
    pub file_path: String,
    /// The imported name as written
    pub name: String,
    /// Why it could not be verified
    pub reason: String,
}

/// Why symbol definitions were NOT flagged, by rejection cause. Large
/// `used_elsewhere` counts on C/C++ scans are expected: header declarations
/// count as usage, so paired .h/.cpp symbols are rejected there.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct SymbolRejections {
    /// Name appears in its own file outside import lines and its own definition
    pub used_here: usize,
    /// Name appears in another scanned file outside import lines and
    /// same-name definition spans
    pub used_elsewhere: usize,
    /// Definition kind/shape is not checkable (imports, entry points, ...)
    pub ineligible: usize,
    /// Cross-index probes that errored; those symbols were conservatively
    /// treated as used
    pub probe_errors: usize,
}

/// Response from find_unused
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FindUnusedResponse {
    /// The root that was scanned (normalized)
    pub scanned_root: String,
    /// Number of files scanned
    pub files_scanned: usize,
    /// Total definitions extracted and considered
    pub definitions_checked: usize,
    /// Unused candidates found, ordered by file then line
    pub candidates: Vec<UnusedCandidate>,
    /// Total candidates found (may exceed candidates.len() when truncated)
    pub total_candidates: usize,
    /// Import bindings that could not be verified and were therefore NOT
    /// reported: system/`<...>` includes, unresolvable local headers, C#
    /// namespace usings, Swift module imports, and files whose language is
    /// unknown. Always the authoritative count.
    pub unverifiable_imports: usize,
    /// Per-import detail for the unverifiable count, capped at 200 entries
    #[serde(default)]
    pub unverifiable_import_details: Vec<UnverifiableImport>,
    /// Why symbol definitions were rejected rather than flagged
    #[serde(default)]
    pub symbol_rejections: SymbolRejections,
    /// Definition nodes the extractor recognised but could not name in the
    /// scanned files; those symbols were never considered at all
    #[serde(default)]
    pub skipped_definitions: usize,
    /// True if candidates were cut to `limit`
    pub truncated: bool,
    /// True if the cross-index probe budget ran out; symbols past the budget
    /// were conservatively treated as used
    pub probes_exhausted: bool,
    #[serde(default)]
    pub analyzed_configurations: Vec<String>,
    pub analysis_completeness: AnalysisCompleteness,
    #[serde(default)]
    pub unresolved_dependency_kinds: Vec<String>,
    #[serde(default)]
    pub limitations: Vec<String>,
    pub safe_for_destructive_edit: bool,
    /// Precision class of the method (text-based AST extraction + whole-word
    /// matching), not a per-run quality measure; see symbol_rejections,
    /// unverifiable_import_details and skipped_definitions for this run's
    /// completeness
    pub precision: String,
    /// Time taken in milliseconds
    pub duration_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_defaults_when_omitted() {
        let json = r#"{"path":"src"}"#;
        let req: FindUnusedRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.check, "all");
        assert_eq!(req.limit, 100);
        assert_eq!(req.max_file_size, 1_048_576);
        assert!(req.configurations.is_empty());
        assert!(req.project.is_none());
        assert!(req.check_imports());
        assert!(req.check_symbols());
    }

    #[test]
    fn test_request_validate_empty_path() {
        let req = FindUnusedRequest {
            path: String::new(),
            project: None,
            check: "all".to_string(),
            limit: 100,
            max_file_size: 1_048_576,
            configurations: Vec::new(),
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_request_validate_bad_check() {
        let req = FindUnusedRequest {
            path: "src".to_string(),
            project: None,
            check: "everything".to_string(),
            limit: 100,
            max_file_size: 1_048_576,
            configurations: Vec::new(),
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_request_validate_zero_limit() {
        let req = FindUnusedRequest {
            path: "src".to_string(),
            project: None,
            check: "imports".to_string(),
            limit: 0,
            max_file_size: 1_048_576,
            configurations: Vec::new(),
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_request_check_flags() {
        let mut req = FindUnusedRequest {
            path: "src".to_string(),
            project: None,
            check: "imports".to_string(),
            limit: 10,
            max_file_size: 1_048_576,
            configurations: Vec::new(),
        };
        assert!(req.validate().is_ok());
        assert!(req.check_imports());
        assert!(!req.check_symbols());

        req.check = "symbols".to_string();
        assert!(!req.check_imports());
        assert!(req.check_symbols());
    }

    #[test]
    fn test_response_serde_roundtrip() {
        let resp = FindUnusedResponse {
            scanned_root: "/proj".to_string(),
            files_scanned: 10,
            definitions_checked: 50,
            candidates: vec![UnusedCandidate {
                file_path: "src/lib.rs".to_string(),
                name: "HashMap".to_string(),
                kind: SymbolKind::Import,
                line: 3,
                confidence: "medium".to_string(),
                reason: "imported name is never referenced in this file".to_string(),
                signature: "use std::collections::HashMap;".to_string(),
                probe: "whole-word search for 'HashMap'".to_string(),
                status: UnusedStatus::Inconclusive,
                analyzed_configurations: Vec::new(),
                analysis_completeness: AnalysisCompleteness::Partial,
                configuration_states: Vec::new(),
                unresolved_dependency_kinds: vec!["build_configuration".to_string()],
                limitations: vec!["no build configuration".to_string()],
                safe_for_destructive_edit: false,
            }],
            total_candidates: 1,
            unverifiable_imports: 2,
            unverifiable_import_details: vec![UnverifiableImport {
                file_path: "src/main.c".to_string(),
                name: "stdio.h".to_string(),
                reason: "system include (<...>); not resolvable".to_string(),
            }],
            symbol_rejections: SymbolRejections {
                used_here: 4,
                used_elsewhere: 3,
                ineligible: 2,
                probe_errors: 1,
            },
            skipped_definitions: 1,
            truncated: false,
            probes_exhausted: false,
            analyzed_configurations: Vec::new(),
            analysis_completeness: AnalysisCompleteness::Partial,
            unresolved_dependency_kinds: vec!["build_configuration".to_string()],
            limitations: vec!["no build configuration".to_string()],
            safe_for_destructive_edit: false,
            precision: "medium".to_string(),
            duration_ms: 12,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: FindUnusedResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.candidates.len(), 1);
        assert_eq!(deserialized.candidates[0].name, "HashMap");
        assert_eq!(deserialized.candidates[0].kind, SymbolKind::Import);
        assert_eq!(
            deserialized.candidates[0].probe,
            "whole-word search for 'HashMap'"
        );
        assert_eq!(deserialized.unverifiable_imports, 2);
        assert_eq!(deserialized.unverifiable_import_details.len(), 1);
        assert_eq!(deserialized.unverifiable_import_details[0].name, "stdio.h");
        assert_eq!(deserialized.symbol_rejections.used_elsewhere, 3);
        assert_eq!(deserialized.skipped_definitions, 1);
    }
}
