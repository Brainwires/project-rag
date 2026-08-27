//! Reference finding via identifier matching.
//!
//! This module finds references to symbols by searching for identifier occurrences
//! that match known symbol names from the definition index.

use std::collections::HashMap;

use anyhow::Result;
use chrono::Utc;
use regex::Regex;

use crate::indexer::FileInfo;
use crate::relations::types::{
    Definition, DispatchKind, EvidenceKind, LocationRole, Reference, ReferenceCandidate,
    ReferenceKind, ResolutionStatus, SourceLocation, SymbolKind,
};

/// Finds references to symbols using text-based identifier matching.
pub struct ReferenceFinder {
    /// Regex for identifying valid identifier characters
    identifier_regex: Regex,
}

impl ReferenceFinder {
    /// Create a new reference finder
    pub fn new() -> Self {
        // Match word boundaries around identifiers
        Self {
            identifier_regex: Regex::new(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b").unwrap(),
        }
    }

    /// Find all references to known symbols in a file
    pub fn find_references(
        &self,
        file_info: &FileInfo,
        symbol_index: &HashMap<String, Vec<Definition>>,
    ) -> Result<Vec<Reference>> {
        let mut references = Vec::new();

        // Skip if no symbols to look for
        if symbol_index.is_empty() {
            return Ok(references);
        }

        let extension = file_info.extension.as_deref().unwrap_or("");
        let (language, language_name) =
            match super::symbol_extractor::get_language_for_extension(extension) {
                Some(value) => value,
                None => return Ok(references),
            };
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&language)?;
        let tree = parser
            .parse(&file_info.content, None)
            .ok_or_else(|| anyhow::anyhow!("tree-sitter returned no syntax tree"))?;
        let root = tree.root_node();
        let parser_name = format!("tree-sitter/{}", language_name.to_lowercase());

        let mut absolute_offset = 0usize;
        // Process each line, retaining byte offsets so syntax nodes can classify matches.
        for (line_num, line_with_newline) in file_info.content.split_inclusive('\n').enumerate() {
            let line = line_with_newline.trim_end_matches(['\r', '\n']);
            let line_number = line_num + 1; // 1-based

            // Find all identifier occurrences in this line
            for mat in self.identifier_regex.find_iter(line) {
                let name = mat.as_str();

                // Check if this identifier matches a known symbol
                if let Some(definitions) = symbol_index.get(name) {
                    // Definitions/declarations are emitted as explicit location rows by
                    // indexing; do not duplicate their name token as a heuristic match.
                    if self.is_definition_site(definitions, &file_info.relative_path, line_number)
                        && definitions.iter().any(|def| {
                            def.file_path() == file_info.relative_path
                                && def.location.start_line == line_number
                                && mat.start() >= def.location.start_col
                                && mat.end() <= def.location.end_col
                        })
                    {
                        continue;
                    }

                    let absolute_start = absolute_offset + mat.start();
                    let syntax_kind = root
                        .descendant_for_byte_range(absolute_start, absolute_start + name.len())
                        .map(|node| node.kind().to_string());
                    let reference_kind = self.determine_reference_kind(
                        line,
                        mat.start(),
                        name,
                        syntax_kind.as_deref(),
                    );

                    let mut unique = std::collections::BTreeMap::new();
                    for def in definitions
                        .iter()
                        .filter(|d| d.kind() != SymbolKind::Import)
                    {
                        unique.entry(def.to_storage_id()).or_insert(def);
                    }
                    if unique.is_empty() {
                        continue;
                    }
                    let candidates: Vec<ReferenceCandidate> = unique
                        .keys()
                        .map(|symbol_id| ReferenceCandidate {
                            symbol_id: symbol_id.clone(),
                            reason:
                                "identifier name matches; binding was not semantically resolved"
                                    .to_string(),
                        })
                        .collect();

                    let qualified_use = qualified_use_at(line, mat.start(), name);
                    let exact_qualified: Vec<_> = unique
                        .values()
                        .filter(|def| {
                            qualified_use.contains("::")
                                && def.symbol_id.qualified_name == qualified_use
                        })
                        .collect();
                    let (target_symbol_id, resolution_status, evidence_kind) =
                        if exact_qualified.len() == 1 {
                            (
                                exact_qualified[0].to_storage_id(),
                                ResolutionStatus::Resolved,
                                EvidenceKind::Syntactic,
                            )
                        } else if unique.len() > 1 {
                            (
                                String::new(),
                                ResolutionStatus::Ambiguous,
                                EvidenceKind::Heuristic,
                            )
                        } else {
                            (
                                String::new(),
                                ResolutionStatus::Unresolved,
                                EvidenceKind::Heuristic,
                            )
                        };
                    let source_symbol_id = definitions
                        .iter()
                        .filter(|def| {
                            def.file_path() == file_info.relative_path
                                && matches!(
                                    def.kind(),
                                    SymbolKind::Function
                                        | SymbolKind::Method
                                        | SymbolKind::Constructor
                                        | SymbolKind::Destructor
                                )
                                && line_number >= def.start_line()
                                && line_number <= def.end_line
                        })
                        .min_by_key(|def| def.end_line.saturating_sub(def.start_line()))
                        .map(Definition::to_storage_id);
                    let location = SourceLocation {
                        project_id: file_info.project.clone().unwrap_or_default(),
                        file_path: file_info.relative_path.clone(),
                        start_line: line_number,
                        start_col: mat.start(),
                        end_line: line_number,
                        end_col: mat.end(),
                        role: LocationRole::Reference,
                    };
                    references.push(Reference {
                        file_path: file_info.relative_path.clone(),
                        root_path: Some(file_info.root_path.clone()),
                        project: file_info.project.clone(),
                        start_line: line_number,
                        end_line: line_number,
                        start_col: mat.start(),
                        end_col: mat.end(),
                        location_id: location.to_storage_id(),
                        source_symbol_id,
                        target_symbol_id,
                        target_name: name.to_string(),
                        candidates,
                        reference_kind,
                        resolution_status,
                        evidence_kind,
                        dispatch_kind: if reference_kind == ReferenceKind::Call
                            && resolution_status == ResolutionStatus::Resolved
                        {
                            DispatchKind::Direct
                        } else {
                            DispatchKind::Unknown
                        },
                        language: language_name.clone(),
                        parser: parser_name.clone(),
                        indexed_at: Utc::now().timestamp(),
                    });
                }
            }
            absolute_offset += line_with_newline.len();
        }

        Ok(references)
    }

    /// Check if a line is likely a definition site
    fn is_definition_site(
        &self,
        definitions: &[Definition],
        file_path: &str,
        line_number: usize,
    ) -> bool {
        definitions.iter().any(|def| {
            def.file_path() == file_path
                && line_number >= def.start_line()
                && line_number <= def.end_line
        })
    }

    /// Determine the kind of reference based on context
    fn determine_reference_kind(
        &self,
        line: &str,
        position: usize,
        name: &str,
        syntax_kind: Option<&str>,
    ) -> ReferenceKind {
        if let Some(kind) = syntax_kind {
            if kind.contains("comment") {
                let trimmed = line.trim_start();
                return if trimmed.starts_with("///")
                    || trimmed.starts_with("//!")
                    || trimmed.starts_with("/**")
                    || trimmed.starts_with("*!")
                {
                    ReferenceKind::Documentation
                } else {
                    ReferenceKind::Comment
                };
            }
            if kind.contains("string") {
                return ReferenceKind::String;
            }
        }
        // Get text before the identifier
        let before = &line[..position];

        // Get text after the identifier (skip past the name itself)
        let after_end = position + name.len();
        let after_name = if after_end <= line.len() {
            &line[after_end..]
        } else {
            ""
        };

        let lower_line = line.to_lowercase();

        // Check for import patterns (highest priority)
        if lower_line.contains("import ")
            || lower_line.contains("from ")
            || lower_line.contains("require(")
            || lower_line.contains("use ")
        {
            return if lower_line.contains("#include") {
                ReferenceKind::Include
            } else {
                ReferenceKind::Import
            };
        }

        // Check for instantiation (before function call, since `new Foo()` looks like a call)
        if before.contains("new ") {
            return ReferenceKind::ConstructorCall;
        }

        // Check for inheritance patterns
        if before.contains("extends") || before.contains("implements") {
            return ReferenceKind::Inheritance;
        }

        // Check for function/method call pattern (identifier followed by parenthesis)
        if after_name.trim_start().starts_with('(') {
            return ReferenceKind::Call;
        }

        // Check for assignment (write)
        if after_name.trim_start().starts_with('=')
            && !after_name.trim_start().starts_with("==")
            && !after_name.trim_start().starts_with("=>")
        {
            return ReferenceKind::Write;
        }

        // Check for type reference patterns
        if before.contains(':') || before.contains("->") || before.contains('<') {
            return ReferenceKind::TypeReference;
        }

        // Default to read
        ReferenceKind::Read
    }
}

fn qualified_use_at(line: &str, position: usize, name: &str) -> String {
    let bytes = line.as_bytes();
    let mut start = position;
    while start > 0 {
        let byte = bytes[start - 1];
        if byte.is_ascii_alphanumeric() || byte == b'_' || byte == b':' {
            start -= 1;
        } else {
            break;
        }
    }
    let prefix = &line[start..position];
    format!("{}{}", prefix, name).trim_matches(':').to_string()
}

impl Default for ReferenceFinder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::relations::types::{SymbolId, SymbolKind, Visibility};
    use std::path::PathBuf;

    fn make_file_info(content: &str, path: &str) -> FileInfo {
        FileInfo {
            path: PathBuf::from(path),
            relative_path: path.to_string(),
            root_path: "/test".to_string(),
            project: None,
            extension: Some("rs".to_string()),
            language: Some("Rust".to_string()),
            content: content.to_string(),
            hash: "test_hash".to_string(),
        }
    }

    fn make_definition(name: &str, file_path: &str, start_line: usize) -> Definition {
        Definition {
            symbol_id: SymbolId::new(file_path, name, SymbolKind::Function, start_line, 0),
            location: SourceLocation {
                project_id: String::new(),
                file_path: file_path.to_string(),
                start_line,
                start_col: 3,
                end_line: start_line,
                end_col: 3 + name.len(),
                role: LocationRole::Definition,
            },
            root_path: Some("/test".to_string()),
            project: None,
            end_line: start_line + 5,
            end_col: 0,
            signature: format!("fn {}()", name),
            doc_comment: None,
            visibility: Visibility::Public,
            parent_id: None,
            parser: "tree-sitter/rust".to_string(),
            indexed_at: 0,
        }
    }

    #[test]
    fn test_find_function_call() {
        let source = r#"
fn main() {
    let result = greet("World");
}
"#;
        let file_info = make_file_info(source, "src/main.rs");

        let mut symbol_index = HashMap::new();
        symbol_index.insert(
            "greet".to_string(),
            vec![make_definition("greet", "src/lib.rs", 1)],
        );

        let finder = ReferenceFinder::new();
        let references = finder.find_references(&file_info, &symbol_index).unwrap();

        assert_eq!(references.len(), 1);
        assert_eq!(references[0].reference_kind, ReferenceKind::Call);
    }

    #[test]
    fn test_skip_definition_site() {
        let source = r#"
fn greet(name: &str) {
    println!("Hello, {}!", name);
}
"#;
        let file_info = make_file_info(source, "src/lib.rs");

        let mut symbol_index = HashMap::new();
        symbol_index.insert(
            "greet".to_string(),
            vec![make_definition("greet", "src/lib.rs", 2)], // Definition is on line 2
        );

        let finder = ReferenceFinder::new();
        let references = finder.find_references(&file_info, &symbol_index).unwrap();

        // Should not include the definition site as a reference
        assert!(references.is_empty());
    }

    #[test]
    fn test_detect_write() {
        let source = "counter = counter + 1";
        let file_info = make_file_info(source, "src/main.rs");

        let mut symbol_index = HashMap::new();
        symbol_index.insert(
            "counter".to_string(),
            vec![make_definition("counter", "src/lib.rs", 1)],
        );

        let finder = ReferenceFinder::new();
        let references = finder.find_references(&file_info, &symbol_index).unwrap();

        // First occurrence is a write, second is a read
        assert!(!references.is_empty());
        assert!(
            references
                .iter()
                .any(|r| r.reference_kind == ReferenceKind::Write)
        );
    }

    #[test]
    fn test_detect_import() {
        let source = "from mymodule import greet";
        let file_info = make_file_info(source, "src/main.py");

        let mut symbol_index = HashMap::new();
        symbol_index.insert(
            "greet".to_string(),
            vec![make_definition("greet", "src/mymodule.py", 1)],
        );

        let finder = ReferenceFinder::new();
        let references = finder.find_references(&file_info, &symbol_index).unwrap();

        assert!(!references.is_empty());
        assert!(
            references
                .iter()
                .any(|r| r.reference_kind == ReferenceKind::Import)
        );
    }

    #[test]
    fn test_detect_instantiation() {
        let source = "let person = new Person()";
        let file_info = make_file_info(source, "src/main.js");

        let mut symbol_index = HashMap::new();
        symbol_index.insert(
            "Person".to_string(),
            vec![make_definition("Person", "src/person.js", 1)],
        );

        let finder = ReferenceFinder::new();
        let references = finder.find_references(&file_info, &symbol_index).unwrap();

        assert!(!references.is_empty());
        assert!(
            references
                .iter()
                .any(|r| r.reference_kind == ReferenceKind::ConstructorCall)
        );
    }

    #[test]
    fn test_empty_symbol_index() {
        let source = "fn main() { greet(); }";
        let file_info = make_file_info(source, "src/main.rs");

        let symbol_index = HashMap::new();

        let finder = ReferenceFinder::new();
        let references = finder.find_references(&file_info, &symbol_index).unwrap();

        assert!(references.is_empty());
    }

    #[test]
    fn comments_documentation_and_strings_are_not_code_references() {
        let source =
            "/// greet documents the API\n// greet is mentioned\nlet text = \"greet\";\ngreet();\n";
        let file_info = make_file_info(source, "src/main.rs");
        let mut symbol_index = HashMap::new();
        symbol_index.insert(
            "greet".to_string(),
            vec![make_definition("greet", "src/lib.rs", 1)],
        );
        let references = ReferenceFinder::new()
            .find_references(&file_info, &symbol_index)
            .unwrap();
        let kinds: Vec<_> = references.iter().map(|r| r.reference_kind).collect();
        assert!(kinds.contains(&ReferenceKind::Documentation));
        assert!(kinds.contains(&ReferenceKind::Comment));
        assert!(kinds.contains(&ReferenceKind::String));
        assert!(kinds.contains(&ReferenceKind::Call));
        assert_eq!(
            references
                .iter()
                .filter(|r| r.reference_kind.is_code())
                .count(),
            1
        );
    }

    #[test]
    fn common_name_candidates_remain_ambiguous() {
        let file_info = make_file_info("Write();", "src/main.rs");
        let mut symbol_index = HashMap::new();
        symbol_index.insert(
            "Write".to_string(),
            vec![
                make_definition("Write", "src/a.rs", 1),
                make_definition("Write", "src/b.rs", 1),
            ],
        );
        let references = ReferenceFinder::new()
            .find_references(&file_info, &symbol_index)
            .unwrap();
        assert_eq!(references.len(), 1);
        assert_eq!(references[0].resolution_status, ResolutionStatus::Ambiguous);
        assert_eq!(references[0].evidence_kind, EvidenceKind::Heuristic);
        assert!(references[0].target_symbol_id.is_empty());
        assert_eq!(references[0].candidates.len(), 2);
    }

    #[test]
    fn unqualified_single_text_candidate_is_not_silently_resolved() {
        let file_info = make_file_info("get();", "src/main.rs");
        let mut symbol_index = HashMap::new();
        symbol_index.insert(
            "get".to_string(),
            vec![make_definition("get", "src/lib.rs", 1)],
        );
        let references = ReferenceFinder::new()
            .find_references(&file_info, &symbol_index)
            .unwrap();
        assert_eq!(
            references[0].resolution_status,
            ResolutionStatus::Unresolved
        );
        assert_eq!(references[0].evidence_kind, EvidenceKind::Heuristic);
        assert!(references[0].target_symbol_id.is_empty());
    }

    #[test]
    fn test_multiple_references() {
        let source = r#"
fn main() {
    greet("Alice");
    greet("Bob");
    greet("Charlie");
}
"#;
        let file_info = make_file_info(source, "src/main.rs");

        let mut symbol_index = HashMap::new();
        symbol_index.insert(
            "greet".to_string(),
            vec![make_definition("greet", "src/lib.rs", 1)],
        );

        let finder = ReferenceFinder::new();
        let references = finder.find_references(&file_info, &symbol_index).unwrap();

        assert_eq!(references.len(), 3);
    }
}
