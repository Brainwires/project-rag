//! Reference finding via identifier matching.
//!
//! This module finds references to symbols by searching for identifier occurrences
//! that match known symbol names from the definition index.

use std::collections::HashMap;

use anyhow::Result;
use chrono::Utc;
use regex::Regex;

use crate::indexer::FileInfo;
use crate::relations::types::{Definition, Reference, ReferenceKind};

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

        // Process each line
        for (line_num, line) in file_info.content.lines().enumerate() {
            let line_number = line_num + 1; // 1-based

            // Find all identifier occurrences in this line
            for mat in self.identifier_regex.find_iter(line) {
                let name = mat.as_str();

                // Check if this identifier matches a known symbol
                if let Some(definitions) = symbol_index.get(name) {
                    // Skip if this is likely a definition site in the same file
                    if self.is_definition_site(definitions, &file_info.relative_path, line_number) {
                        continue;
                    }

                    // Determine reference kind based on context
                    let reference_kind = self.determine_reference_kind(line, mat.start(), name);

                    // Get the best matching definition
                    // For now, just use the first one (could be improved with scope analysis)
                    if let Some(def) = definitions.first() {
                        references.push(Reference {
                            file_path: file_info.relative_path.clone(),
                            root_path: Some(file_info.root_path.clone()),
                            project: file_info.project.clone(),
                            start_line: line_number,
                            end_line: line_number,
                            start_col: mat.start(),
                            end_col: mat.end(),
                            target_symbol_id: def.to_storage_id(),
                            reference_kind,
                            indexed_at: Utc::now().timestamp(),
                        });
                    }
                }
            }
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
    ) -> ReferenceKind {
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
            return ReferenceKind::Import;
        }

        // Check for instantiation (before function call, since `new Foo()` looks like a call)
        if before.contains("new ") {
            return ReferenceKind::Instantiation;
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
            root_path: Some("/test".to_string()),
            project: None,
            end_line: start_line + 5,
            end_col: 0,
            signature: format!("fn {}()", name),
            doc_comment: None,
            visibility: Visibility::Public,
            parent_id: None,
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
        assert!(references.len() >= 1);
        assert!(references.iter().any(|r| r.reference_kind == ReferenceKind::Write));
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
        assert!(references.iter().any(|r| r.reference_kind == ReferenceKind::Import));
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
        assert!(references.iter().any(|r| r.reference_kind == ReferenceKind::Instantiation));
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
