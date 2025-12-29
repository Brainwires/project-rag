//! Symbol extraction from AST nodes.
//!
//! This module extracts symbol definitions (functions, classes, methods, etc.)
//! from source code using tree-sitter AST parsing.

use anyhow::{Context, Result};
use chrono::Utc;
use tree_sitter::{Language, Node, Parser};

use crate::indexer::FileInfo;
use crate::relations::types::{Definition, SymbolId, SymbolKind, Visibility};

/// Extracts symbol definitions from source code using AST parsing.
pub struct SymbolExtractor {
    // No persistent state needed - parser created per-file
}

impl SymbolExtractor {
    /// Create a new symbol extractor
    pub fn new() -> Self {
        Self {}
    }

    /// Extract all symbol definitions from a file
    pub fn extract_definitions(&self, file_info: &FileInfo) -> Result<Vec<Definition>> {
        let extension = file_info.extension.as_deref().unwrap_or("");

        // Get language and parser
        let (language, language_name) = match get_language_for_extension(extension) {
            Some(lang) => lang,
            None => return Ok(Vec::new()), // Unsupported language
        };

        let mut parser = Parser::new();
        parser
            .set_language(&language)
            .context("Failed to set parser language")?;

        let tree = parser
            .parse(&file_info.content, None)
            .context("Failed to parse source code")?;

        let root_node = tree.root_node();
        let mut definitions = Vec::new();

        // Extract definitions recursively
        self.extract_from_node(
            root_node,
            &file_info.content,
            &language_name,
            file_info,
            None,
            &mut definitions,
        );

        Ok(definitions)
    }

    /// Extract definitions from a node and its children
    fn extract_from_node(
        &self,
        node: Node,
        source: &str,
        language: &str,
        file_info: &FileInfo,
        parent_id: Option<String>,
        result: &mut Vec<Definition>,
    ) {
        let kind = node.kind();

        // Check if this node is a definition we care about
        if is_definition_node(kind, language) {
            if let Some(def) = self.node_to_definition(node, source, language, file_info, &parent_id)
            {
                let new_parent_id = Some(def.to_storage_id());
                result.push(def);

                // Extract nested definitions with this as parent
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    self.extract_from_node(
                        child,
                        source,
                        language,
                        file_info,
                        new_parent_id.clone(),
                        result,
                    );
                }
                return;
            }
        }

        // Recurse into children
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.extract_from_node(child, source, language, file_info, parent_id.clone(), result);
        }
    }

    /// Convert an AST node to a Definition
    fn node_to_definition(
        &self,
        node: Node,
        source: &str,
        language: &str,
        file_info: &FileInfo,
        parent_id: &Option<String>,
    ) -> Option<Definition> {
        let kind = node.kind();
        let symbol_kind = SymbolKind::from_ast_kind(kind);

        // Extract the symbol name
        let name = extract_symbol_name(node, source, language)?;

        // Get position info
        let start_pos = node.start_position();
        let end_pos = node.end_position();

        // Extract signature (first line or declaration)
        let signature = extract_signature(node, source, language);

        // Extract doc comment
        let doc_comment = extract_doc_comment(node, source, language);

        // Determine visibility
        let node_text = &source[node.start_byte()..node.end_byte().min(source.len())];
        let visibility = Visibility::from_keywords(node_text);

        Some(Definition {
            symbol_id: SymbolId::new(
                &file_info.relative_path,
                name,
                symbol_kind,
                start_pos.row + 1, // Convert to 1-based
                start_pos.column,
            ),
            root_path: Some(file_info.root_path.clone()),
            project: file_info.project.clone(),
            end_line: end_pos.row + 1,
            end_col: end_pos.column,
            signature,
            doc_comment,
            visibility,
            parent_id: parent_id.clone(),
            indexed_at: Utc::now().timestamp(),
        })
    }
}

impl Default for SymbolExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the tree-sitter language for a file extension
fn get_language_for_extension(extension: &str) -> Option<(Language, String)> {
    match extension.to_lowercase().as_str() {
        "rs" => Some((tree_sitter_rust::LANGUAGE.into(), "Rust".to_string())),
        "py" => Some((tree_sitter_python::LANGUAGE.into(), "Python".to_string())),
        "js" | "mjs" | "cjs" | "jsx" => Some((
            tree_sitter_javascript::LANGUAGE.into(),
            "JavaScript".to_string(),
        )),
        "ts" | "tsx" => Some((
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            "TypeScript".to_string(),
        )),
        "go" => Some((tree_sitter_go::LANGUAGE.into(), "Go".to_string())),
        "java" => Some((tree_sitter_java::LANGUAGE.into(), "Java".to_string())),
        "swift" => Some((tree_sitter_swift::LANGUAGE.into(), "Swift".to_string())),
        "c" | "h" => Some((tree_sitter_c::LANGUAGE.into(), "C".to_string())),
        "cpp" | "cc" | "cxx" | "hpp" | "hxx" | "hh" => {
            Some((tree_sitter_cpp::LANGUAGE.into(), "C++".to_string()))
        }
        "cs" => Some((tree_sitter_c_sharp::LANGUAGE.into(), "C#".to_string())),
        "rb" => Some((tree_sitter_ruby::LANGUAGE.into(), "Ruby".to_string())),
        "php" => Some((tree_sitter_php::LANGUAGE_PHP.into(), "PHP".to_string())),
        _ => None,
    }
}

/// Check if a node kind represents a definition
fn is_definition_node(kind: &str, language: &str) -> bool {
    match language {
        "Rust" => matches!(
            kind,
            "function_item"
                | "impl_item"
                | "trait_item"
                | "struct_item"
                | "enum_item"
                | "mod_item"
                | "const_item"
                | "static_item"
                | "type_item"
        ),
        "Python" => matches!(
            kind,
            "function_definition" | "class_definition" | "decorated_definition"
        ),
        "JavaScript" | "TypeScript" => matches!(
            kind,
            "function_declaration"
                | "function_expression"
                | "arrow_function"
                | "method_definition"
                | "class_declaration"
                | "interface_declaration"
                | "type_alias_declaration"
        ),
        "Go" => matches!(
            kind,
            "function_declaration" | "method_declaration" | "type_declaration"
        ),
        "Java" => matches!(
            kind,
            "method_declaration"
                | "class_declaration"
                | "interface_declaration"
                | "constructor_declaration"
                | "enum_declaration"
        ),
        "Swift" => matches!(
            kind,
            "function_declaration"
                | "class_declaration"
                | "struct_declaration"
                | "enum_declaration"
                | "protocol_declaration"
        ),
        "C" => matches!(
            kind,
            "function_definition" | "struct_specifier" | "enum_specifier"
        ),
        "C++" => matches!(
            kind,
            "function_definition"
                | "class_specifier"
                | "struct_specifier"
                | "enum_specifier"
                | "namespace_definition"
        ),
        "C#" => matches!(
            kind,
            "method_declaration"
                | "class_declaration"
                | "struct_declaration"
                | "interface_declaration"
                | "enum_declaration"
                | "constructor_declaration"
        ),
        "Ruby" => matches!(kind, "method" | "singleton_method" | "class" | "module"),
        "PHP" => matches!(
            kind,
            "function_definition"
                | "method_declaration"
                | "class_declaration"
                | "interface_declaration"
                | "trait_declaration"
        ),
        _ => false,
    }
}

/// Extract the symbol name from an AST node
fn extract_symbol_name(node: Node, source: &str, language: &str) -> Option<String> {
    // Strategy: Find the identifier/name child node based on language
    let name_node = find_name_node(node, language)?;

    let start = name_node.start_byte();
    let end = name_node.end_byte();

    if end > source.len() {
        return None;
    }

    let name = source[start..end].to_string();

    // Filter out empty or whitespace-only names
    if name.trim().is_empty() {
        return None;
    }

    Some(name)
}

/// Find the child node containing the symbol name
fn find_name_node<'a>(node: Node<'a>, language: &str) -> Option<Node<'a>> {
    let kind = node.kind();

    // Language-specific name extraction
    match language {
        "Rust" => {
            // Rust: name is usually in "name" field or first identifier
            if let Some(name_node) = node.child_by_field_name("name") {
                return Some(name_node);
            }
            // For impl items, look for type name
            if kind == "impl_item" {
                if let Some(type_node) = node.child_by_field_name("type") {
                    return Some(type_node);
                }
            }
        }
        "Python" => {
            // Python: class and function have "name" field
            if let Some(name_node) = node.child_by_field_name("name") {
                return Some(name_node);
            }
            // Decorated definitions: look inside for the actual definition
            if kind == "decorated_definition" {
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if child.kind() == "function_definition" || child.kind() == "class_definition" {
                        return find_name_node(child, language);
                    }
                }
            }
        }
        "JavaScript" | "TypeScript" => {
            // JS/TS: "name" field for most declarations
            if let Some(name_node) = node.child_by_field_name("name") {
                return Some(name_node);
            }
            // Arrow functions in variable declarations need special handling
            if kind == "arrow_function" {
                // Look at parent for variable name
                if let Some(parent) = node.parent() {
                    if parent.kind() == "variable_declarator" {
                        if let Some(name_node) = parent.child_by_field_name("name") {
                            return Some(name_node);
                        }
                    }
                }
            }
        }
        "Go" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                return Some(name_node);
            }
        }
        "Java" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                return Some(name_node);
            }
        }
        "Swift" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                return Some(name_node);
            }
        }
        "C" | "C++" => {
            // C/C++: declarator contains the name
            if let Some(declarator) = node.child_by_field_name("declarator") {
                // Navigate through possible pointer/reference declarators
                return find_innermost_identifier(declarator);
            }
            // For struct/class, name is in the type specifier
            if kind == "struct_specifier" || kind == "class_specifier" || kind == "enum_specifier" {
                if let Some(name_node) = node.child_by_field_name("name") {
                    return Some(name_node);
                }
            }
        }
        "C#" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                return Some(name_node);
            }
        }
        "Ruby" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                return Some(name_node);
            }
        }
        "PHP" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                return Some(name_node);
            }
        }
        _ => {}
    }

    // Fallback: find first identifier child
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "identifier"
            || child.kind() == "type_identifier"
            || child.kind() == "name"
        {
            return Some(child);
        }
    }

    None
}

/// Find the innermost identifier in a declarator chain (for C/C++)
fn find_innermost_identifier<'a>(node: Node<'a>) -> Option<Node<'a>> {
    // If this is an identifier, return it
    if node.kind() == "identifier" || node.kind() == "field_identifier" {
        return Some(node);
    }

    // Check for name field
    if let Some(name_node) = node.child_by_field_name("declarator") {
        return find_innermost_identifier(name_node);
    }

    // Fallback: look through children
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if let Some(id) = find_innermost_identifier(child) {
            return Some(id);
        }
    }

    None
}

/// Extract the signature (first line of declaration)
fn extract_signature(node: Node, source: &str, _language: &str) -> String {
    let start = node.start_byte();
    let end = node.end_byte().min(source.len());
    let text = &source[start..end];

    // Get first line or first 200 chars, whichever is shorter
    let first_line = text.lines().next().unwrap_or("");
    if first_line.len() > 200 {
        format!("{}...", &first_line[..200])
    } else {
        first_line.to_string()
    }
}

/// Extract documentation comment preceding the node
fn extract_doc_comment(node: Node, source: &str, language: &str) -> Option<String> {
    // Look for comment sibling before this node
    let mut prev = node.prev_sibling();

    while let Some(sibling) = prev {
        let kind = sibling.kind();

        // Check if it's a comment
        let is_doc_comment = match language {
            "Rust" => kind == "line_comment" || kind == "block_comment",
            "Python" => kind == "comment" || kind == "expression_statement", // docstrings
            "JavaScript" | "TypeScript" => kind == "comment",
            "Java" => kind == "line_comment" || kind == "block_comment",
            "Go" => kind == "comment",
            "C" | "C++" => kind == "comment",
            "C#" => kind == "comment",
            "Ruby" => kind == "comment",
            "PHP" => kind == "comment",
            _ => kind.contains("comment"),
        };

        if is_doc_comment {
            let start = sibling.start_byte();
            let end = sibling.end_byte().min(source.len());
            let comment = source[start..end].trim().to_string();

            // Clean up comment syntax
            let cleaned = clean_comment(&comment, language);
            if !cleaned.is_empty() {
                return Some(cleaned);
            }
        }

        // Stop if we hit a non-comment, non-whitespace node
        if !kind.contains("comment") && kind != "decorator" && kind != "attribute" {
            break;
        }

        prev = sibling.prev_sibling();
    }

    None
}

/// Clean comment syntax from a comment string
fn clean_comment(comment: &str, _language: &str) -> String {
    let lines: Vec<&str> = comment.lines().collect();

    let cleaned: Vec<String> = lines
        .iter()
        .map(|line| {
            let mut s = line.trim();
            // Remove common prefixes
            for prefix in ["///", "//!", "//", "/*", "*/", "*", "#", "\"\"\"", "'''"] {
                s = s.trim_start_matches(prefix);
            }
            s.trim().to_string()
        })
        .filter(|s| !s.is_empty())
        .collect();

    cleaned.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_file_info(content: &str, extension: &str) -> FileInfo {
        FileInfo {
            path: PathBuf::from(format!("test.{}", extension)),
            relative_path: format!("test.{}", extension),
            root_path: "/test".to_string(),
            project: None,
            extension: Some(extension.to_string()),
            language: None,
            content: content.to_string(),
            hash: "test_hash".to_string(),
        }
    }

    #[test]
    fn test_rust_extraction() {
        let source = r#"
/// A greeting function
pub fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}

struct Person {
    name: String,
}

impl Person {
    fn new(name: String) -> Self {
        Self { name }
    }
}
"#;
        let file_info = make_file_info(source, "rs");
        let extractor = SymbolExtractor::new();
        let definitions = extractor.extract_definitions(&file_info).unwrap();

        assert!(!definitions.is_empty());

        // Find the greet function
        let greet = definitions.iter().find(|d| d.name() == "greet");
        assert!(greet.is_some(), "Should find greet function");

        let greet = greet.unwrap();
        assert_eq!(greet.kind(), SymbolKind::Function);
        assert_eq!(greet.visibility, Visibility::Public);
        assert!(greet.doc_comment.is_some());
    }

    #[test]
    fn test_python_extraction() {
        let source = r#"
def hello(name):
    """Say hello."""
    print(f"Hello, {name}!")

class MyClass:
    def __init__(self, value):
        self.value = value
"#;
        let file_info = make_file_info(source, "py");
        let extractor = SymbolExtractor::new();
        let definitions = extractor.extract_definitions(&file_info).unwrap();

        assert!(!definitions.is_empty());

        // Find hello function
        let hello = definitions.iter().find(|d| d.name() == "hello");
        assert!(hello.is_some(), "Should find hello function");

        // Find MyClass
        let my_class = definitions.iter().find(|d| d.name() == "MyClass");
        assert!(my_class.is_some(), "Should find MyClass");
    }

    #[test]
    fn test_javascript_extraction() {
        let source = r#"
function add(a, b) {
    return a + b;
}

class Calculator {
    constructor() {
        this.result = 0;
    }

    add(x) {
        this.result += x;
    }
}
"#;
        let file_info = make_file_info(source, "js");
        let extractor = SymbolExtractor::new();
        let definitions = extractor.extract_definitions(&file_info).unwrap();

        assert!(!definitions.is_empty());

        // Find add function
        let add = definitions.iter().find(|d| d.name() == "add");
        assert!(add.is_some(), "Should find add function");
    }

    #[test]
    fn test_unsupported_extension() {
        let source = "some content";
        let file_info = make_file_info(source, "xyz");
        let extractor = SymbolExtractor::new();
        let definitions = extractor.extract_definitions(&file_info).unwrap();

        assert!(definitions.is_empty());
    }

    #[test]
    fn test_definition_storage_id() {
        let source = "fn foo() {}";
        let file_info = make_file_info(source, "rs");
        let extractor = SymbolExtractor::new();
        let definitions = extractor.extract_definitions(&file_info).unwrap();

        assert!(!definitions.is_empty());
        let def = &definitions[0];
        let storage_id = def.to_storage_id();
        assert!(storage_id.contains("foo"));
    }
}
