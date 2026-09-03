//! Symbol extraction from AST nodes.
//!
//! This module extracts symbol definitions (functions, classes, methods, etc.)
//! from source code using tree-sitter AST parsing.

use anyhow::{Context, Result};
use chrono::Utc;
use tree_sitter::{Language, Node, Parser};

use crate::indexer::FileInfo;
use crate::relations::types::{
    Definition, LinkageKind, LocationRole, SkippedDefinition, SourceLocation, SymbolId, SymbolKind,
    Visibility,
};

#[derive(Clone)]
struct ParentScope {
    symbol_id: String,
    qualified_name: String,
    kind: SymbolKind,
    anonymous: bool,
}

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
        let (definitions, _skipped) = self.extract_definitions_reporting(file_info)?;
        Ok(definitions)
    }

    /// Extract all symbol definitions, and report every node that was recognised as a
    /// definition but could not be named.
    ///
    /// Those nodes are omitted from the returned definitions -- they used to be dropped
    /// with no diagnostic at all, which made an incomplete listing indistinguishable
    /// from a complete one.
    pub fn extract_definitions_reporting(
        &self,
        file_info: &FileInfo,
    ) -> Result<(Vec<Definition>, Vec<SkippedDefinition>)> {
        let extension = file_info.extension.as_deref().unwrap_or("");

        // Get language and parser
        let (language, language_name) = match get_language_for_extension(extension) {
            Some(lang) => lang,
            None => return Ok((Vec::new(), Vec::new())), // Unsupported language
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
        let mut skipped = Vec::new();

        // Extract definitions recursively
        self.extract_from_node(
            root_node,
            &file_info.content,
            &language_name,
            file_info,
            None,
            &mut definitions,
            &mut skipped,
        );

        if !skipped.is_empty() {
            tracing::warn!(
                file = %file_info.relative_path,
                skipped = skipped.len(),
                found = definitions.len(),
                "symbol extraction omitted definition nodes it could not name; the symbol list for this file is incomplete"
            );
        }

        Ok((definitions, skipped))
    }

    /// Extract definitions from a node and its children
    #[allow(clippy::too_many_arguments)]
    fn extract_from_node(
        &self,
        node: Node,
        source: &str,
        language: &str,
        file_info: &FileInfo,
        parent: Option<ParentScope>,
        result: &mut Vec<Definition>,
        skipped: &mut Vec<SkippedDefinition>,
    ) {
        let kind = node.kind();

        // Import nodes bind names into scope; each bound name becomes its own
        // SymbolKind::Import definition (`use a::{B, C}` yields two). A statement
        // that binds no checkable name (globs, side-effect imports) is recorded as
        // skipped so the listing is visibly incomplete rather than silently short.
        if super::import_extractor::is_import_node(kind, language) {
            let parent_id = parent.as_ref().map(|p| p.symbol_id.clone());
            let imports = super::import_extractor::extract_imports(
                node, source, language, file_info, &parent_id,
            );
            if imports.is_empty() {
                skipped.push(skipped_from_node(
                    node,
                    source,
                    "could not extract bound names from this import",
                ));
            } else {
                result.extend(imports);
            }
            return; // nothing definable nests inside an import statement
        }

        // Check if this node is a definition we care about
        if is_definition_node(node, language) {
            if let Some(def) =
                self.node_to_definition(node, source, language, file_info, parent.as_ref())
            {
                let new_parent = Some(ParentScope {
                    symbol_id: def.to_storage_id(),
                    qualified_name: def.symbol_id.qualified_name.clone(),
                    kind: def.symbol_id.kind,
                    anonymous: def.symbol_id.linkage == LinkageKind::Anonymous,
                });
                result.push(def);

                // Extract nested definitions with this as parent
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    self.extract_from_node(
                        child,
                        source,
                        language,
                        file_info,
                        new_parent.clone(),
                        result,
                        skipped,
                    );
                }
                return;
            }

            // This node IS a definition but no name could be extracted from it, so it
            // will not appear in the symbol list. Record it rather than dropping it
            // silently -- the caller cannot otherwise tell that the listing is short.
            skipped.push(skipped_from_node(
                node,
                source,
                "could not extract a name from this node",
            ));
        }

        // Recurse into children
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.extract_from_node(
                child,
                source,
                language,
                file_info,
                parent.clone(),
                result,
                skipped,
            );
        }
    }

    /// Convert an AST node to a Definition
    fn node_to_definition(
        &self,
        node: Node,
        source: &str,
        language: &str,
        file_info: &FileInfo,
        parent: Option<&ParentScope>,
    ) -> Option<Definition> {
        let kind = node.kind();
        let mut symbol_kind = SymbolKind::from_ast_kind(kind);

        // Extract the symbol name
        let name_node = find_name_node(node, language);
        let name = match name_node {
            Some(name_node) => source
                .get(name_node.start_byte()..name_node.end_byte())?
                .to_string(),
            None if kind == "namespace_definition" => "<anonymous>".to_string(),
            None => return None,
        };

        if kind == "namespace_definition" {
            symbol_kind = SymbolKind::Namespace;
        } else if matches!(kind, "declaration" | "field_declaration") {
            symbol_kind = if parent.is_some_and(|p| is_type_scope(p.kind)) {
                SymbolKind::Method
            } else {
                SymbolKind::Function
            };
        } else if symbol_kind == SymbolKind::Function
            && parent.is_some_and(|p| is_type_scope(p.kind))
        {
            symbol_kind = SymbolKind::Method;
        }

        let qualified_from_source = if language == "C++" {
            cpp_qualified_callable_name(node, source)
        } else {
            None
        };
        let qualified_name = qualified_from_source.unwrap_or_else(|| match parent {
            Some(parent) if !parent.qualified_name.is_empty() => {
                format!("{}::{}", parent.qualified_name, name)
            }
            _ => name.clone(),
        });

        if symbol_kind == SymbolKind::Function && qualified_name.contains("::") {
            symbol_kind = SymbolKind::Method;
        }

        if matches!(symbol_kind, SymbolKind::Function | SymbolKind::Method) {
            let owner = qualified_name.rsplit_once("::").map(|(owner, _)| owner);
            let qualified_leaf = qualified_name.rsplit("::").next().unwrap_or(&name);
            if qualified_leaf.starts_with('~') {
                symbol_kind = SymbolKind::Destructor;
            } else if owner.and_then(|o| o.rsplit("::").next()) == Some(qualified_leaf) {
                symbol_kind = SymbolKind::Constructor;
            }
        }

        // Get position info
        let start_pos = node.start_position();
        let end_pos = node.end_position();

        // Extract signature (first line or declaration)
        let signature = extract_signature(node, source, language);
        let canonical_signature = canonical_signature(node, source, &name, language);

        // Extract doc comment
        let doc_comment = extract_doc_comment(node, source, language);

        // Determine visibility
        let node_text = &source[node.start_byte()..node.end_byte().min(source.len())];
        let visibility = Visibility::from_keywords(node_text);
        let is_anonymous = name == "<anonymous>" || parent.is_some_and(|p| p.anonymous);
        let linkage = if is_anonymous {
            LinkageKind::Anonymous
        } else if node_text.trim_start().starts_with("static ") {
            LinkageKind::Internal
        } else if parent
            .is_some_and(|p| matches!(p.kind, SymbolKind::Function | SymbolKind::Method))
        {
            LinkageKind::Local
        } else {
            LinkageKind::External
        };
        let scope_discriminator =
            (!matches!(linkage, LinkageKind::External)).then(|| file_info.relative_path.clone());

        let location_role = if matches!(kind, "declaration" | "field_declaration") {
            LocationRole::Declaration
        } else {
            LocationRole::Definition
        };
        let name_start = name_node.unwrap_or(node).start_position();
        let name_end = name_node.unwrap_or(node).end_position();
        let location = SourceLocation {
            project_id: file_info.project.clone().unwrap_or_default(),
            file_path: file_info.relative_path.clone(),
            start_line: name_start.row + 1,
            start_col: name_start.column,
            end_line: name_end.row + 1,
            end_col: name_end.column,
            role: location_role,
        };

        Some(Definition {
            symbol_id: SymbolId::new_logical(
                file_info.project.clone().unwrap_or_default(),
                language,
                qualified_name,
                name,
                symbol_kind,
                canonical_signature,
                linkage,
                scope_discriminator,
                &file_info.relative_path,
                start_pos.row + 1, // Convert to 1-based
                start_pos.column,
            ),
            location,
            root_path: Some(file_info.root_path.clone()),
            project: file_info.project.clone(),
            end_line: end_pos.row + 1,
            end_col: end_pos.column,
            signature,
            doc_comment,
            visibility,
            parent_id: parent.map(|p| p.symbol_id.clone()),
            parser: format!("tree-sitter/{}", language.to_lowercase()),
            indexed_at: Utc::now().timestamp(),
        })
    }
}

impl Default for SymbolExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the skipped-definition record for a node whose name (or bound names)
/// could not be extracted.
fn skipped_from_node(node: Node, source: &str, reason: &str) -> SkippedDefinition {
    let snippet = source
        .get(node.start_byte()..node.end_byte().min(source.len()))
        .unwrap_or("")
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .chars()
        .take(120)
        .collect::<String>();
    SkippedDefinition {
        line: node.start_position().row + 1,
        kind: node.kind().to_string(),
        reason: reason.to_string(),
        snippet,
    }
}

/// Extractor-taxonomy language name for a file extension.
///
/// Single source of truth for language dispatch: get_language_for_extension
/// derives its grammar from this name, so a file whose imports were extracted
/// under one language name can never be usage-checked under another. Headers
/// map to "C++" -- that grammar accepts C structs and enums and additionally
/// yields classes and namespaces, which the C grammar cannot.
pub fn language_name_for_extension(extension: &str) -> Option<&'static str> {
    Some(match extension.to_lowercase().as_str() {
        "rs" => "Rust",
        "py" => "Python",
        "js" | "mjs" | "cjs" | "jsx" => "JavaScript",
        "ts" | "tsx" => "TypeScript",
        "go" => "Go",
        "java" => "Java",
        "swift" => "Swift",
        "c" => "C",
        "h" | "cpp" | "cc" | "cxx" | "cu" | "hpp" | "hxx" | "hh" | "cuh" => "C++",
        "cs" => "C#",
        "rb" => "Ruby",
        "php" => "PHP",
        _ => return None,
    })
}

/// Get the tree-sitter language for a file extension
pub(super) fn get_language_for_extension(extension: &str) -> Option<(Language, String)> {
    let name = language_name_for_extension(extension)?;
    let language: Language = match name {
        "Rust" => tree_sitter_rust::LANGUAGE.into(),
        "Python" => tree_sitter_python::LANGUAGE.into(),
        "JavaScript" => tree_sitter_javascript::LANGUAGE.into(),
        "TypeScript" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        "Go" => tree_sitter_go::LANGUAGE.into(),
        "Java" => tree_sitter_java::LANGUAGE.into(),
        "Swift" => tree_sitter_swift::LANGUAGE.into(),
        "C" => tree_sitter_c::LANGUAGE.into(),
        "C++" => tree_sitter_cpp::LANGUAGE.into(),
        "C#" => tree_sitter_c_sharp::LANGUAGE.into(),
        "Ruby" => tree_sitter_ruby::LANGUAGE.into(),
        "PHP" => tree_sitter_php::LANGUAGE_PHP.into(),
        _ => return None,
    };
    Some((language, name.to_string()))
}

/// Check if a node kind represents a definition
fn is_definition_node(node: Node<'_>, language: &str) -> bool {
    let kind = node.kind();
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
        "C++" => {
            matches!(
                kind,
                "function_definition"
                    | "class_specifier"
                    | "struct_specifier"
                    | "enum_specifier"
                    | "namespace_definition"
            ) || (matches!(kind, "declaration" | "field_declaration")
                && contains_node_kind(node, "function_declarator"))
        }
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

fn contains_node_kind(node: Node<'_>, wanted: &str) -> bool {
    if node.kind() == wanted {
        return true;
    }
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .any(|child| contains_node_kind(child, wanted))
}

fn is_type_scope(kind: SymbolKind) -> bool {
    matches!(
        kind,
        SymbolKind::Class
            | SymbolKind::Struct
            | SymbolKind::Interface
            | SymbolKind::Trait
            | SymbolKind::Namespace
            | SymbolKind::Module
    )
}

fn cpp_qualified_callable_name(node: Node<'_>, source: &str) -> Option<String> {
    let declarator = node.child_by_field_name("declarator")?;
    let text = source.get(declarator.start_byte()..declarator.end_byte())?;
    let before_params = text.split('(').next()?.trim();
    let candidate = before_params
        .split_whitespace()
        .last()?
        .trim_matches(|c| matches!(c, '*' | '&'));
    candidate.contains("::").then(|| candidate.to_string())
}

fn canonical_signature(node: Node<'_>, source: &str, name: &str, language: &str) -> String {
    let raw = if matches!(language, "C" | "C++") {
        node.child_by_field_name("declarator")
            .and_then(|d| source.get(d.start_byte()..d.end_byte()))
            .unwrap_or_else(|| source.get(node.start_byte()..node.end_byte()).unwrap_or(""))
    } else {
        source.get(node.start_byte()..node.end_byte()).unwrap_or("")
    };
    let header = raw
        .split('{')
        .next()
        .unwrap_or(raw)
        .trim_end_matches(';')
        .trim();
    let callable = if let Some(open) = header.find('(') {
        let prefix = &header[..open];
        let callable_name = prefix
            .split_whitespace()
            .last()
            .unwrap_or(name)
            .rsplit("::")
            .next()
            .unwrap_or(name);
        format!("{}{}", callable_name, &header[open..])
    } else {
        name.to_string()
    };
    normalize_signature(&callable)
}

fn normalize_signature(signature: &str) -> String {
    let collapsed = signature.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed.replace(" ", "").trim_end_matches(';').to_string()
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
            if kind == "impl_item"
                && let Some(type_node) = node.child_by_field_name("type")
            {
                return Some(type_node);
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
            if kind == "arrow_function"
                && let Some(parent) = node.parent()
                && parent.kind() == "variable_declarator"
                && let Some(name_node) = parent.child_by_field_name("name")
            {
                return Some(name_node);
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
                // Navigate through possible pointer/reference declarators.
                // Only return on success: returning None here would skip the generic
                // identifier fallback at the end of this function, which is what made
                // unnameable-but-valid definitions disappear without a trace.
                if let Some(id) = find_innermost_identifier(declarator) {
                    return Some(id);
                }
            }
            // For struct/class/enum/namespace, the name is its own field. The
            // namespace name node's kind is namespace_identifier, which the
            // generic identifier fallback below does not match.
            if matches!(
                kind,
                "struct_specifier" | "class_specifier" | "enum_specifier" | "namespace_definition"
            ) && let Some(name_node) = node.child_by_field_name("name")
            {
                return Some(name_node);
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
    node.children(&mut cursor).find(|child| {
        child.kind() == "identifier" || child.kind() == "type_identifier" || child.kind() == "name"
    })
}

/// Find the innermost identifier in a declarator chain (for C/C++)
fn find_innermost_identifier<'a>(node: Node<'a>) -> Option<Node<'a>> {
    // If this is an identifier, return it
    if node.kind() == "identifier" || node.kind() == "field_identifier" {
        return Some(node);
    }

    if let Some(name_node) = node.child_by_field_name("name")
        && let Some(id) = find_innermost_identifier(name_node)
    {
        return Some(id);
    }

    // Check for name field. Only return on success -- an unconditional return here
    // skips the child scan below, which is the same defect as in find_name_node.
    if let Some(name_node) = node.child_by_field_name("declarator")
        && let Some(id) = find_innermost_identifier(name_node)
    {
        return Some(id);
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

    #[test]
    fn test_language_name_for_extension_c_family() {
        assert_eq!(language_name_for_extension("c"), Some("C"));
        for ext in ["h", "hh", "hxx", "hpp", "cuh", "cpp", "cc", "cxx", "cu"] {
            assert_eq!(language_name_for_extension(ext), Some("C++"), "{}", ext);
        }
        assert_eq!(language_name_for_extension("xyz"), None);
    }

    #[test]
    fn test_header_extracted_with_cpp_grammar() {
        let source = "class KioskNotify {\npublic:\n  void fire();\n};\nnamespace kiosk {\nstruct S {};\n}\n";
        let file_info = make_file_info(source, "h");
        let extractor = SymbolExtractor::new();
        let definitions = extractor.extract_definitions(&file_info).unwrap();

        // The C grammar yielded no class or namespace nodes for headers.
        assert!(definitions.iter().any(|d| d.name() == "KioskNotify"));
        assert!(definitions.iter().any(|d| d.name() == "kiosk"));
        assert!(definitions.iter().any(|d| d.name() == "S"));
    }

    #[test]
    fn test_cuda_extracted_with_cpp_grammar() {
        let source = "__global__ void scale_kernel(float * data) { data[threadIdx.x] *= 2.0f; }\n__device__ int lane_id() { return threadIdx.x; }\n";
        let extractor = SymbolExtractor::new();

        for extension in ["cu", "cuh"] {
            let file_info = make_file_info(source, extension);
            let definitions = extractor.extract_definitions(&file_info).unwrap();

            assert!(
                definitions.iter().any(|d| d.name() == "scale_kernel"),
                "{}",
                extension
            );
            assert!(
                definitions.iter().any(|d| d.name() == "lane_id"),
                "{}",
                extension
            );
        }
    }

    #[test]
    fn cpp_declaration_and_definition_share_logical_id() {
        let mut header = make_file_info(
            "class Writer { public: void Write(int value) const; };",
            "hpp",
        );
        header.relative_path = "include/writer.hpp".to_string();
        header.project = Some("stable-project".to_string());
        let mut source = make_file_info("void Writer::Write(int value) const { }", "cpp");
        source.relative_path = "src/writer.cpp".to_string();
        source.project = Some("stable-project".to_string());

        let extractor = SymbolExtractor::new();
        let header_defs = extractor.extract_definitions(&header).unwrap();
        let source_defs = extractor.extract_definitions(&source).unwrap();
        let declaration = header_defs.iter().find(|d| d.name() == "Write").unwrap();
        let definition = source_defs.iter().find(|d| d.name() == "Write").unwrap();

        assert_eq!(declaration.kind(), SymbolKind::Method);
        assert_eq!(definition.kind(), SymbolKind::Method);
        assert_eq!(declaration.location.role, LocationRole::Declaration);
        assert_eq!(definition.location.role, LocationRole::Definition);
        assert_eq!(declaration.to_storage_id(), definition.to_storage_id());
        assert_ne!(
            declaration.location.to_storage_id(),
            definition.location.to_storage_id()
        );
    }

    #[test]
    fn cpp_overloads_and_file_local_symbols_have_distinct_ids() {
        let mut overloads = make_file_info(
            "void Write(int value) {}\nvoid Write(const char *value) {}",
            "cpp",
        );
        overloads.project = Some("stable-project".to_string());
        let extractor = SymbolExtractor::new();
        let defs = extractor.extract_definitions(&overloads).unwrap();
        let writes: Vec<_> = defs.iter().filter(|d| d.name() == "Write").collect();
        assert_eq!(writes.len(), 2);
        assert_ne!(writes[0].to_storage_id(), writes[1].to_storage_id());

        let mut first = make_file_info("static void helper() {}", "cpp");
        first.relative_path = "src/a/common.cpp".to_string();
        first.project = Some("stable-project".to_string());
        let mut second = make_file_info("static void helper() {}", "cpp");
        second.relative_path = "src/b/common.cpp".to_string();
        second.project = Some("stable-project".to_string());
        let first_id = extractor.extract_definitions(&first).unwrap()[0].to_storage_id();
        let second_id = extractor.extract_definitions(&second).unwrap()[0].to_storage_id();
        assert_ne!(first_id, second_id);
    }

    #[test]
    fn anonymous_namespace_symbols_are_file_scoped() {
        let mut first = make_file_info("namespace { void helper() {} }", "cpp");
        first.relative_path = "src/a.cpp".to_string();
        first.project = Some("stable-project".to_string());
        let mut second = make_file_info("namespace { void helper() {} }", "cpp");
        second.relative_path = "src/b.cpp".to_string();
        second.project = Some("stable-project".to_string());
        let extractor = SymbolExtractor::new();
        let first_def = extractor
            .extract_definitions(&first)
            .unwrap()
            .into_iter()
            .find(|d| d.name() == "helper")
            .unwrap();
        let second_def = extractor
            .extract_definitions(&second)
            .unwrap()
            .into_iter()
            .find(|d| d.name() == "helper")
            .unwrap();
        assert_eq!(first_def.symbol_id.linkage, LinkageKind::Anonymous);
        assert_ne!(first_def.to_storage_id(), second_def.to_storage_id());
    }
}
