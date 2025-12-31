//! Type definitions for code relationships (definitions, references, call graphs).
//!
//! This module provides the core data structures for representing code relationships:
//! - `SymbolId`: Unique identifier for a symbol in the codebase
//! - `Definition`: A symbol definition (function, class, method, etc.)
//! - `Reference`: A reference to a symbol
//! - `CallEdge`: An edge in the call graph

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

/// Kind of symbol in the codebase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    /// A function (standalone)
    Function,
    /// A method (belongs to a class/struct/impl)
    Method,
    /// A class definition
    Class,
    /// A struct definition
    Struct,
    /// An interface definition
    Interface,
    /// A trait definition (Rust)
    Trait,
    /// An enum definition
    Enum,
    /// A module/namespace
    Module,
    /// A variable/binding
    Variable,
    /// A constant
    Constant,
    /// A function/method parameter
    Parameter,
    /// A class/struct field
    Field,
    /// An import statement
    Import,
    /// An export statement
    Export,
    /// An enum variant
    EnumVariant,
    /// A type alias
    TypeAlias,
    /// Unknown or unclassified symbol
    Unknown,
}

impl SymbolKind {
    /// Convert from AST node kind string to SymbolKind.
    ///
    /// This consolidates AST node kinds from all supported languages into
    /// a single mapping to avoid duplicates.
    pub fn from_ast_kind(kind: &str) -> Self {
        match kind {
            // Functions (various languages)
            "function_item" // Rust
            | "function_definition" // Python, C, PHP
            | "function_declaration" // JS/TS, Go, Swift
            | "function_expression" // JS/TS
            | "arrow_function" // JS/TS
            | "decorated_definition" // Python (could be either, default to function)
            => Self::Function,

            // Methods
            "method_definition" // JS/TS
            | "method_declaration" // Java, Go, PHP
            | "method" // Ruby
            | "singleton_method" // Ruby
            | "constructor_declaration" // Java
            => Self::Method,

            // Classes
            "impl_item" // Rust (impl blocks treated as class-like)
            | "class_definition" // Python
            | "class_declaration" // JS/TS, Java, PHP, Swift
            | "class_specifier" // C++
            | "class" // Ruby
            => Self::Class,

            // Structs
            "struct_item" // Rust
            | "struct_specifier" // C/C++
            | "struct_declaration" // Swift, C#
            => Self::Struct,

            // Interfaces/Protocols
            "interface_declaration" // JS/TS, Java, PHP, C#
            | "protocol_declaration" // Swift
            => Self::Interface,

            // Traits
            "trait_item" // Rust
            | "trait_declaration" // PHP
            => Self::Trait,

            // Enums
            "enum_item" // Rust
            | "enum_declaration" // JS/TS, Java, Swift, C#
            | "enum_specifier" // C/C++
            => Self::Enum,

            // Modules/Namespaces
            "mod_item" // Rust
            | "module" // Ruby
            | "namespace_definition" // C++, PHP
            | "namespace_declaration" // C#
            => Self::Module,

            // Variables
            "static_item" // Rust
            | "variable_declaration" // JS/TS
            | "lexical_declaration" // JS/TS
            => Self::Variable,

            // Constants
            "const_item" // Rust
            => Self::Constant,

            // Type aliases
            "type_item" // Rust
            | "type_alias_declaration" // JS/TS
            | "type_declaration" // Go
            => Self::TypeAlias,

            _ => Self::Unknown,
        }
    }

    /// Get a human-readable display name for this kind
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::Method => "method",
            Self::Class => "class",
            Self::Struct => "struct",
            Self::Interface => "interface",
            Self::Trait => "trait",
            Self::Enum => "enum",
            Self::Module => "module",
            Self::Variable => "variable",
            Self::Constant => "constant",
            Self::Parameter => "parameter",
            Self::Field => "field",
            Self::Import => "import",
            Self::Export => "export",
            Self::EnumVariant => "enum variant",
            Self::TypeAlias => "type alias",
            Self::Unknown => "unknown",
        }
    }
}

/// Visibility/access modifier for a symbol
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    /// Public - accessible from anywhere
    Public,
    /// Private - accessible only within the same scope
    #[default]
    Private,
    /// Protected - accessible within class hierarchy
    Protected,
    /// Internal/package-private
    Internal,
}

impl Visibility {
    /// Parse visibility from source code keywords
    pub fn from_keywords(text: &str) -> Self {
        let lower = text.to_lowercase();
        if lower.contains("pub ") || lower.contains("public ") || lower.contains("export ") {
            Self::Public
        } else if lower.contains("protected ") {
            Self::Protected
        } else if lower.contains("internal ") || lower.contains("package ") {
            Self::Internal
        } else {
            Self::Private
        }
    }
}

/// Kind of reference to a symbol
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceKind {
    /// Function or method call
    Call,
    /// Variable read access
    Read,
    /// Variable write/assignment
    Write,
    /// Import statement
    Import,
    /// Type annotation or type reference
    TypeReference,
    /// Class inheritance (extends/implements)
    Inheritance,
    /// Instantiation (new Foo())
    Instantiation,
    /// Unknown reference type
    Unknown,
}

/// A unique identifier for a symbol in the codebase.
///
/// Symbols are identified by their file path, name, kind, and position.
/// This allows distinguishing between symbols with the same name in different files
/// or different positions within the same file.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SymbolId {
    /// Relative file path from the project root
    pub file_path: String,
    /// Symbol name (e.g., function name, class name)
    pub name: String,
    /// Kind of symbol
    pub kind: SymbolKind,
    /// Starting line number (1-based)
    pub start_line: usize,
    /// Starting column (0-based)
    pub start_col: usize,
}

impl SymbolId {
    /// Create a new SymbolId
    pub fn new(
        file_path: impl Into<String>,
        name: impl Into<String>,
        kind: SymbolKind,
        start_line: usize,
        start_col: usize,
    ) -> Self {
        Self {
            file_path: file_path.into(),
            name: name.into(),
            kind,
            start_line,
            start_col,
        }
    }

    /// Generate a unique string ID for storage
    pub fn to_storage_id(&self) -> String {
        format!(
            "{}:{}:{}:{}",
            self.file_path, self.name, self.start_line, self.start_col
        )
    }

    /// Parse from a storage ID string
    pub fn from_storage_id(id: &str) -> Option<Self> {
        let parts: Vec<&str> = id.rsplitn(4, ':').collect();
        if parts.len() != 4 {
            return None;
        }
        // rsplitn gives parts in reverse order
        let start_col = parts[0].parse().ok()?;
        let start_line = parts[1].parse().ok()?;
        let name = parts[2].to_string();
        let file_path = parts[3].to_string();

        Some(Self {
            file_path,
            name,
            kind: SymbolKind::Unknown, // Kind not stored in ID
            start_line,
            start_col,
        })
    }
}

impl PartialEq for SymbolId {
    fn eq(&self, other: &Self) -> bool {
        self.file_path == other.file_path
            && self.name == other.name
            && self.start_line == other.start_line
            && self.start_col == other.start_col
    }
}

impl Eq for SymbolId {}

impl Hash for SymbolId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.file_path.hash(state);
        self.name.hash(state);
        self.start_line.hash(state);
        self.start_col.hash(state);
    }
}

/// A definition of a symbol in the codebase.
///
/// Contains full information about where a symbol is defined,
/// its signature, documentation, and relationships.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Definition {
    /// Unique identifier for this symbol
    pub symbol_id: SymbolId,
    /// Absolute root path of the indexed codebase
    pub root_path: Option<String>,
    /// Project name (for multi-project support)
    pub project: Option<String>,
    /// Ending line number (1-based)
    pub end_line: usize,
    /// Ending column (0-based)
    pub end_col: usize,
    /// Full signature or declaration text
    pub signature: String,
    /// Documentation comment if available
    pub doc_comment: Option<String>,
    /// Visibility modifier
    pub visibility: Visibility,
    /// Parent symbol ID (e.g., containing class for a method)
    pub parent_id: Option<String>,
    /// Timestamp when this definition was indexed
    pub indexed_at: i64,
}

impl Definition {
    /// Generate a unique storage ID for this definition
    pub fn to_storage_id(&self) -> String {
        format!(
            "def:{}:{}:{}",
            self.symbol_id.file_path, self.symbol_id.name, self.symbol_id.start_line
        )
    }

    /// Get the file path
    pub fn file_path(&self) -> &str {
        &self.symbol_id.file_path
    }

    /// Get the symbol name
    pub fn name(&self) -> &str {
        &self.symbol_id.name
    }

    /// Get the symbol kind
    pub fn kind(&self) -> SymbolKind {
        self.symbol_id.kind
    }

    /// Get the start line
    pub fn start_line(&self) -> usize {
        self.symbol_id.start_line
    }
}

/// A reference to a symbol from another location in the codebase.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Reference {
    /// File path where the reference occurs
    pub file_path: String,
    /// Absolute root path of the indexed codebase
    pub root_path: Option<String>,
    /// Project name
    pub project: Option<String>,
    /// Starting line number (1-based)
    pub start_line: usize,
    /// Ending line number (1-based)
    pub end_line: usize,
    /// Starting column (0-based)
    pub start_col: usize,
    /// Ending column (0-based)
    pub end_col: usize,
    /// Storage ID of the target symbol being referenced
    pub target_symbol_id: String,
    /// Kind of reference
    pub reference_kind: ReferenceKind,
    /// Timestamp when this reference was indexed
    pub indexed_at: i64,
}

impl Reference {
    /// Generate a unique storage ID for this reference
    pub fn to_storage_id(&self) -> String {
        format!(
            "ref:{}:{}:{}",
            self.file_path, self.start_line, self.start_col
        )
    }
}

/// An edge in the call graph representing a function/method call.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CallEdge {
    /// The symbol making the call (caller)
    pub caller_id: String,
    /// The symbol being called (callee)
    pub callee_id: String,
    /// File where the call occurs
    pub call_site_file: String,
    /// Line where the call occurs
    pub call_site_line: usize,
    /// Column where the call occurs
    pub call_site_col: usize,
}

/// Precision level of the relations provider
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionLevel {
    /// High precision: stack-graphs with full name resolution (~95% accuracy)
    High,
    /// Medium precision: AST-based with heuristic matching (~70% accuracy)
    Medium,
    /// Low precision: text-based pattern matching (~50% accuracy)
    Low,
}

impl PrecisionLevel {
    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::High => "high (stack-graphs)",
            Self::Medium => "medium (AST-based)",
            Self::Low => "low (text-based)",
        }
    }
}

// ============================================================================
// Result types for MCP tools
// ============================================================================

/// Result from find_definition containing the found definition
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DefinitionResult {
    /// File path where the definition is located
    pub file_path: String,
    /// Symbol name
    pub name: String,
    /// Symbol kind
    pub kind: SymbolKind,
    /// Starting line (1-based)
    pub start_line: usize,
    /// Ending line (1-based)
    pub end_line: usize,
    /// Starting column (0-based)
    pub start_col: usize,
    /// Ending column (0-based)
    pub end_col: usize,
    /// Full signature or declaration
    pub signature: String,
    /// Documentation comment
    pub doc_comment: Option<String>,
}

impl From<&Definition> for DefinitionResult {
    fn from(def: &Definition) -> Self {
        Self {
            file_path: def.symbol_id.file_path.clone(),
            name: def.symbol_id.name.clone(),
            kind: def.symbol_id.kind,
            start_line: def.symbol_id.start_line,
            end_line: def.end_line,
            start_col: def.symbol_id.start_col,
            end_col: def.end_col,
            signature: def.signature.clone(),
            doc_comment: def.doc_comment.clone(),
        }
    }
}

/// Result from find_references containing a found reference
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ReferenceResult {
    /// File path where the reference occurs
    pub file_path: String,
    /// Starting line (1-based)
    pub start_line: usize,
    /// Ending line (1-based)
    pub end_line: usize,
    /// Starting column (0-based)
    pub start_col: usize,
    /// Ending column (0-based)
    pub end_col: usize,
    /// Kind of reference
    pub reference_kind: ReferenceKind,
    /// Preview of the line containing the reference
    pub preview: Option<String>,
}

impl From<&Reference> for ReferenceResult {
    fn from(r: &Reference) -> Self {
        Self {
            file_path: r.file_path.clone(),
            start_line: r.start_line,
            end_line: r.end_line,
            start_col: r.start_col,
            end_col: r.end_col,
            reference_kind: r.reference_kind,
            preview: None,
        }
    }
}

/// A node in the call graph
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CallGraphNode {
    /// Symbol name
    pub name: String,
    /// Symbol kind
    pub kind: SymbolKind,
    /// File path
    pub file_path: String,
    /// Line number
    pub line: usize,
    /// Nested callers/callees (for depth > 1)
    pub children: Vec<CallGraphNode>,
}

/// Symbol info for call graph root
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SymbolInfo {
    /// Symbol name
    pub name: String,
    /// Symbol kind
    pub kind: SymbolKind,
    /// File path
    pub file_path: String,
    /// Starting line
    pub start_line: usize,
    /// Ending line
    pub end_line: usize,
    /// Signature
    pub signature: String,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_kind_from_ast_kind() {
        assert_eq!(SymbolKind::from_ast_kind("function_item"), SymbolKind::Function);
        assert_eq!(SymbolKind::from_ast_kind("class_definition"), SymbolKind::Class);
        assert_eq!(SymbolKind::from_ast_kind("method_definition"), SymbolKind::Method);
        assert_eq!(SymbolKind::from_ast_kind("unknown_node"), SymbolKind::Unknown);
    }

    #[test]
    fn test_symbol_kind_display_name() {
        assert_eq!(SymbolKind::Function.display_name(), "function");
        assert_eq!(SymbolKind::Class.display_name(), "class");
        assert_eq!(SymbolKind::Unknown.display_name(), "unknown");
    }

    #[test]
    fn test_visibility_from_keywords() {
        assert_eq!(Visibility::from_keywords("pub fn foo"), Visibility::Public);
        assert_eq!(Visibility::from_keywords("public void bar"), Visibility::Public);
        assert_eq!(Visibility::from_keywords("protected int x"), Visibility::Protected);
        assert_eq!(Visibility::from_keywords("fn private_func"), Visibility::Private);
    }

    #[test]
    fn test_symbol_id_equality() {
        let id1 = SymbolId::new("src/main.rs", "foo", SymbolKind::Function, 10, 0);
        let id2 = SymbolId::new("src/main.rs", "foo", SymbolKind::Function, 10, 0);
        let id3 = SymbolId::new("src/main.rs", "foo", SymbolKind::Function, 20, 0);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_symbol_id_hash() {
        use std::collections::HashSet;

        let id1 = SymbolId::new("src/main.rs", "foo", SymbolKind::Function, 10, 0);
        let id2 = SymbolId::new("src/main.rs", "foo", SymbolKind::Function, 10, 0);

        let mut set = HashSet::new();
        set.insert(id1);
        assert!(set.contains(&id2));
    }

    #[test]
    fn test_symbol_id_storage_id() {
        let id = SymbolId::new("src/main.rs", "foo", SymbolKind::Function, 10, 5);
        let storage_id = id.to_storage_id();
        assert_eq!(storage_id, "src/main.rs:foo:10:5");
    }

    #[test]
    fn test_definition_storage_id() {
        let def = Definition {
            symbol_id: SymbolId::new("src/lib.rs", "MyClass", SymbolKind::Class, 15, 0),
            root_path: Some("/project".to_string()),
            project: Some("test".to_string()),
            end_line: 50,
            end_col: 1,
            signature: "class MyClass".to_string(),
            doc_comment: None,
            visibility: Visibility::Public,
            parent_id: None,
            indexed_at: 12345,
        };

        assert_eq!(def.to_storage_id(), "def:src/lib.rs:MyClass:15");
        assert_eq!(def.file_path(), "src/lib.rs");
        assert_eq!(def.name(), "MyClass");
        assert_eq!(def.kind(), SymbolKind::Class);
    }

    #[test]
    fn test_reference_storage_id() {
        let reference = Reference {
            file_path: "src/consumer.rs".to_string(),
            root_path: None,
            project: None,
            start_line: 25,
            end_line: 25,
            start_col: 10,
            end_col: 20,
            target_symbol_id: "def:src/lib.rs:foo:10".to_string(),
            reference_kind: ReferenceKind::Call,
            indexed_at: 12345,
        };

        assert_eq!(reference.to_storage_id(), "ref:src/consumer.rs:25:10");
    }

    #[test]
    fn test_precision_level_description() {
        assert_eq!(PrecisionLevel::High.description(), "high (stack-graphs)");
        assert_eq!(PrecisionLevel::Medium.description(), "medium (AST-based)");
        assert_eq!(PrecisionLevel::Low.description(), "low (text-based)");
    }

    #[test]
    fn test_definition_result_from_definition() {
        let def = Definition {
            symbol_id: SymbolId::new("src/lib.rs", "my_func", SymbolKind::Function, 10, 0),
            root_path: None,
            project: None,
            end_line: 20,
            end_col: 1,
            signature: "fn my_func()".to_string(),
            doc_comment: Some("Does stuff".to_string()),
            visibility: Visibility::Public,
            parent_id: None,
            indexed_at: 0,
        };

        let result = DefinitionResult::from(&def);
        assert_eq!(result.file_path, "src/lib.rs");
        assert_eq!(result.name, "my_func");
        assert_eq!(result.kind, SymbolKind::Function);
        assert_eq!(result.start_line, 10);
        assert_eq!(result.end_line, 20);
        assert_eq!(result.doc_comment, Some("Does stuff".to_string()));
    }

    #[test]
    fn test_serialization() {
        let id = SymbolId::new("src/main.rs", "test", SymbolKind::Function, 1, 0);
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: SymbolId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }

    #[test]
    fn test_reference_kind_serialization() {
        let kind = ReferenceKind::Call;
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(json, "\"call\"");

        let deserialized: ReferenceKind = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, ReferenceKind::Call);
    }
}
