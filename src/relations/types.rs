//! Type definitions for code relationships (definitions, references, call graphs).
//!
//! This module provides the core data structures for representing code relationships:
//! - `SymbolId`: Unique identifier for a symbol in the codebase
//! - `Definition`: A symbol definition (function, class, method, etc.)
//! - `Reference`: A reference to a symbol
//! - `CallEdge`: An edge in the call graph

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::hash::{Hash, Hasher};

use crate::build_config::ConfigurationState;

/// Kind of symbol in the codebase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    /// A function (standalone)
    Function,
    /// A method (belongs to a class/struct/impl)
    Method,
    /// A constructor
    Constructor,
    /// A destructor
    Destructor,
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
    /// A language namespace
    Namespace,
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
    /// A preprocessor macro
    Macro,
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
            => Self::Method,

            "constructor_declaration" => Self::Constructor,

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

            // Imports
            "use_declaration" // Rust
            | "extern_crate_declaration" // Rust
            | "import_statement" // Python, JS/TS
            | "import_from_statement" // Python
            | "import_declaration" // Go, Java, Swift
            | "preproc_include" // C/C++
            | "using_directive" // C#
            | "namespace_use_declaration" // PHP
            => Self::Import,

            _ => Self::Unknown,
        }
    }

    /// Get a human-readable display name for this kind
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::Method => "method",
            Self::Constructor => "constructor",
            Self::Destructor => "destructor",
            Self::Class => "class",
            Self::Struct => "struct",
            Self::Interface => "interface",
            Self::Trait => "trait",
            Self::Enum => "enum",
            Self::Module => "module",
            Self::Namespace => "namespace",
            Self::Variable => "variable",
            Self::Constant => "constant",
            Self::Parameter => "parameter",
            Self::Field => "field",
            Self::Import => "import",
            Self::Export => "export",
            Self::EnumVariant => "enum variant",
            Self::TypeAlias => "type alias",
            Self::Macro => "macro",
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
    /// The symbol's defining source location
    Definition,
    /// A non-defining declaration
    Declaration,
    /// Function or method call
    Call,
    /// Constructor invocation
    ConstructorCall,
    /// Variable read access
    Read,
    /// Variable write/assignment
    Write,
    /// Read through an object/member selector.
    MemberRead,
    /// Write through an object/member selector.
    MemberWrite,
    /// Call through an object/member selector.
    MethodCall,
    /// Taking the address of a symbol
    AddressTake,
    /// Import statement
    Import,
    /// Type annotation or type reference
    TypeReference,
    /// Preferred spelling for a type dependency
    TypeUse,
    /// Class inheritance (extends/implements)
    Inheritance,
    /// Instantiation (new Foo())
    Instantiation,
    /// Object construction dependency.
    ObjectConstruction,
    /// Assignment involving an object value.
    ObjectAssignment,
    /// Statically visible ownership relationship.
    Owns,
    /// Statically visible reference relationship.
    References,
    /// Statically visible pointer relationship.
    PointsTo,
    /// Explicit creation relationship.
    Creates,
    /// Explicit destruction relationship.
    Destroys,
    /// Template or generic use
    TemplateUse,
    /// Include directive
    Include,
    /// Macro use
    MacroUse,
    /// Match inside a documentation comment
    Documentation,
    /// Match inside a non-documentation comment
    Comment,
    /// Match inside a string literal
    String,
    /// Unknown reference type
    Unknown,
}

impl ReferenceKind {
    /// Whether this kind is executable/source dependency evidence by default.
    pub fn is_code(self) -> bool {
        !matches!(self, Self::Documentation | Self::Comment | Self::String)
    }

    pub fn is_call(self) -> bool {
        matches!(
            self,
            Self::Call
                | Self::MethodCall
                | Self::ConstructorCall
                | Self::ObjectConstruction
                | Self::Creates
                | Self::Destroys
        )
    }
}

/// Whether a reference target has actually been established.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionStatus {
    Resolved,
    Ambiguous,
    #[default]
    Unresolved,
}

/// Nature of the evidence supporting a relation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    Semantic,
    Syntactic,
    #[default]
    Heuristic,
}

/// Dispatch form for call-like references.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum DispatchKind {
    Direct,
    Virtual,
    Indirect,
    Callback,
    Dynamic,
    #[default]
    Unknown,
}

/// Role of a concrete source location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum LocationRole {
    Declaration,
    #[default]
    Definition,
    Reference,
}

/// Linkage/scope discriminator used by logical symbol identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum LinkageKind {
    External,
    Internal,
    FileLocal,
    Local,
    Anonymous,
    #[default]
    Unknown,
}

/// A declaration, definition, or reference location, separate from logical identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct SourceLocation {
    #[serde(default)]
    pub project_id: String,
    pub file_path: String,
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
    pub role: LocationRole,
}

impl SourceLocation {
    pub fn to_storage_id(&self) -> String {
        let raw = format!(
            "{}\0{}\0{}\0{}\0{}\0{}\0{:?}",
            self.project_id,
            self.file_path,
            self.start_line,
            self.start_col,
            self.end_line,
            self.end_col,
            self.role
        );
        format!("loc:v3:{:x}", Sha256::digest(raw.as_bytes()))
    }
}

/// One plausible target retained for an ambiguous or unresolved reference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReferenceCandidate {
    pub symbol_id: String,
    pub reason: String,
}

/// A unique identifier for a symbol in the codebase.
///
/// Logical identity is independent of declaration/definition position. Location fields
/// remain as compatibility metadata; equality, hashing, and storage IDs use the logical
/// fields below.
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
    /// Stable persisted project identity (never the absolute root path).
    #[serde(default)]
    pub project_id: String,
    /// Parser language name.
    #[serde(default)]
    pub language: String,
    /// Namespace/type-qualified name.
    #[serde(default)]
    pub qualified_name: String,
    /// Normalized overload-disambiguating signature.
    #[serde(default)]
    pub canonical_signature: String,
    /// Linkage/scope category.
    #[serde(default)]
    pub linkage: LinkageKind,
    /// File/scope discriminator when linkage is not external.
    #[serde(default)]
    pub scope_discriminator: Option<String>,
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
        let file_path = file_path.into();
        Self {
            scope_discriminator: Some(file_path.clone()),
            file_path,
            name: name.into(),
            kind,
            start_line,
            start_col,
            project_id: String::new(),
            language: String::new(),
            qualified_name: String::new(),
            canonical_signature: String::new(),
            linkage: LinkageKind::FileLocal,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_logical(
        project_id: impl Into<String>,
        language: impl Into<String>,
        qualified_name: impl Into<String>,
        name: impl Into<String>,
        kind: SymbolKind,
        canonical_signature: impl Into<String>,
        linkage: LinkageKind,
        scope_discriminator: Option<String>,
        file_path: impl Into<String>,
        start_line: usize,
        start_col: usize,
    ) -> Self {
        Self {
            project_id: project_id.into(),
            language: language.into(),
            qualified_name: qualified_name.into(),
            name: name.into(),
            kind,
            canonical_signature: canonical_signature.into(),
            linkage,
            scope_discriminator,
            file_path: file_path.into(),
            start_line,
            start_col,
        }
    }

    /// Generate a unique string ID for storage
    pub fn to_storage_id(&self) -> String {
        let qualified = if self.qualified_name.is_empty() {
            &self.name
        } else {
            &self.qualified_name
        };
        let scope = self.scope_discriminator.as_deref().unwrap_or("");
        let raw = format!(
            "{}\0{}\0{}\0{:?}\0{}\0{:?}\0{}",
            self.project_id,
            self.language,
            qualified,
            self.kind,
            self.canonical_signature,
            self.linkage,
            scope
        );
        format!("sym:v3:{:x}:{}", Sha256::digest(raw.as_bytes()), self.name)
    }

    /// Parse from a storage ID string
    pub fn from_storage_id(id: &str) -> Option<Self> {
        let name = id.strip_prefix("sym:v3:")?.rsplit(':').next()?.to_string();
        Some(Self::new("", name, SymbolKind::Unknown, 0, 0))
    }
}

impl PartialEq for SymbolId {
    fn eq(&self, other: &Self) -> bool {
        self.to_storage_id() == other.to_storage_id()
    }
}

impl Eq for SymbolId {}

impl Hash for SymbolId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.to_storage_id().hash(state);
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
    /// Concrete declaration/definition location.
    pub location: SourceLocation,
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
    /// Parser which produced this evidence.
    pub parser: String,
    /// Timestamp when this definition was indexed
    pub indexed_at: i64,
}

impl Definition {
    /// Extract the symbol name from an id produced by [`Definition::to_storage_id`].
    ///
    /// Logical IDs use `sym:v3:<digest>:<simple-name>` so callers can show the
    /// simple name without treating a source location as identity.
    pub fn name_from_storage_id(id: &str) -> Option<&str> {
        let name = id.strip_prefix("sym:v3:")?.rsplit(':').next()?;
        if name.is_empty() { None } else { Some(name) }
    }

    /// Generate a unique storage ID for this definition
    pub fn to_storage_id(&self) -> String {
        self.symbol_id.to_storage_id()
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
    /// Stable identity for this occurrence.
    pub location_id: String,
    /// Enclosing source symbol, when known.
    pub source_symbol_id: Option<String>,
    /// Storage ID of the target symbol being referenced
    pub target_symbol_id: String,
    /// Exact identifier text used to query unresolved candidate sets.
    pub target_name: String,
    /// Plausible targets; retained without claiming resolution.
    #[serde(default)]
    pub candidates: Vec<ReferenceCandidate>,
    /// Kind of reference
    pub reference_kind: ReferenceKind,
    pub resolution_status: ResolutionStatus,
    pub evidence_kind: EvidenceKind,
    pub dispatch_kind: DispatchKind,
    /// Per-build-configuration preprocessor scope for this source occurrence.
    #[serde(default)]
    pub configuration_states: Vec<ConfigurationState>,
    pub language: String,
    pub parser: String,
    /// Timestamp when this reference was indexed
    pub indexed_at: i64,
}

impl Reference {
    /// Generate a unique storage ID for this reference
    pub fn to_storage_id(&self) -> String {
        self.location_id.clone()
    }

    pub fn from_definition(definition: &Definition) -> Self {
        let symbol_id = definition.to_storage_id();
        Self {
            file_path: definition.location.file_path.clone(),
            root_path: definition.root_path.clone(),
            project: definition.project.clone(),
            start_line: definition.location.start_line,
            end_line: definition.location.end_line,
            start_col: definition.location.start_col,
            end_col: definition.location.end_col,
            location_id: definition.location.to_storage_id(),
            source_symbol_id: None,
            target_symbol_id: symbol_id.clone(),
            target_name: definition.name().to_string(),
            candidates: vec![ReferenceCandidate {
                symbol_id,
                reason: "parser produced this declaration/definition".to_string(),
            }],
            reference_kind: match definition.location.role {
                LocationRole::Declaration => ReferenceKind::Declaration,
                _ => ReferenceKind::Definition,
            },
            resolution_status: ResolutionStatus::Resolved,
            evidence_kind: EvidenceKind::Syntactic,
            dispatch_kind: DispatchKind::Unknown,
            configuration_states: Vec::new(),
            language: definition.symbol_id.language.clone(),
            parser: definition.parser.clone(),
            indexed_at: definition.indexed_at,
        }
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
    pub reference_kind: ReferenceKind,
    pub resolution_status: ResolutionStatus,
    pub evidence_kind: EvidenceKind,
    pub parser: String,
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
    /// Stable logical symbol identity.
    pub symbol_id: String,
    /// Stable identity of this declaration/definition location.
    pub location_id: String,
    /// File path where the definition is located
    pub file_path: String,
    /// Symbol name
    pub name: String,
    /// Namespace/type-qualified name.
    pub qualified_name: String,
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
    pub location_role: LocationRole,
    pub language: String,
    pub canonical_signature: String,
    pub parser: String,
    pub resolution_status: ResolutionStatus,
    pub evidence_kind: EvidenceKind,
}

impl From<&Definition> for DefinitionResult {
    fn from(def: &Definition) -> Self {
        Self {
            symbol_id: def.to_storage_id(),
            location_id: def.location.to_storage_id(),
            file_path: def.symbol_id.file_path.clone(),
            name: def.symbol_id.name.clone(),
            qualified_name: def.symbol_id.qualified_name.clone(),
            kind: def.symbol_id.kind,
            start_line: def.symbol_id.start_line,
            end_line: def.end_line,
            start_col: def.symbol_id.start_col,
            end_col: def.end_col,
            signature: def.signature.clone(),
            doc_comment: def.doc_comment.clone(),
            location_role: def.location.role,
            language: def.symbol_id.language.clone(),
            canonical_signature: def.symbol_id.canonical_signature.clone(),
            parser: def.parser.clone(),
            resolution_status: ResolutionStatus::Resolved,
            evidence_kind: EvidenceKind::Syntactic,
        }
    }
}

/// Result from find_references containing a found reference
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ReferenceResult {
    pub location_id: String,
    pub source_symbol_id: Option<String>,
    pub target_symbol_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub candidates: Vec<ReferenceCandidate>,
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
    pub resolution_status: ResolutionStatus,
    pub evidence_kind: EvidenceKind,
    pub dispatch_kind: DispatchKind,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub configuration_states: Vec<ConfigurationState>,
    pub language: String,
    pub parser: String,
    /// Preview of the line containing the reference
    pub preview: Option<String>,
}

impl From<&Reference> for ReferenceResult {
    fn from(r: &Reference) -> Self {
        Self {
            location_id: r.location_id.clone(),
            source_symbol_id: r.source_symbol_id.clone(),
            target_symbol_id: (!r.target_symbol_id.is_empty()).then(|| r.target_symbol_id.clone()),
            candidates: r.candidates.clone(),
            file_path: r.file_path.clone(),
            start_line: r.start_line,
            end_line: r.end_line,
            start_col: r.start_col,
            end_col: r.end_col,
            reference_kind: r.reference_kind,
            resolution_status: r.resolution_status,
            evidence_kind: r.evidence_kind,
            dispatch_kind: r.dispatch_kind,
            configuration_states: r.configuration_states.clone(),
            language: r.language.clone(),
            parser: r.parser.clone(),
            preview: None,
        }
    }
}

/// A unique logical-symbol node in a dependency graph.
///
/// Location fields are optional because a trustworthy edge can occasionally
/// point at a logical symbol whose definition location was not extracted. Such
/// nodes stay visible rather than making the edge disappear.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CallGraphNode {
    pub symbol_id: String,
    pub location_id: Option<String>,
    pub name: String,
    pub qualified_name: String,
    pub kind: SymbolKind,
    pub file_path: Option<String>,
    pub start_line: Option<usize>,
    pub end_line: Option<usize>,
    pub signature: Option<String>,
    pub language: Option<String>,
    pub location_role: Option<LocationRole>,
    /// Shortest number of traversed edges from the requested root.
    pub distance: usize,
    pub definition_available: bool,
}

impl CallGraphNode {
    pub fn from_definition(definition: &Definition, distance: usize) -> Self {
        Self {
            symbol_id: definition.to_storage_id(),
            location_id: Some(definition.location.to_storage_id()),
            name: definition.symbol_id.name.clone(),
            qualified_name: definition.symbol_id.qualified_name.clone(),
            kind: definition.symbol_id.kind,
            file_path: Some(definition.symbol_id.file_path.clone()),
            start_line: Some(definition.symbol_id.start_line),
            end_line: Some(definition.end_line),
            signature: Some(definition.signature.clone()),
            language: Some(definition.symbol_id.language.clone()),
            location_role: Some(definition.location.role),
            distance,
            definition_available: true,
        }
    }

    pub fn without_definition(symbol_id: String, distance: usize) -> Self {
        let name = Definition::name_from_storage_id(&symbol_id)
            .unwrap_or(&symbol_id)
            .to_string();
        Self {
            symbol_id,
            location_id: None,
            name: name.clone(),
            qualified_name: name,
            kind: SymbolKind::Unknown,
            file_path: None,
            start_line: None,
            end_line: None,
            signature: None,
            language: None,
            location_role: None,
            distance,
            definition_available: false,
        }
    }
}

/// One source occurrence in graph form. An ambiguous or unresolved observation
/// has no authoritative target and is never traversed through its candidates.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GraphEdge {
    pub edge_id: String,
    pub source_symbol_id: String,
    pub target_symbol_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub candidates: Vec<ReferenceCandidate>,
    pub reference_kind: ReferenceKind,
    pub resolution_status: ResolutionStatus,
    pub evidence_kind: EvidenceKind,
    pub dispatch_kind: DispatchKind,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub configuration_states: Vec<ConfigurationState>,
    pub path: String,
    pub start_line: usize,
    pub start_column: usize,
    pub end_line: usize,
    pub end_column: usize,
    pub language: String,
    pub parser: String,
}

impl GraphEdge {
    pub fn from_reference(reference: &Reference) -> Option<Self> {
        let source_symbol_id = reference.source_symbol_id.clone()?;
        let target_symbol_id = (reference.resolution_status == ResolutionStatus::Resolved
            && !reference.target_symbol_id.is_empty())
        .then(|| reference.target_symbol_id.clone());
        Some(Self {
            edge_id: reference.location_id.clone(),
            source_symbol_id,
            target_symbol_id,
            candidates: reference.candidates.clone(),
            reference_kind: reference.reference_kind,
            resolution_status: reference.resolution_status,
            evidence_kind: reference.evidence_kind,
            dispatch_kind: reference.dispatch_kind,
            configuration_states: reference.configuration_states.clone(),
            path: reference.file_path.clone(),
            start_line: reference.start_line,
            start_column: reference.start_col,
            end_line: reference.end_line,
            end_column: reference.end_col,
            language: reference.language.clone(),
            parser: reference.parser.clone(),
        })
    }
}

/// Symbol info for call graph root
/// A node the extractor recognised as a definition but could not name, and
/// therefore left out of the symbol list.
///
/// Before this existed such nodes were dropped silently, so a caller had no way to
/// tell an empty or short symbol list from a complete one. A non-empty vector of
/// these means the listing is INCOMPLETE.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SkippedDefinition {
    /// 1-based line the definition starts on
    pub line: usize,
    /// tree-sitter node kind, e.g. "function_definition"
    pub kind: String,
    /// Why it was skipped
    pub reason: String,
    /// First line of the node text, truncated -- enough to identify it by eye
    pub snippet: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SymbolInfo {
    pub symbol_id: String,
    pub location_id: String,
    /// Symbol name
    pub name: String,
    pub qualified_name: String,
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
    pub language: String,
    pub location_role: LocationRole,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_kind_from_ast_kind() {
        assert_eq!(
            SymbolKind::from_ast_kind("function_item"),
            SymbolKind::Function
        );
        assert_eq!(
            SymbolKind::from_ast_kind("class_definition"),
            SymbolKind::Class
        );
        assert_eq!(
            SymbolKind::from_ast_kind("method_definition"),
            SymbolKind::Method
        );
        assert_eq!(
            SymbolKind::from_ast_kind("unknown_node"),
            SymbolKind::Unknown
        );
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
        assert_eq!(
            Visibility::from_keywords("public void bar"),
            Visibility::Public
        );
        assert_eq!(
            Visibility::from_keywords("protected int x"),
            Visibility::Protected
        );
        assert_eq!(
            Visibility::from_keywords("fn private_func"),
            Visibility::Private
        );
    }

    #[test]
    fn test_symbol_id_equality() {
        let id1 = SymbolId::new("src/main.rs", "foo", SymbolKind::Function, 10, 0);
        let id2 = SymbolId::new("src/main.rs", "foo", SymbolKind::Function, 10, 0);
        let id3 = SymbolId::new("src/main.rs", "foo", SymbolKind::Function, 20, 0);

        assert_eq!(id1, id2);
        // Source positions are locations, not logical symbol identity.
        assert_eq!(id1, id3);
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
        assert!(storage_id.starts_with("sym:v3:"));
        assert!(storage_id.ends_with(":foo"));
    }

    #[test]
    fn test_definition_storage_id() {
        let def = Definition {
            symbol_id: SymbolId::new("src/lib.rs", "MyClass", SymbolKind::Class, 15, 0),
            location: SourceLocation {
                project_id: String::new(),
                file_path: "src/lib.rs".to_string(),
                start_line: 15,
                start_col: 0,
                end_line: 15,
                end_col: 7,
                role: LocationRole::Definition,
            },
            root_path: Some("/project".to_string()),
            project: Some("test".to_string()),
            end_line: 50,
            end_col: 1,
            signature: "class MyClass".to_string(),
            doc_comment: None,
            visibility: Visibility::Public,
            parent_id: None,
            parser: "tree-sitter/test".to_string(),
            indexed_at: 12345,
        };

        assert!(def.to_storage_id().starts_with("sym:v3:"));
        assert_eq!(def.file_path(), "src/lib.rs");
        assert_eq!(def.name(), "MyClass");
        assert_eq!(def.kind(), SymbolKind::Class);
    }

    #[test]
    fn test_reference_storage_id() {
        let location = SourceLocation {
            project_id: String::new(),
            file_path: "src/consumer.rs".to_string(),
            start_line: 25,
            start_col: 10,
            end_line: 25,
            end_col: 20,
            role: LocationRole::Reference,
        };
        let reference = Reference {
            file_path: "src/consumer.rs".to_string(),
            root_path: None,
            project: None,
            start_line: 25,
            end_line: 25,
            start_col: 10,
            end_col: 20,
            location_id: location.to_storage_id(),
            source_symbol_id: None,
            target_symbol_id: "sym:v3:test:foo".to_string(),
            target_name: "foo".to_string(),
            candidates: Vec::new(),
            reference_kind: ReferenceKind::Call,
            resolution_status: ResolutionStatus::Resolved,
            evidence_kind: EvidenceKind::Syntactic,
            dispatch_kind: DispatchKind::Direct,
            configuration_states: Vec::new(),
            language: "Rust".to_string(),
            parser: "tree-sitter/test".to_string(),
            indexed_at: 12345,
        };

        assert_eq!(reference.to_storage_id(), location.to_storage_id());
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
            location: SourceLocation {
                project_id: String::new(),
                file_path: "src/lib.rs".to_string(),
                start_line: 10,
                start_col: 0,
                end_line: 10,
                end_col: 7,
                role: LocationRole::Definition,
            },
            root_path: None,
            project: None,
            end_line: 20,
            end_col: 1,
            signature: "fn my_func()".to_string(),
            doc_comment: Some("Does stuff".to_string()),
            visibility: Visibility::Public,
            parent_id: None,
            parser: "tree-sitter/test".to_string(),
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
