use anyhow::{Context, Result};
use tree_sitter::{Language, Parser, Node};

/// AST node information for chunking
#[derive(Debug, Clone)]
pub struct AstNode {
    pub kind: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub end_line: usize,
}

/// AST parser for extracting semantic code units
pub struct AstParser {
    parser: Parser,
    _language: Language,
    language_name: String,
}

impl AstParser {
    /// Create a new AST parser for the given language
    pub fn new(extension: &str) -> Result<Self> {
        let (language, language_name) = match extension.to_lowercase().as_str() {
            "rs" => (tree_sitter_rust::LANGUAGE.into(), "Rust"),
            "py" => (tree_sitter_python::LANGUAGE.into(), "Python"),
            "js" | "mjs" | "cjs" | "jsx" => (tree_sitter_javascript::LANGUAGE.into(), "JavaScript"),
            "ts" | "tsx" => (tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(), "TypeScript"),
            "go" => (tree_sitter_go::LANGUAGE.into(), "Go"),
            "java" => (tree_sitter_java::LANGUAGE.into(), "Java"),
            "swift" => (tree_sitter_swift::LANGUAGE.into(), "Swift"),
            "c" | "h" => (tree_sitter_c::LANGUAGE.into(), "C"),
            "cpp" | "cc" | "cxx" | "hpp" | "hxx" | "hh" => (tree_sitter_cpp::LANGUAGE.into(), "C++"),
            "cs" => (tree_sitter_c_sharp::LANGUAGE.into(), "C#"),
            "rb" => (tree_sitter_ruby::LANGUAGE.into(), "Ruby"),
            "php" => (tree_sitter_php::LANGUAGE_PHP.into(), "PHP"),
            _ => anyhow::bail!("Unsupported language for AST parsing: {}", extension),
        };

        let mut parser = Parser::new();
        parser
            .set_language(&language)
            .context("Failed to set parser language")?;

        Ok(Self {
            parser,
            _language: language,
            language_name: language_name.to_string(),
        })
    }

    /// Parse source code and extract semantic units (functions, classes, etc.)
    pub fn parse(&mut self, source_code: &str) -> Result<Vec<AstNode>> {
        let tree = self
            .parser
            .parse(source_code, None)
            .context("Failed to parse source code")?;

        let root_node = tree.root_node();
        let mut nodes = Vec::new();

        // Extract semantic units based on language
        self.extract_semantic_units(root_node, source_code, &mut nodes);

        Ok(nodes)
    }

    /// Extract semantic units (functions, classes, methods) from the AST
    fn extract_semantic_units(&self, node: Node, source_code: &str, result: &mut Vec<AstNode>) {
        // Define node types we want to chunk by language
        let target_kinds = match self.language_name.as_str() {
            "Rust" => vec![
                "function_item",
                "impl_item",
                "trait_item",
                "struct_item",
                "enum_item",
                "mod_item",
            ],
            "Python" => vec![
                "function_definition",
                "class_definition",
                "decorated_definition",
            ],
            "JavaScript" | "TypeScript" => vec![
                "function_declaration",
                "function_expression",
                "arrow_function",
                "method_definition",
                "class_declaration",
            ],
            "Go" => vec![
                "function_declaration",
                "method_declaration",
                "type_declaration",
            ],
            "Java" => vec![
                "method_declaration",
                "class_declaration",
                "interface_declaration",
                "constructor_declaration",
            ],
            "Swift" => vec![
                "function_declaration",
                "class_declaration",
                "protocol_declaration",
                "struct_declaration",
                "enum_declaration",
                "extension_declaration",
                "deinit_declaration",
                "initializer_declaration",
                "subscript_declaration",
            ],
            "C" => vec![
                "function_definition",
                "struct_specifier",
                "enum_specifier",
                "union_specifier",
                "type_definition",
            ],
            "C++" => vec![
                "function_definition",
                "class_specifier",
                "struct_specifier",
                "enum_specifier",
                "union_specifier",
                "namespace_definition",
                "template_declaration",
            ],
            "C#" => vec![
                "method_declaration",
                "class_declaration",
                "struct_declaration",
                "interface_declaration",
                "enum_declaration",
                "namespace_declaration",
                "constructor_declaration",
                "property_declaration",
            ],
            "Ruby" => vec![
                "method",
                "singleton_method",
                "class",
                "singleton_class",
                "module",
            ],
            "PHP" => vec![
                "function_definition",
                "method_declaration",
                "class_declaration",
                "interface_declaration",
                "trait_declaration",
                "namespace_definition",
            ],
            _ => vec![],
        };

        // Check if current node is a target kind
        let kind = node.kind();
        if target_kinds.contains(&kind) {
            let start_position = node.start_position();
            let end_position = node.end_position();

            result.push(AstNode {
                kind: kind.to_string(),
                start_byte: node.start_byte(),
                end_byte: node.end_byte(),
                start_line: start_position.row + 1, // Tree-sitter uses 0-indexed rows
                end_line: end_position.row + 1,
            });
        }

        // Recursively process children
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.extract_semantic_units(child, source_code, result);
        }
    }

    /// Get the language name
    pub fn language_name(&self) -> &str {
        &self.language_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_parsing() {
        let source = r#"
fn main() {
    println!("Hello, world!");
}

struct MyStruct {
    field: i32,
}

impl MyStruct {
    fn new() -> Self {
        MyStruct { field: 0 }
    }
}
"#;

        let mut parser = AstParser::new("rs").unwrap();
        let nodes = parser.parse(source).unwrap();

        assert!(nodes.len() >= 3); // function, struct, impl
        assert!(nodes.iter().any(|n| n.kind == "function_item"));
        assert!(nodes.iter().any(|n| n.kind == "struct_item"));
        assert!(nodes.iter().any(|n| n.kind == "impl_item"));
    }

    #[test]
    fn test_python_parsing() {
        let source = r#"
def hello():
    print("Hello")

class MyClass:
    def __init__(self):
        self.value = 0

    def method(self):
        return self.value
"#;

        let mut parser = AstParser::new("py").unwrap();
        let nodes = parser.parse(source).unwrap();

        assert!(nodes.len() >= 2); // function and class
        assert!(nodes.iter().any(|n| n.kind == "function_definition"));
        assert!(nodes.iter().any(|n| n.kind == "class_definition"));
    }

    #[test]
    fn test_javascript_parsing() {
        let source = r#"
function hello() {
    console.log("Hello");
}

const arrow = () => {
    return 42;
};

class MyClass {
    constructor() {
        this.value = 0;
    }

    method() {
        return this.value;
    }
}
"#;

        let mut parser = AstParser::new("js").unwrap();
        let nodes = parser.parse(source).unwrap();

        assert!(nodes.len() >= 2); // At least function and class
    }

    #[test]
    fn test_swift_parsing() {
        let source = r#"
func greet(name: String) {
    print("Hello, \(name)!")
}

class MyClass {
    var value: Int

    init(value: Int) {
        self.value = value
    }

    func method() -> Int {
        return value
    }
}
"#;

        let mut parser = AstParser::new("swift").unwrap();
        let nodes = parser.parse(source).unwrap();

        // Swift parser should extract function and class declarations
        assert!(nodes.len() >= 1); // At least some declarations found
        // Check we can parse Swift without errors
        assert!(parser.language_name() == "Swift");
    }

    #[test]
    fn test_unsupported_language() {
        let result = AstParser::new("xyz");
        assert!(result.is_err());
    }

    #[test]
    fn test_c_parsing() {
        let source = r#"
int add(int a, int b) {
    return a + b;
}

struct Point {
    int x;
    int y;
};
"#;

        let mut parser = AstParser::new("c").unwrap();
        let nodes = parser.parse(source).unwrap();

        assert!(nodes.len() >= 1);
        assert!(parser.language_name() == "C");
    }

    #[test]
    fn test_cpp_parsing() {
        let source = r#"
class MyClass {
public:
    int value;
    MyClass() : value(0) {}
    int getValue() { return value; }
};

namespace MyNamespace {
    void function() {}
}
"#;

        let mut parser = AstParser::new("cpp").unwrap();
        let nodes = parser.parse(source).unwrap();

        assert!(nodes.len() >= 1);
        assert!(parser.language_name() == "C++");
    }

    #[test]
    fn test_csharp_parsing() {
        let source = r#"
class MyClass {
    private int value;

    public MyClass() {
        value = 0;
    }

    public int GetValue() {
        return value;
    }
}
"#;

        let mut parser = AstParser::new("cs").unwrap();
        let nodes = parser.parse(source).unwrap();

        assert!(nodes.len() >= 1);
        assert!(parser.language_name() == "C#");
    }

    #[test]
    fn test_ruby_parsing() {
        let source = r#"
def hello(name)
  puts "Hello, #{name}!"
end

class MyClass
  def initialize(value)
    @value = value
  end

  def method
    @value
  end
end
"#;

        let mut parser = AstParser::new("rb").unwrap();
        let nodes = parser.parse(source).unwrap();

        assert!(nodes.len() >= 1);
        assert!(parser.language_name() == "Ruby");
    }

    #[test]
    fn test_php_parsing() {
        let source = r#"
<?php
function hello($name) {
    echo "Hello, $name!";
}

class MyClass {
    private $value;

    public function __construct($value) {
        $this->value = $value;
    }

    public function getValue() {
        return $this->value;
    }
}
?>
"#;

        let mut parser = AstParser::new("php").unwrap();
        let nodes = parser.parse(source).unwrap();

        assert!(nodes.len() >= 1);
        assert!(parser.language_name() == "PHP");
    }
}
