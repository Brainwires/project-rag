//! Programming language detection from file extensions

/// Detect programming language from file extension
pub fn detect_language(extension: &str) -> Option<String> {
    let lang = match extension.to_lowercase().as_str() {
        // Programming languages
        "rs" => "Rust",
        "py" => "Python",
        "js" | "mjs" | "cjs" => "JavaScript",
        "ts" => "TypeScript",
        "jsx" => "JavaScript (JSX)",
        "tsx" => "TypeScript (TSX)",
        "java" => "Java",
        "cpp" | "cc" | "cxx" => "C++",
        "c" => "C",
        "h" | "hpp" => "C/C++ Header",
        "go" => "Go",
        "rb" => "Ruby",
        "php" => "PHP",
        "swift" => "Swift",
        "kt" | "kts" => "Kotlin",
        "scala" => "Scala",
        "sh" | "bash" => "Shell",
        "sql" => "SQL",

        // Web technologies
        "html" | "htm" => "HTML",
        "css" => "CSS",
        "scss" | "sass" => "SCSS",

        // Data formats and config files
        "json" => "JSON",
        "yaml" | "yml" => "YAML",
        "toml" => "TOML",
        "xml" => "XML",
        "ini" => "INI",
        "conf" | "config" | "cfg" => "Config",
        "properties" => "Properties",
        "env" => "Environment",

        // Documentation formats
        "md" | "markdown" => "Markdown",
        "rst" => "reStructuredText",
        "adoc" | "asciidoc" => "AsciiDoc",
        "org" => "Org Mode",
        "txt" => "Text",
        "log" => "Log",
        "pdf" => "PDF",

        _ => return None,
    };

    Some(lang.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_language_rust() {
        assert_eq!(detect_language("rs"), Some("Rust".to_string()));
    }

    #[test]
    fn test_detect_language_python() {
        assert_eq!(detect_language("py"), Some("Python".to_string()));
    }

    #[test]
    fn test_detect_language_javascript() {
        assert_eq!(detect_language("js"), Some("JavaScript".to_string()));
        assert_eq!(detect_language("mjs"), Some("JavaScript".to_string()));
        assert_eq!(detect_language("cjs"), Some("JavaScript".to_string()));
    }

    #[test]
    fn test_detect_language_typescript() {
        assert_eq!(detect_language("ts"), Some("TypeScript".to_string()));
        assert_eq!(detect_language("tsx"), Some("TypeScript (TSX)".to_string()));
    }

    #[test]
    fn test_detect_language_jsx() {
        assert_eq!(detect_language("jsx"), Some("JavaScript (JSX)".to_string()));
    }

    #[test]
    fn test_detect_language_java() {
        assert_eq!(detect_language("java"), Some("Java".to_string()));
    }

    #[test]
    fn test_detect_language_cpp() {
        assert_eq!(detect_language("cpp"), Some("C++".to_string()));
        assert_eq!(detect_language("cc"), Some("C++".to_string()));
        assert_eq!(detect_language("cxx"), Some("C++".to_string()));
    }

    #[test]
    fn test_detect_language_c() {
        assert_eq!(detect_language("c"), Some("C".to_string()));
    }

    #[test]
    fn test_detect_language_headers() {
        assert_eq!(detect_language("h"), Some("C/C++ Header".to_string()));
        assert_eq!(detect_language("hpp"), Some("C/C++ Header".to_string()));
    }

    #[test]
    fn test_detect_language_go() {
        assert_eq!(detect_language("go"), Some("Go".to_string()));
    }

    #[test]
    fn test_detect_language_ruby() {
        assert_eq!(detect_language("rb"), Some("Ruby".to_string()));
    }

    #[test]
    fn test_detect_language_php() {
        assert_eq!(detect_language("php"), Some("PHP".to_string()));
    }

    #[test]
    fn test_detect_language_swift() {
        assert_eq!(detect_language("swift"), Some("Swift".to_string()));
    }

    #[test]
    fn test_detect_language_kotlin() {
        assert_eq!(detect_language("kt"), Some("Kotlin".to_string()));
        assert_eq!(detect_language("kts"), Some("Kotlin".to_string()));
    }

    #[test]
    fn test_detect_language_scala() {
        assert_eq!(detect_language("scala"), Some("Scala".to_string()));
    }

    #[test]
    fn test_detect_language_shell() {
        assert_eq!(detect_language("sh"), Some("Shell".to_string()));
        assert_eq!(detect_language("bash"), Some("Shell".to_string()));
    }

    #[test]
    fn test_detect_language_sql() {
        assert_eq!(detect_language("sql"), Some("SQL".to_string()));
    }

    #[test]
    fn test_detect_language_html() {
        assert_eq!(detect_language("html"), Some("HTML".to_string()));
        assert_eq!(detect_language("htm"), Some("HTML".to_string()));
    }

    #[test]
    fn test_detect_language_css() {
        assert_eq!(detect_language("css"), Some("CSS".to_string()));
        assert_eq!(detect_language("scss"), Some("SCSS".to_string()));
        assert_eq!(detect_language("sass"), Some("SCSS".to_string()));
    }

    #[test]
    fn test_detect_language_json() {
        assert_eq!(detect_language("json"), Some("JSON".to_string()));
    }

    #[test]
    fn test_detect_language_yaml() {
        assert_eq!(detect_language("yaml"), Some("YAML".to_string()));
        assert_eq!(detect_language("yml"), Some("YAML".to_string()));
    }

    #[test]
    fn test_detect_language_toml() {
        assert_eq!(detect_language("toml"), Some("TOML".to_string()));
    }

    #[test]
    fn test_detect_language_xml() {
        assert_eq!(detect_language("xml"), Some("XML".to_string()));
    }

    #[test]
    fn test_detect_language_markdown() {
        assert_eq!(detect_language("md"), Some("Markdown".to_string()));
        assert_eq!(detect_language("markdown"), Some("Markdown".to_string()));
    }

    #[test]
    fn test_detect_language_text() {
        assert_eq!(detect_language("txt"), Some("Text".to_string()));
    }

    #[test]
    fn test_detect_language_config_files() {
        assert_eq!(detect_language("ini"), Some("INI".to_string()));
        assert_eq!(detect_language("conf"), Some("Config".to_string()));
        assert_eq!(detect_language("config"), Some("Config".to_string()));
        assert_eq!(detect_language("cfg"), Some("Config".to_string()));
        assert_eq!(
            detect_language("properties"),
            Some("Properties".to_string())
        );
        assert_eq!(detect_language("env"), Some("Environment".to_string()));
    }

    #[test]
    fn test_detect_language_documentation() {
        assert_eq!(detect_language("rst"), Some("reStructuredText".to_string()));
        assert_eq!(detect_language("adoc"), Some("AsciiDoc".to_string()));
        assert_eq!(detect_language("asciidoc"), Some("AsciiDoc".to_string()));
        assert_eq!(detect_language("org"), Some("Org Mode".to_string()));
        assert_eq!(detect_language("log"), Some("Log".to_string()));
        assert_eq!(detect_language("pdf"), Some("PDF".to_string()));
    }

    #[test]
    fn test_detect_language_case_insensitive() {
        assert_eq!(detect_language("RS"), Some("Rust".to_string()));
        assert_eq!(detect_language("Py"), Some("Python".to_string()));
        assert_eq!(detect_language("JS"), Some("JavaScript".to_string()));
        assert_eq!(detect_language("TOML"), Some("TOML".to_string()));
        assert_eq!(detect_language("CONF"), Some("Config".to_string()));
    }

    #[test]
    fn test_detect_language_unknown() {
        assert_eq!(detect_language("unknown"), None);
        assert_eq!(detect_language("xyz"), None);
        assert_eq!(detect_language(""), None);
    }
}
