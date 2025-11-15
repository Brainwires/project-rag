use anyhow::{Context, Result};
use std::path::Path;

/// Extract text from a PDF file and convert to Markdown format
pub fn extract_pdf_to_markdown(path: &Path) -> Result<String> {
    // Extract text from PDF using pdf-extract
    let text = pdf_extract::extract_text(path).context("Failed to extract text from PDF")?;

    // Convert to markdown format
    let markdown = format_as_markdown(&text);

    Ok(markdown)
}

/// Format extracted PDF text as Markdown
/// This adds structure to the raw text extraction
fn format_as_markdown(text: &str) -> String {
    let mut markdown = String::new();
    let mut in_table = false;

    for line in text.lines() {
        let trimmed = line.trim();

        // Skip empty lines
        if trimmed.is_empty() {
            if in_table {
                markdown.push_str("\n");
                in_table = false;
            }
            markdown.push_str("\n");
            continue;
        }

        // Detect potential table rows (lines with multiple tab-separated or spaced columns)
        if is_likely_table_row(trimmed) {
            if !in_table {
                // Start table - add header separator
                markdown.push_str(&format_table_row(trimmed));
                markdown.push('\n');
                markdown.push_str(&create_table_separator(trimmed));
                markdown.push('\n');
                in_table = true;
            } else {
                markdown.push_str(&format_table_row(trimmed));
                markdown.push('\n');
            }
        } else {
            if in_table {
                markdown.push('\n');
                in_table = false;
            }

            // Detect headings (ALL CAPS lines or lines ending with :)
            if is_likely_heading(trimmed) {
                let level = if trimmed.len() < 30 { "##" } else { "###" };
                markdown.push_str(&format!("{} {}\n\n", level, trimmed.trim_end_matches(':')));
            } else {
                markdown.push_str(trimmed);
                markdown.push('\n');
            }
        }
    }

    markdown
}

/// Check if a line looks like a table row
fn is_likely_table_row(line: &str) -> bool {
    // Check for multiple whitespace-separated columns (3+ columns)
    let columns: Vec<&str> = line.split_whitespace().collect();
    if columns.len() >= 3 {
        // Check if columns have consistent spacing (indicating a table)
        let has_tabs = line.contains('\t');
        let has_multiple_spaces = line.contains("  ");
        return has_tabs || has_multiple_spaces;
    }
    false
}

/// Format a line as a markdown table row
fn format_table_row(line: &str) -> String {
    // Split by tabs or multiple spaces
    let columns: Vec<&str> = if line.contains('\t') {
        line.split('\t').map(|s| s.trim()).collect()
    } else {
        // Split by 2+ spaces
        line.split("  ")
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim())
            .collect()
    };

    format!("| {} |", columns.join(" | "))
}

/// Create a markdown table separator
fn create_table_separator(header: &str) -> String {
    let column_count = if header.contains('\t') {
        header.split('\t').count()
    } else {
        header.split("  ").filter(|s| !s.trim().is_empty()).count()
    };

    let separators = vec!["---"; column_count];
    format!("| {} |", separators.join(" | "))
}

/// Check if a line looks like a heading
fn is_likely_heading(line: &str) -> bool {
    // Check for ALL CAPS (at least 50% uppercase)
    let uppercase_count = line.chars().filter(|c| c.is_uppercase()).count();
    let alpha_count = line.chars().filter(|c| c.is_alphabetic()).count();

    if alpha_count > 0 {
        let uppercase_ratio = uppercase_count as f64 / alpha_count as f64;
        if uppercase_ratio > 0.8 && line.len() < 100 {
            return true;
        }
    }

    // Check for lines ending with colon (section headers)
    if line.ends_with(':') && line.len() < 80 && !line.contains("://") {
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_likely_table_row() {
        assert!(is_likely_table_row("Column1  Column2  Column3"));
        assert!(is_likely_table_row("Name\tAge\tCity"));
        assert!(!is_likely_table_row("This is a normal sentence"));
        assert!(!is_likely_table_row("Only two"));
    }

    #[test]
    fn test_format_table_row() {
        let row = "Name  Age  City";
        assert_eq!(format_table_row(row), "| Name | Age | City |");

        let tab_row = "Name\tAge\tCity";
        assert_eq!(format_table_row(tab_row), "| Name | Age | City |");
    }

    #[test]
    fn test_create_table_separator() {
        let header = "Name  Age  City";
        assert_eq!(create_table_separator(header), "| --- | --- | --- |");
    }

    #[test]
    fn test_is_likely_heading() {
        assert!(is_likely_heading("INTRODUCTION"));
        assert!(is_likely_heading("Chapter 1:"));
        assert!(is_likely_heading("Section Title:"));
        assert!(!is_likely_heading("This is a normal sentence"));
        assert!(!is_likely_heading("https://example.com"));
    }

    #[test]
    fn test_format_as_markdown_simple() {
        let text = "INTRODUCTION\n\nThis is some text.\n\nSection 1:\nMore text here.";
        let markdown = format_as_markdown(text);
        assert!(markdown.contains("## INTRODUCTION"));
        assert!(markdown.contains("## Section 1"));
    }

    #[test]
    fn test_format_as_markdown_with_table() {
        let text = "Name  Age  City\nJohn  30  NYC\nJane  25  LA";
        let markdown = format_as_markdown(text);
        assert!(markdown.contains("| Name | Age | City |"));
        assert!(markdown.contains("| --- | --- | --- |"));
        assert!(markdown.contains("| John | 30 | NYC |"));
    }
}
