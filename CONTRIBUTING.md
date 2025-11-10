# Contributing to Project RAG

Thank you for your interest in contributing to Project RAG! We welcome contributions from the community.

## How to Contribute

### 1. Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/project-rag.git
   cd project-rag
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/project-rag.git
   ```

### 2. Create a Branch

Create a new branch for your changes:
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 3. Make Your Changes

Follow these guidelines when making changes:

#### Code Quality Requirements

- **Format your code**: Run `cargo fmt` before committing
- **Lint your code**: Run `cargo clippy` and fix all warnings
- **Test your changes**: Run `cargo test --lib` and ensure all tests pass
- **File size limit**: Keep source files under 600 lines (enforced)

#### Testing Requirements

- Add unit tests for new functionality
- Ensure existing tests still pass
- Use `#[cfg(test)]` modules in the same file as the code being tested
- Example:
  ```rust
  #[cfg(test)]
  mod tests {
      use super::*;

      #[test]
      fn test_your_feature() {
          // Your test here
      }
  }
  ```

#### Code Style

- Use descriptive variable and function names
- Add doc comments for public APIs
- Use `anyhow::Result` for error handling
- Add `.context("Description")` to provide error context
- Follow existing patterns in the codebase

### 4. Commit Your Changes

Write clear, descriptive commit messages:
```bash
git add .
git commit -m "Add feature: description of your changes"
```

Good commit message examples:
- `Add support for Python language detection`
- `Fix binary file detection threshold`
- `Update FastEmbed to use safe Mutex pattern`
- `Add tests for sliding window chunking`

### 5. Push and Create Pull Request

1. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Go to GitHub and create a Pull Request from your fork to the main repository

3. In your PR description:
   - Describe what changes you made and why
   - Reference any related issues (e.g., "Fixes #123")
   - Include any testing you performed
   - Note any breaking changes

### 6. Code Review

- Respond to feedback from maintainers
- Make requested changes by pushing new commits to your branch
- Your PR will be merged once approved

## Pull Request Checklist

Before submitting your PR, make sure:

- [ ] Code is formatted with `cargo fmt`
- [ ] No warnings from `cargo clippy`
- [ ] All tests pass with `cargo test --lib`
- [ ] New functionality includes tests
- [ ] Source files are under 600 lines
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the changes

## What to Contribute

We welcome contributions in these areas:

### Good First Issues

- Adding support for new programming languages
- Improving documentation
- Adding more unit tests
- Fixing typos or clarifying comments

### Feature Ideas

- Alternative embedding models
- AST-based code chunking
- Configuration file support (TOML)
- Persistent hash cache for incremental updates
- Additional MCP tools or prompts

### Bug Fixes

- Any bugs you encounter while using the project
- Performance improvements
- Memory optimization

## Development Setup

### Prerequisites

1. Install Rust (1.83+ with Rust 2024 edition support):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. Start Qdrant:
   ```bash
   docker run -p 6333:6333 -p 6334:6334 \
       -v $(pwd)/qdrant_data:/qdrant/storage \
       qdrant/qdrant
   ```

### Running Locally

```bash
# Build and run
cargo build
cargo run

# Run tests
cargo test --lib

# Check without building
cargo check

# Format code
cargo fmt

# Run lints
cargo clippy
```

## Questions or Issues?

- Open an issue on GitHub for bug reports or feature requests
- Use discussions for questions about using or contributing to the project

## License

By contributing to Project RAG, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing!
