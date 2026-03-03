# Contributing to PageIndex

First off, thank you for considering contributing to PageIndex! It's people like you that make PageIndex such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the [existing issues](https://github.com/NP-compete/pageindex/issues) to avoid duplicates.

When creating a bug report, please include:

- **Clear title** describing the issue
- **Steps to reproduce** the behavior
- **Expected behavior** vs what actually happened
- **Environment details** (Python version, OS, package version)
- **Code samples** or minimal reproducible examples
- **Error messages** and stack traces

### Suggesting Features

Feature requests are welcome! Please:

1. Check if the feature has already been requested
2. Open an issue with the `enhancement` label
3. Describe the feature and its use case
4. Explain why this would be useful to most users

### Pull Requests

1. **Fork the repo** and create your branch from `main`
2. **Install development dependencies**: `pip install -e ".[dev]"`
3. **Set up pre-commit hooks**: `pre-commit install`
4. **Make your changes** with clear, descriptive commits
5. **Add tests** for any new functionality
6. **Ensure tests pass**: `pytest tests/ -v`
7. **Ensure linting passes**: `ruff check src/ tests/`
8. **Update documentation** if needed
9. **Submit a pull request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/pageindex.git
cd pageindex

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Run linting
ruff check src/ tests/
ruff format src/ tests/
```

## Coding Standards

### Style Guide

- We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Follow PEP 8 conventions
- Use type hints for all function signatures
- Write docstrings for public functions and classes

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new PDF parsing option
fix: handle empty TOC gracefully
docs: update installation instructions
test: add tests for markdown processor
refactor: extract retry logic to helper
```

### Testing

- Write tests for all new functionality
- Maintain or improve code coverage
- Use pytest fixtures for common setup
- Mock external services (LLM API calls)

### Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Include examples for new features

## Project Structure

```
pageindex/
├── src/pageindex/          # Main package
│   ├── pdf/                # PDF processing
│   ├── markdown/           # Markdown processing
│   ├── cli.py              # CLI interface
│   ├── config.py           # Configuration
│   ├── llm.py              # LLM client
│   └── ...
├── tests/                  # Test suite
├── .github/workflows/      # CI/CD
└── pyproject.toml          # Project metadata
```

## Getting Help

- Open an [issue](https://github.com/NP-compete/pageindex/issues) for bugs or features
- Start a [discussion](https://github.com/NP-compete/pageindex/discussions) for questions

## Recognition

Contributors will be recognized in our release notes. Thank you for helping make PageIndex better!
