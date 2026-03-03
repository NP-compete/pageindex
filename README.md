# PageIndex

[![CI](https://github.com/NP-compete/pageindex/actions/workflows/ci.yml/badge.svg)](https://github.com/NP-compete/pageindex/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Vectorless, reasoning-based RAG using hierarchical document indexing with Vertex AI**

> *Why chunk and embed when you can reason and structure?*

PageIndex builds semantic tree structures from documents without embeddings or vector databases. Instead of chunking and embedding, it uses LLM reasoning to extract hierarchical structure, making document navigation and retrieval more intuitive.

> **Note:** This is an independent implementation inspired by the [PageIndex framework](https://pageindex.ai/) by [VectifyAI](https://github.com/VectifyAI/PageIndex). While the original uses OpenAI, this implementation uses **Google Vertex AI (Gemini)** and adds features like batch processing, repository indexing, and CLI tooling.

<p align="center">
  <img src="https://img.shields.io/badge/PDF-Supported-green" alt="PDF">
  <img src="https://img.shields.io/badge/Markdown-Supported-green" alt="Markdown">
  <img src="https://img.shields.io/badge/DOCX-Via%20Docling-blue" alt="DOCX">
  <img src="https://img.shields.io/badge/Repository-Indexing-purple" alt="Repo">
</p>

## Features

- **PDF Processing** - Extracts table of contents, detects document structure, and builds hierarchical trees with page-level precision
- **Markdown Processing** - Parses header hierarchy into navigable tree structures
- **Batch Processing** - Process entire folders of documents concurrently
- **Repository Indexing** - Generate semantic summaries for codebases
- **Format Conversion** - Convert DOCX, PPTX, HTML, and images via [docling](https://github.com/DS4SD/docling)

## When NOT to Use PageIndex

PageIndex excels at structured, hierarchical documents but isn't the right tool for every use case:

| Use Case | Why PageIndex May Not Be Ideal | Better Alternative |
|----------|-------------------------------|-------------------|
| **Short documents** (< 10 pages) | Overhead of tree construction isn't worth it | Direct LLM context or simple chunking |
| **Unstructured content** (chat logs, social media) | No inherent hierarchy to extract | Vector search with semantic embeddings |
| **High-volume real-time queries** | LLM reasoning per query adds latency | Pre-computed vector indices |
| **Keyword/exact match search** | PageIndex focuses on semantic structure | Full-text search (Elasticsearch, etc.) |
| **Frequently updated documents** | Tree must be regenerated on each change | Incremental vector indexing |
| **Multi-document corpus search** | Designed for single-document navigation | Vector DB with cross-document retrieval |
| **Cost-sensitive applications** | Each indexing run uses LLM API calls | One-time embedding generation |

### PageIndex Shines When:

- Documents have **clear hierarchical structure** (reports, manuals, textbooks, legal docs)
- You need **explainable, traceable retrieval** with section/page references
- **Accuracy matters more than speed** (financial analysis, compliance, research)
- Documents are **long** (50+ pages) where vector chunking loses context
- You want **human-like navigation** through complex documents

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/NP-compete/pageindex.git
cd pageindex
pip install -e .
```

With document conversion support:

```bash
pip install -e ".[docling]"
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

### CLI Usage

Process a PDF:

```bash
pageindex pdf document.pdf --project-id your-gcp-project
```

Process a Markdown file:

```bash
pageindex md document.md --project-id your-gcp-project
```

Process all documents in a folder:

```bash
pageindex folder ./docs --project-id your-gcp-project
```

Index a code repository:

```bash
pageindex repo ./my-project --project-id your-gcp-project
```

### Python API

```python
from pageindex import page_index, md_to_tree, process_folder_sync, index_repository_sync

# Process a PDF
result = page_index(
    "document.pdf",
    project_id="your-gcp-project",
    model="gemini-1.5-flash",
)

# Process Markdown
import asyncio
from pageindex import md_to_tree, PageIndexConfig

config = PageIndexConfig(project_id="your-gcp-project")
result = asyncio.run(md_to_tree("document.md", config=config))

# Batch process a folder
result = process_folder_sync(
    "./docs",
    project_id="your-gcp-project",
    max_concurrent=5,
)

# Index a repository
result = index_repository_sync(
    "./my-project",
    project_id="your-gcp-project",
    add_summaries=True,
)
```

## Configuration

Set your Google Cloud project ID via environment variable:

```bash
export PAGEINDEX_PROJECT_ID=your-gcp-project
```

Or pass it directly to commands and functions.

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--project-id`, `-p` | Google Cloud project ID | `PAGEINDEX_PROJECT_ID` env |
| `--location`, `-l` | Vertex AI location | `us-central1` |
| `--model`, `-m` | Gemini model | `gemini-1.5-flash` |
| `--output`, `-o` | Output file/directory | `./results/` |
| `--add-summary/--no-summary` | Generate node summaries | varies by command |
| `--add-text/--no-text` | Include full text in nodes | `--no-text` |
| `--add-node-id/--no-node-id` | Add hierarchical node IDs | `--add-node-id` |

### PDF-Specific Options

| Option | Description | Default |
|--------|-------------|---------|
| `--toc-check-pages` | Pages to scan for TOC | `20` |
| `--max-pages-per-node` | Max pages before splitting | `10` |
| `--max-tokens-per-node` | Max tokens before splitting | `20000` |

### Folder Processing Options

| Option | Description | Default |
|--------|-------------|---------|
| `--max-concurrent`, `-c` | Concurrent processing tasks | `5` |
| `--convert/--no-convert` | Convert unsupported formats | `--convert` |
| `--docling-serve-url` | Remote docling-serve API URL | None |
| `--docling-serve-timeout` | API timeout (seconds) | `300` |

### Repository Indexing Options

| Option | Description | Default |
|--------|-------------|---------|
| `--summaries/--no-summaries` | Generate directory summaries | `--summaries` |
| `--include`, `-i` | File patterns to include | See defaults |
| `--exclude`, `-e` | Patterns to exclude | See defaults |
| `--max-depth` | Tree display depth | `4` |

## Output Format

PageIndex outputs JSON with a hierarchical structure:

```json
{
  "doc_name": "example",
  "doc_description": "A technical guide covering...",
  "structure": [
    {
      "title": "Introduction",
      "node_id": "0001",
      "summary": "Overview of the document...",
      "start_index": 1,
      "end_index": 5,
      "nodes": [
        {
          "title": "Background",
          "node_id": "0001.0001",
          "summary": "Historical context...",
          "start_index": 2,
          "end_index": 4
        }
      ]
    }
  ]
}
```

## Document Conversion

PageIndex supports converting various formats to Markdown using docling:

**Supported formats:** DOCX, PPTX, XLSX, HTML, PNG, JPG, TIFF, BMP

### Using docling-serve (recommended for production)

```bash
# Start docling-serve
docker run -p 5001:5001 quay.io/docling-project/docling-serve

# Process with remote conversion
pageindex folder ./docs --docling-serve-url http://localhost:5001
```

### Using local docling

```bash
pip install pageindex[docling]
pageindex folder ./docs --convert
```

## How It Works

### PDF Processing Pipeline

1. **TOC Detection** - Scans initial pages for table of contents
2. **Structure Extraction** - Uses LLM to extract hierarchical structure from TOC or content
3. **Page Mapping** - Maps logical sections to physical page numbers
4. **Verification** - Validates extracted structure against actual content
5. **Large Node Splitting** - Recursively splits oversized sections
6. **Summary Generation** - Optionally generates summaries for each node

### Markdown Processing

1. **Header Extraction** - Parses markdown headers (H1-H6)
2. **Tree Building** - Constructs hierarchy based on header levels
3. **Tree Thinning** - Optionally merges small nodes
4. **Summary Generation** - Optionally summarizes each section

### Repository Indexing

1. **Directory Scanning** - Walks repository respecting include/exclude patterns
2. **Context Building** - Reads README files and key entry points
3. **Summary Generation** - Uses LLM to summarize each directory's purpose
4. **Tree Construction** - Builds navigable directory tree with metadata

## Requirements

- Python 3.10+
- Google Cloud project with Vertex AI API enabled
- Authentication via `gcloud auth application-default login` or service account

## Related Projects

- **[PageIndex by VectifyAI](https://github.com/VectifyAI/PageIndex)** - The original PageIndex framework for vectorless, reasoning-based RAG using OpenAI
- **[PageIndex.ai](https://pageindex.ai/)** - Commercial platform for human-like document AI by VectifyAI

## License

MIT - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

This project is inspired by the [PageIndex framework](https://pageindex.ai/) developed by [VectifyAI](https://github.com/VectifyAI). Their research on vectorless, reasoning-based RAG demonstrates that **similarity ≠ relevance** — true document retrieval requires reasoning, not just embedding similarity.

## Author

Soham Dutta ([@NP-compete](https://github.com/NP-compete))
