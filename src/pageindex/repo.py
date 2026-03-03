"""Repository indexing for semantic code search."""

from __future__ import annotations

import asyncio
import fnmatch
import logging
from pathlib import Path
from typing import Any

from pageindex.config import PageIndexConfig
from pageindex.llm import LLMClient
from pageindex.tree import write_node_id

logger = logging.getLogger(__name__)

DEFAULT_INCLUDE_PATTERNS = [
    "*.py",
    "*.js",
    "*.ts",
    "*.tsx",
    "*.jsx",
    "*.go",
    "*.rs",
    "*.java",
    "*.kt",
    "*.scala",
    "*.rb",
    "*.php",
    "*.c",
    "*.cpp",
    "*.h",
    "*.hpp",
    "*.md",
    "*.rst",
    "*.txt",
    "*.yaml",
    "*.yml",
    "*.json",
    "*.toml",
    "Makefile",
    "Dockerfile",
    "*.sh",
]

DEFAULT_EXCLUDE_PATTERNS = [
    ".git/**",
    ".github/**",
    ".gitlab/**",
    "node_modules/**",
    "vendor/**",
    "venv/**",
    ".venv/**",
    "__pycache__/**",
    "*.pyc",
    "*.pyo",
    ".idea/**",
    ".vscode/**",
    ".cursor/**",
    "dist/**",
    "build/**",
    "target/**",
    "out/**",
    "*.min.js",
    "*.min.css",
    "*.map",
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "*.egg-info/**",
    "*.egg",
    ".DS_Store",
    "Thumbs.db",
    "coverage/**",
    ".coverage",
    "htmlcov/**",
    ".pytest_cache/**",
    ".mypy_cache/**",
    ".ruff_cache/**",
]

SUMMARY_FILE_NAMES = [
    "README.md",
    "README.rst",
    "README.txt",
    "README",
    "__init__.py",
    "index.ts",
    "index.js",
    "mod.rs",
    "lib.rs",
]


def _matches_any_pattern(path: Path, patterns: list[str], base_path: Path) -> bool:
    """Check if path matches any of the glob patterns."""
    rel_path = str(path.relative_to(base_path))
    for pattern in patterns:
        if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(path.name, pattern):
            return True
    return False


def _should_include_file(
    path: Path, include: list[str], exclude: list[str], base_path: Path
) -> bool:
    """Determine if a file should be included in the index."""
    if _matches_any_pattern(path, exclude, base_path):
        return False
    if _matches_any_pattern(path, include, base_path):
        return True
    return False


def _should_include_dir(path: Path, exclude: list[str], base_path: Path) -> bool:
    """Determine if a directory should be traversed."""
    if path.name.startswith("."):
        return False
    return not _matches_any_pattern(path, exclude, base_path)


def _get_file_preview(file_path: Path, max_lines: int = 50, max_chars: int = 2000) -> str:
    """Get a preview of file contents for summarization."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        lines = content.split("\n")[:max_lines]
        preview = "\n".join(lines)
        if len(preview) > max_chars:
            preview = preview[:max_chars] + "\n... (truncated)"
        return preview
    except Exception:
        return ""


def _get_directory_context(dir_path: Path, files: list[Path]) -> str:
    """Build context string for directory summarization."""
    context_parts = []

    for summary_name in SUMMARY_FILE_NAMES:
        summary_file = dir_path / summary_name
        if summary_file.exists() and summary_file.is_file():
            preview = _get_file_preview(summary_file, max_lines=30, max_chars=1500)
            if preview:
                context_parts.append(f"=== {summary_name} ===\n{preview}")
                break

    file_list = [f.name for f in files[:20]]
    if file_list:
        context_parts.append(f"Files: {', '.join(file_list)}")
        if len(files) > 20:
            context_parts.append(f"... and {len(files) - 20} more files")

    return "\n\n".join(context_parts)


def scan_repository(
    repo_path: Path,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> dict[str, Any]:
    """Scan repository and build directory tree structure.

    Args:
        repo_path: Path to the repository root
        include_patterns: Glob patterns for files to include
        exclude_patterns: Glob patterns for files/dirs to exclude

    Returns:
        Dictionary with directory tree structure
    """
    if include_patterns is None:
        include_patterns = DEFAULT_INCLUDE_PATTERNS
    if exclude_patterns is None:
        exclude_patterns = DEFAULT_EXCLUDE_PATTERNS

    repo_path = Path(repo_path).resolve()

    def build_tree(current_path: Path) -> dict[str, Any] | None:
        if not current_path.is_dir():
            return None

        if not _should_include_dir(current_path, exclude_patterns, repo_path):
            return None

        rel_path = current_path.relative_to(repo_path)
        node: dict[str, Any] = {
            "title": current_path.name or repo_path.name,
            "path": str(rel_path) if str(rel_path) != "." else "",
            "type": "directory",
            "files": [],
            "nodes": [],
        }

        try:
            entries = sorted(current_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return None

        for entry in entries:
            if entry.is_dir():
                child_node = build_tree(entry)
                if child_node and (child_node["files"] or child_node["nodes"]):
                    node["nodes"].append(child_node)
            elif entry.is_file():
                if _should_include_file(entry, include_patterns, exclude_patterns, repo_path):
                    node["files"].append(
                        {
                            "name": entry.name,
                            "path": str(entry.relative_to(repo_path)),
                        }
                    )

        if not node["files"] and not node["nodes"]:
            return None

        return node

    tree = build_tree(repo_path)
    if tree is None:
        tree = {
            "title": repo_path.name,
            "path": "",
            "type": "directory",
            "files": [],
            "nodes": [],
        }

    return tree


async def _generate_directory_summary(
    node: dict[str, Any],
    repo_path: Path,
    llm: LLMClient,
) -> str:
    """Generate a summary for a directory based on its contents."""
    dir_path = repo_path / node["path"] if node["path"] else repo_path
    files = [repo_path / f["path"] for f in node.get("files", [])]

    context = _get_directory_context(dir_path, files)
    if not context:
        return ""

    child_summaries = []
    for child in node.get("nodes", []):
        if child.get("summary"):
            child_summaries.append(f"- {child['title']}: {child['summary']}")

    if child_summaries:
        context += "\n\nSubdirectories:\n" + "\n".join(child_summaries[:10])

    prompt = f"""Summarize this code directory in 1-2 sentences. Focus on:
- What this module/package does
- Key functionality or purpose
- Main technologies/frameworks used

Directory: {node["title"]}

Context:
{context}

Respond with ONLY the summary, no explanations or prefixes."""

    try:
        summary = await llm.chat_async(prompt, max_retries=3)
        return summary.strip()
    except Exception as e:
        logger.warning(f"Failed to generate summary for {node['path']}: {e}")
        return ""


async def _add_summaries_recursive(
    node: dict[str, Any],
    repo_path: Path,
    llm: LLMClient,
    max_concurrent: int = 5,
) -> None:
    """Recursively add summaries to all directories (bottom-up)."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_node(n: dict[str, Any]) -> None:
        for child in n.get("nodes", []):
            await process_node(child)

        async with semaphore:
            summary = await _generate_directory_summary(n, repo_path, llm)
            n["summary"] = summary

    await process_node(node)


def _flatten_structure(node: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten tree structure for easier output."""
    result = []

    flat_node = {
        "title": node["title"],
        "path": node["path"],
        "type": node["type"],
        "summary": node.get("summary", ""),
        "file_count": len(node.get("files", [])),
        "subdir_count": len(node.get("nodes", [])),
    }
    result.append(flat_node)

    for child in node.get("nodes", []):
        result.extend(_flatten_structure(child))

    return result


async def index_repository(
    repo_path: str | Path,
    config: PageIndexConfig,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    add_summaries: bool = True,
    max_concurrent: int = 5,
) -> dict[str, Any]:
    """Index a repository and generate semantic structure.

    Args:
        repo_path: Path to the repository
        config: PageIndex configuration
        include_patterns: File patterns to include
        exclude_patterns: File/directory patterns to exclude
        add_summaries: Whether to generate LLM summaries
        max_concurrent: Max concurrent LLM calls

    Returns:
        Dictionary with repository structure and summaries
    """
    repo_path = Path(repo_path).resolve()

    if not repo_path.is_dir():
        raise ValueError(f"Not a directory: {repo_path}")

    tree = scan_repository(repo_path, include_patterns, exclude_patterns)

    if add_summaries and config.project_id:
        llm = LLMClient(config)
        await _add_summaries_recursive(tree, repo_path, llm, max_concurrent)

    write_node_id(tree)

    total_files = sum(1 for _ in _count_files(tree))
    total_dirs = sum(1 for _ in _count_dirs(tree))

    return {
        "repo_name": repo_path.name,
        "repo_path": str(repo_path),
        "structure": [tree] if tree else [],
        "statistics": {
            "total_files": total_files,
            "total_directories": total_dirs,
        },
        "flat_index": _flatten_structure(tree),
    }


def _count_files(node: dict[str, Any]):
    """Generator to count all files in tree."""
    yield from node.get("files", [])
    for child in node.get("nodes", []):
        yield from _count_files(child)


def _count_dirs(node: dict[str, Any]):
    """Generator to count all directories in tree."""
    yield node
    for child in node.get("nodes", []):
        yield from _count_dirs(child)


def index_repository_sync(
    repo_path: str | Path,
    config: PageIndexConfig | None = None,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    add_summaries: bool = True,
    max_concurrent: int = 5,
    project_id: str | None = None,
    location: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Synchronous wrapper for index_repository.

    Args:
        repo_path: Path to the repository
        config: PageIndex configuration (optional if project_id provided)
        include_patterns: File patterns to include
        exclude_patterns: File/directory patterns to exclude
        add_summaries: Whether to generate LLM summaries
        max_concurrent: Max concurrent LLM calls
        project_id: Google Cloud project ID (alternative to config)
        location: Vertex AI location
        model: Gemini model to use

    Returns:
        Dictionary with repository structure
    """
    if config is None:
        config = PageIndexConfig(
            project_id=project_id or "",
            location=location or "us-central1",
            model=model or "gemini-1.5-flash",
        )

    return asyncio.run(
        index_repository(
            repo_path=repo_path,
            config=config,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            add_summaries=add_summaries,
            max_concurrent=max_concurrent,
        )
    )
