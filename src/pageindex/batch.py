"""Batch processing for multiple documents in a folder."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console

from pageindex.config import PageIndexConfig
from pageindex.llm import LLMClient

console = Console()
logger = logging.getLogger(__name__)

# Supported formats
NATIVE_FORMATS = {".pdf", ".md", ".markdown"}
DOCLING_FORMATS = {
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".bmp",
}


class DoclingServeClient:
    """Client for docling-serve API."""

    def __init__(self, base_url: str, timeout: int = 300):
        """Initialize the docling-serve client.

        Args:
            base_url: Base URL of docling-serve (e.g., http://localhost:5001)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> DoclingServeClient:
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if docling-serve is available."""
        try:
            if self._client is None:
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.get(f"{self.base_url}/health")
                    return response.status_code == 200
            else:
                response = await self._client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False

    async def convert_file(self, file_path: Path) -> str | None:
        """Convert a file to Markdown using docling-serve.

        Args:
            file_path: Path to the file to convert

        Returns:
            Markdown content as string, or None if conversion failed
        """
        if self._client is None:
            raise RuntimeError("Client not initialized. Use async with context manager.")

        try:
            file_content = file_path.read_bytes()
            file_base64 = base64.standard_b64encode(file_content).decode("utf-8")

            payload = {
                "sources": [
                    {
                        "kind": "base64",
                        "base64": file_base64,
                        "filename": file_path.name,
                    }
                ],
                "options": {
                    "to_markdown": True,
                },
            }

            response = await self._client.post(
                f"{self.base_url}/v1/convert/source",
                json=payload,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()

            result = response.json()

            if "document" in result and "md_content" in result["document"]:
                return result["document"]["md_content"]

            if isinstance(result, list) and len(result) > 0:
                doc = result[0]
                if "document" in doc and "md_content" in doc["document"]:
                    return doc["document"]["md_content"]

            logger.warning(f"Unexpected response format from docling-serve: {list(result.keys())}")
            return None

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error from docling-serve: {e.response.status_code} - {e.response.text}"
            )
            return None
        except Exception as e:
            logger.error(f"Error converting {file_path.name} with docling-serve: {e}")
            return None

    async def convert_url(self, url: str) -> str | None:
        """Convert a document from URL to Markdown.

        Args:
            url: URL of the document to convert

        Returns:
            Markdown content as string, or None if conversion failed
        """
        if self._client is None:
            raise RuntimeError("Client not initialized. Use async with context manager.")

        try:
            payload = {
                "sources": [{"kind": "http", "url": url}],
                "options": {
                    "to_markdown": True,
                },
            }

            response = await self._client.post(
                f"{self.base_url}/v1/convert/source",
                json=payload,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()

            result = response.json()

            if "document" in result and "md_content" in result["document"]:
                return result["document"]["md_content"]

            return None

        except Exception as e:
            logger.error(f"Error converting URL {url} with docling-serve: {e}")
            return None


def _check_docling_available() -> bool:
    """Check if docling is installed locally."""
    try:
        import importlib.util

        return importlib.util.find_spec("docling") is not None
    except ImportError:
        return False


def _convert_with_docling_local(file_path: Path, output_dir: Path) -> Path | None:
    """Convert a document to Markdown using local docling.

    Args:
        file_path: Path to the source document
        output_dir: Directory to save the converted markdown

    Returns:
        Path to the converted markdown file, or None if conversion failed
    """
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        markdown_content = result.document.export_to_markdown()

        output_path = output_dir / f"{file_path.stem}.md"
        output_path.write_text(markdown_content, encoding="utf-8")

        return output_path
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to convert {file_path.name} locally: {e}[/yellow]")
        return None


async def _convert_with_docling_serve(
    file_path: Path,
    output_dir: Path,
    client: DoclingServeClient,
) -> Path | None:
    """Convert a document to Markdown using docling-serve.

    Args:
        file_path: Path to the source document
        output_dir: Directory to save the converted markdown
        client: DoclingServeClient instance

    Returns:
        Path to the converted markdown file, or None if conversion failed
    """
    try:
        markdown_content = await client.convert_file(file_path)
        if markdown_content:
            output_path = output_dir / f"{file_path.stem}.md"
            output_path.write_text(markdown_content, encoding="utf-8")
            return output_path
        return None
    except Exception as e:
        console.print(
            f"[yellow]Warning: Failed to convert {file_path.name} via docling-serve: {e}[/yellow]"
        )
        return None


def get_supported_files(folder: Path) -> dict[str, list[Path]]:
    """Get all supported files in a folder, categorized by processing type.

    Args:
        folder: Path to the folder to scan

    Returns:
        Dictionary with keys 'pdf', 'markdown', 'docling', 'unsupported'
    """
    files: dict[str, list[Path]] = {
        "pdf": [],
        "markdown": [],
        "docling": [],
        "unsupported": [],
    }

    for file_path in folder.iterdir():
        if file_path.is_file():
            suffix = file_path.suffix.lower()
            if suffix == ".pdf":
                files["pdf"].append(file_path)
            elif suffix in {".md", ".markdown"}:
                files["markdown"].append(file_path)
            elif suffix in DOCLING_FORMATS:
                files["docling"].append(file_path)
            else:
                files["unsupported"].append(file_path)

    return files


async def process_pdf_async(
    file_path: Path,
    config: PageIndexConfig,
    llm: LLMClient,
) -> dict[str, Any]:
    """Process a single PDF file asynchronously."""
    from pageindex.pdf.processor import page_index_main

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: page_index_main(str(file_path), config))
    return result


async def process_markdown_async(
    file_path: Path,
    config: PageIndexConfig,
) -> dict[str, Any]:
    """Process a single Markdown file asynchronously."""
    from pageindex.markdown.processor import md_to_tree

    result = await md_to_tree(
        md_path=file_path,
        config=config,
        if_add_node_summary=config.if_add_node_summary,
        if_add_doc_description=config.if_add_doc_description,
        if_add_node_text=config.if_add_node_text,
        if_add_node_id=config.if_add_node_id,
    )
    return result


async def process_folder(
    folder: str | Path,
    config: PageIndexConfig,
    output_dir: str | Path | None = None,
    max_concurrent: int = 5,
    convert_unsupported: bool = True,
    on_progress: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Process all documents in a folder.

    Args:
        folder: Path to the folder containing documents
        config: PageIndex configuration
        output_dir: Directory to save results (defaults to ./results)
        max_concurrent: Maximum number of concurrent processing tasks
        convert_unsupported: Whether to use docling for unsupported formats
        on_progress: Optional callback for progress updates (file_name, status)

    Returns:
        Dictionary with processing results and statistics

    Note:
        Document conversion priority:
        1. If config.docling_serve_url is set, use docling-serve API
        2. Else if local docling is installed, use local docling
        3. Else skip unsupported formats
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    if output_dir is None:
        output_dir = Path("./results")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = get_supported_files(folder)

    total_files = len(files["pdf"]) + len(files["markdown"])
    if convert_unsupported:
        total_files += len(files["docling"])

    if total_files == 0:
        return {
            "success": [],
            "failed": [],
            "skipped": [str(f) for f in files["unsupported"]],
            "statistics": {
                "total": 0,
                "success": 0,
                "failed": 0,
                "skipped": len(files["unsupported"]),
            },
        }

    llm = LLMClient(config)
    results: dict[str, Any] = {
        "success": [],
        "failed": [],
        "skipped": [],
        "conversion_method": None,
    }

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(
        file_path: Path,
        processor: Callable,
        file_type: str,
        original_file: Path | None = None,
    ) -> tuple[Path, Path | None, dict | None, str | None]:
        async with semaphore:
            try:
                if on_progress:
                    on_progress(file_path.name, "processing")

                if file_type == "pdf":
                    result = await process_pdf_async(file_path, config, llm)
                else:
                    result = await process_markdown_async(file_path, config)

                stem = original_file.stem if original_file else file_path.stem
                output_file = output_dir / f"{stem}_structure.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                if on_progress:
                    on_progress(file_path.name, "done")

                return file_path, original_file, result, None
            except Exception as e:
                if on_progress:
                    on_progress(file_path.name, f"failed: {e}")
                return file_path, original_file, None, str(e)

    tasks = []

    for pdf_file in files["pdf"]:
        tasks.append(process_with_semaphore(pdf_file, process_pdf_async, "pdf"))

    for md_file in files["markdown"]:
        tasks.append(process_with_semaphore(md_file, process_markdown_async, "markdown"))

    if convert_unsupported and files["docling"]:
        use_serve = False
        use_local = False
        docling_serve_client: DoclingServeClient | None = None

        if config.docling_serve_url:
            docling_serve_client = DoclingServeClient(
                config.docling_serve_url,
                timeout=config.docling_serve_timeout,
            )
            async with docling_serve_client:
                if await docling_serve_client.health_check():
                    use_serve = True
                    results["conversion_method"] = "docling-serve"
                    console.print(
                        f"[green]Using docling-serve at {config.docling_serve_url}[/green]"
                    )
                else:
                    console.print(
                        f"[yellow]Warning: docling-serve at {config.docling_serve_url} "
                        "is not available. Falling back to local docling.[/yellow]"
                    )

        if not use_serve and _check_docling_available():
            use_local = True
            results["conversion_method"] = "docling-local"
            console.print("[cyan]Using local docling for conversion[/cyan]")

        if not use_serve and not use_local:
            console.print(
                "[yellow]Warning: Neither docling-serve nor local docling available. "
                "Install with: pip install pageindex[docling] "
                "or set --docling-serve-url[/yellow]"
            )
            results["skipped"].extend([str(f) for f in files["docling"]])
        else:
            temp_dir = Path(tempfile.mkdtemp(prefix="pageindex_"))

            if use_serve and docling_serve_client:
                async with DoclingServeClient(
                    config.docling_serve_url,
                    timeout=config.docling_serve_timeout,
                ) as client:
                    for docling_file in files["docling"]:
                        if on_progress:
                            on_progress(docling_file.name, "converting (serve)")

                        converted_path = await _convert_with_docling_serve(
                            docling_file, temp_dir, client
                        )
                        if converted_path:
                            tasks.append(
                                process_with_semaphore(
                                    converted_path,
                                    process_markdown_async,
                                    "markdown",
                                    original_file=docling_file,
                                )
                            )
                        else:
                            results["failed"].append(
                                {
                                    "file": str(docling_file),
                                    "error": "Docling-serve conversion failed",
                                }
                            )
            else:
                for docling_file in files["docling"]:
                    if on_progress:
                        on_progress(docling_file.name, "converting (local)")

                    converted_path = _convert_with_docling_local(docling_file, temp_dir)
                    if converted_path:
                        tasks.append(
                            process_with_semaphore(
                                converted_path,
                                process_markdown_async,
                                "markdown",
                                original_file=docling_file,
                            )
                        )
                    else:
                        results["failed"].append(
                            {
                                "file": str(docling_file),
                                "error": "Local docling conversion failed",
                            }
                        )

    results["skipped"].extend([str(f) for f in files["unsupported"]])

    if tasks:
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for item in completed:
            if isinstance(item, Exception):
                results["failed"].append(
                    {
                        "file": "unknown",
                        "error": str(item),
                    }
                )
            else:
                file_path, original_file, _result, error = item
                display_file = original_file if original_file else file_path
                if error:
                    results["failed"].append(
                        {
                            "file": str(display_file),
                            "error": error,
                        }
                    )
                else:
                    stem = original_file.stem if original_file else file_path.stem
                    results["success"].append(
                        {
                            "file": str(display_file),
                            "output": str(output_dir / f"{stem}_structure.json"),
                        }
                    )

    results["statistics"] = {
        "total": total_files,
        "success": len(results["success"]),
        "failed": len(results["failed"]),
        "skipped": len(results["skipped"]),
    }

    return results


def process_folder_sync(
    folder: str | Path,
    config: PageIndexConfig | None = None,
    output_dir: str | Path | None = None,
    max_concurrent: int = 5,
    convert_unsupported: bool = True,
    project_id: str | None = None,
    location: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Synchronous wrapper for process_folder.

    Args:
        folder: Path to the folder containing documents
        config: PageIndex configuration (optional if project_id provided)
        output_dir: Directory to save results
        max_concurrent: Maximum concurrent tasks
        convert_unsupported: Use docling for unsupported formats
        project_id: Google Cloud project ID (alternative to config)
        location: Vertex AI location
        model: Gemini model to use

    Returns:
        Dictionary with processing results
    """
    if config is None:
        config = PageIndexConfig(
            project_id=project_id or "",
            location=location or "us-central1",
            model=model or "gemini-1.5-flash",
        )

    return asyncio.run(
        process_folder(
            folder=folder,
            config=config,
            output_dir=output_dir,
            max_concurrent=max_concurrent,
            convert_unsupported=convert_unsupported,
        )
    )
