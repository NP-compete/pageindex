"""PDF parsing utilities for PageIndex."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import pymupdf
import PyPDF2

if TYPE_CHECKING:
    from pageindex.config import PageIndexConfig


def get_page_tokens(
    pdf_path: str | Path | BytesIO,
    config: PageIndexConfig | None = None,
    pdf_parser: str = "PyMuPDF",
) -> list[tuple[str, int]]:
    """Extract pages with their text and token counts.

    Args:
        pdf_path: Path to PDF file or BytesIO object
        config: PageIndex configuration (used for token counting)
        pdf_parser: Parser to use ("PyMuPDF" or "PyPDF2")

    Returns:
        List of tuples (page_text, token_count)
    """
    from pageindex.llm import count_tokens

    if pdf_parser == "PyPDF2":
        pdf_reader = PyPDF2.PdfReader(str(pdf_path) if isinstance(pdf_path, Path) else pdf_path)
        page_list = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text() or ""
            if config:
                token_length = count_tokens(config, page_text)
            else:
                token_length = len(page_text) // 4
            page_list.append((page_text, token_length))
        return page_list

    elif pdf_parser == "PyMuPDF":
        if isinstance(pdf_path, BytesIO):
            doc = pymupdf.open(stream=pdf_path, filetype="pdf")
        elif isinstance(pdf_path, (str, Path)) and os.path.isfile(pdf_path):
            doc = pymupdf.open(str(pdf_path))
        else:
            raise ValueError(f"Invalid PDF path: {pdf_path}")

        page_list = []
        for page in doc:
            page_text = page.get_text() or ""
            if config:
                token_length = count_tokens(config, page_text)
            else:
                token_length = len(page_text) // 4
            page_list.append((page_text, token_length))
        doc.close()
        return page_list

    else:
        raise ValueError(f"Unsupported PDF parser: {pdf_parser}")


def get_text_of_pages(
    pdf_path: str | Path,
    start_page: int,
    end_page: int,
    tag: bool = True,
) -> str:
    """Extract text from a range of pages.

    Args:
        pdf_path: Path to PDF file
        start_page: Starting page (1-indexed)
        end_page: Ending page (inclusive)
        tag: Whether to add page tags

    Returns:
        Concatenated text from the pages
    """
    pdf_reader = PyPDF2.PdfReader(str(pdf_path))
    text = ""
    for page_num in range(start_page - 1, end_page):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text() or ""
        if tag:
            text += (
                f"<physical_index_{page_num + 1}>\n{page_text}\n</physical_index_{page_num + 1}>\n"
            )
        else:
            text += page_text
    return text


def get_text_of_pdf_pages(
    pdf_pages: list[tuple[str, int]],
    start_page: int,
    end_page: int,
) -> str:
    """Get text from pre-extracted pages.

    Args:
        pdf_pages: List of (text, token_count) tuples
        start_page: Starting page (1-indexed)
        end_page: Ending page (inclusive)

    Returns:
        Concatenated text
    """
    text = ""
    for page_num in range(start_page - 1, end_page):
        if 0 <= page_num < len(pdf_pages):
            text += pdf_pages[page_num][0]
    return text


def get_text_of_pdf_pages_with_labels(
    pdf_pages: list[tuple[str, int]],
    start_page: int,
    end_page: int,
) -> str:
    """Get text from pre-extracted pages with page labels.

    Args:
        pdf_pages: List of (text, token_count) tuples
        start_page: Starting page (1-indexed)
        end_page: Ending page (inclusive)

    Returns:
        Concatenated text with page tags
    """
    text = ""
    for page_num in range(start_page - 1, end_page):
        if 0 <= page_num < len(pdf_pages):
            text += f"<physical_index_{page_num + 1}>\n{pdf_pages[page_num][0]}\n</physical_index_{page_num + 1}>\n"
    return text


def get_number_of_pages(pdf_path: str | Path) -> int:
    """Get the number of pages in a PDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Number of pages
    """
    pdf_reader = PyPDF2.PdfReader(str(pdf_path))
    return len(pdf_reader.pages)


def add_node_text(node: dict, pdf_pages: list[tuple[str, int]]) -> None:
    """Add text content to tree nodes.

    Args:
        node: Tree node or list of nodes
        pdf_pages: List of (text, token_count) tuples
    """
    if isinstance(node, dict):
        start_page = node.get("start_index")
        end_page = node.get("end_index")
        if start_page and end_page:
            node["text"] = get_text_of_pdf_pages(pdf_pages, start_page, end_page)
        if "nodes" in node:
            add_node_text(node["nodes"], pdf_pages)
    elif isinstance(node, list):
        for item in node:
            add_node_text(item, pdf_pages)


def add_node_text_with_labels(node: dict, pdf_pages: list[tuple[str, int]]) -> None:
    """Add text content with page labels to tree nodes.

    Args:
        node: Tree node or list of nodes
        pdf_pages: List of (text, token_count) tuples
    """
    if isinstance(node, dict):
        start_page = node.get("start_index")
        end_page = node.get("end_index")
        if start_page and end_page:
            node["text"] = get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page)
        if "nodes" in node:
            add_node_text_with_labels(node["nodes"], pdf_pages)
    elif isinstance(node, list):
        for item in node:
            add_node_text_with_labels(item, pdf_pages)
