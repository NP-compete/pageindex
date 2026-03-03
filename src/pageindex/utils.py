"""Utility functions for PageIndex."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import PyPDF2

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def extract_json(content: str) -> dict[str, Any]:
    """Extract JSON from LLM response, handling markdown code blocks."""
    try:
        start_idx = content.find("```json")
        if start_idx != -1:
            start_idx += 7
            end_idx = content.rfind("```")
            json_content = content[start_idx:end_idx].strip()
        else:
            json_content = content.strip()

        json_content = json_content.replace("None", "null")
        json_content = json_content.replace("\n", " ").replace("\r", " ")
        json_content = " ".join(json_content.split())

        return json.loads(json_content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to extract JSON: {e}")
        try:
            json_content = json_content.replace(",]", "]").replace(",}", "}")
            return json.loads(json_content)
        except Exception:
            logger.error("Failed to parse JSON even after cleanup")
            return {}
    except Exception as e:
        logger.error(f"Unexpected error while extracting JSON: {e}")
        return {}


def get_json_content(response: str) -> str:
    """Extract JSON content from response, removing markdown delimiters."""
    start_idx = response.find("```json")
    if start_idx != -1:
        start_idx += 7
        response = response[start_idx:]

    end_idx = response.rfind("```")
    if end_idx != -1:
        response = response[:end_idx]

    return response.strip()


def sanitize_filename(filename: str, replacement: str = "-") -> str:
    """Sanitize filename by replacing invalid characters."""
    return filename.replace("/", replacement).replace("\\", replacement)


def get_pdf_name(pdf_path: str | Path | BytesIO) -> str:
    """Extract PDF name from path or BytesIO object."""
    if isinstance(pdf_path, (str, Path)):
        return Path(pdf_path).name
    elif isinstance(pdf_path, BytesIO):
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        meta = pdf_reader.metadata
        pdf_name = meta.title if meta and meta.title else "Untitled"
        return sanitize_filename(pdf_name)
    return "Unknown"


def get_pdf_title(pdf_path: str | Path) -> str:
    """Extract PDF title from metadata."""
    pdf_reader = PyPDF2.PdfReader(str(pdf_path))
    meta = pdf_reader.metadata
    return meta.title if meta and meta.title else "Untitled"


def convert_physical_index_to_int(data: Any) -> Any:
    """Convert physical_index strings to integers."""
    if isinstance(data, list):
        for i in range(len(data)):
            if isinstance(data[i], dict) and "physical_index" in data[i]:
                if isinstance(data[i]["physical_index"], str):
                    idx_str = data[i]["physical_index"]
                    if idx_str.startswith("<physical_index_"):
                        data[i]["physical_index"] = int(
                            idx_str.replace("<physical_index_", "").replace(">", "").strip()
                        )
                    elif idx_str.startswith("physical_index_"):
                        data[i]["physical_index"] = int(idx_str.split("_")[-1].strip())
    elif isinstance(data, str):
        if data.startswith("<physical_index_"):
            return int(data.replace("<physical_index_", "").replace(">", "").strip())
        elif data.startswith("physical_index_"):
            return int(data.split("_")[-1].strip())
    return data


def convert_page_to_int(data: list[dict]) -> list[dict]:
    """Convert page strings to integers in a list of dicts."""
    for item in data:
        if "page" in item and isinstance(item["page"], str):
            try:
                item["page"] = int(item["page"])
            except ValueError:
                pass
    return data


def get_first_start_page_from_text(text: str) -> int:
    """Extract first page number from tagged text."""
    match = re.search(r"<physical_index_(\d+)>", text)
    return int(match.group(1)) if match else -1


def get_last_start_page_from_text(text: str) -> int:
    """Extract last page number from tagged text."""
    matches = list(re.finditer(r"<physical_index_(\d+)>", text))
    return int(matches[-1].group(1)) if matches else -1


class JsonLogger:
    """JSON-based logger for tracking processing steps."""

    def __init__(self, file_path: str | Path | BytesIO):
        pdf_name = get_pdf_name(file_path)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{pdf_name}_{current_time}.json"
        os.makedirs("./logs", exist_ok=True)
        self.log_data: list[dict] = []

    def log(self, level: str, message: Any, **kwargs: Any) -> None:
        """Log a message at the specified level."""
        if isinstance(message, dict):
            self.log_data.append(message)
        else:
            self.log_data.append({"message": message})

        with open(self._filepath(), "w") as f:
            json.dump(self.log_data, f, indent=2)

    def info(self, message: Any, **kwargs: Any) -> None:
        """Log an info message."""
        self.log("INFO", message, **kwargs)

    def error(self, message: Any, **kwargs: Any) -> None:
        """Log an error message."""
        self.log("ERROR", message, **kwargs)

    def debug(self, message: Any, **kwargs: Any) -> None:
        """Log a debug message."""
        self.log("DEBUG", message, **kwargs)

    def _filepath(self) -> str:
        """Get the full path to the log file."""
        return os.path.join("logs", self.filename)


def print_toc(tree: list[dict], indent: int = 0) -> None:
    """Print tree structure as table of contents."""
    for node in tree:
        print(" " * indent + node["title"])
        if node.get("nodes"):
            print_toc(node["nodes"], indent + 1)


def print_json(data: Any, max_len: int = 40, indent: int = 2) -> None:
    """Print JSON with truncated long strings."""

    def simplify_data(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: simplify_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [simplify_data(item) for item in obj]
        elif isinstance(obj, str) and len(obj) > max_len:
            return obj[:max_len] + "..."
        return obj

    simplified = simplify_data(data)
    print(json.dumps(simplified, indent=indent, ensure_ascii=False))
