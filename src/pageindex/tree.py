"""Tree data structures and operations for PageIndex."""

from __future__ import annotations

import copy
from typing import Any


def write_node_id(data: Any, node_id: int = 0) -> int:
    """Recursively assign node IDs to tree structure."""
    if isinstance(data, dict):
        data["node_id"] = str(node_id).zfill(4)
        node_id += 1
        for key in list(data.keys()):
            if "nodes" in key:
                node_id = write_node_id(data[key], node_id)
    elif isinstance(data, list):
        for item in data:
            node_id = write_node_id(item, node_id)
    return node_id


def get_nodes(structure: Any) -> list[dict]:
    """Get all nodes from tree structure (flattened)."""
    if isinstance(structure, dict):
        structure_node = copy.deepcopy(structure)
        structure_node.pop("nodes", None)
        nodes = [structure_node]
        for key in list(structure.keys()):
            if "nodes" in key:
                nodes.extend(get_nodes(structure[key]))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(get_nodes(item))
        return nodes
    return []


def structure_to_list(structure: Any) -> list[dict]:
    """Convert tree structure to flat list preserving hierarchy info."""
    if isinstance(structure, dict):
        nodes = [structure]
        if "nodes" in structure:
            nodes.extend(structure_to_list(structure["nodes"]))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(structure_to_list(item))
        return nodes
    return []


def get_leaf_nodes(structure: Any) -> list[dict]:
    """Get all leaf nodes (nodes without children)."""
    if isinstance(structure, dict):
        if not structure.get("nodes"):
            structure_node = copy.deepcopy(structure)
            structure_node.pop("nodes", None)
            return [structure_node]
        else:
            leaf_nodes = []
            for key in list(structure.keys()):
                if "nodes" in key:
                    leaf_nodes.extend(get_leaf_nodes(structure[key]))
            return leaf_nodes
    elif isinstance(structure, list):
        leaf_nodes = []
        for item in structure:
            leaf_nodes.extend(get_leaf_nodes(item))
        return leaf_nodes
    return []


def is_leaf_node(data: Any, node_id: str) -> bool:
    """Check if a node with given ID is a leaf node."""

    def find_node(data: Any, node_id: str) -> dict | None:
        if isinstance(data, dict):
            if data.get("node_id") == node_id:
                return data
            for key in data.keys():
                if "nodes" in key:
                    result = find_node(data[key], node_id)
                    if result:
                        return result
        elif isinstance(data, list):
            for item in data:
                result = find_node(item, node_id)
                if result:
                    return result
        return None

    node = find_node(data, node_id)
    return node is not None and not node.get("nodes")


def list_to_tree(data: list[dict]) -> list[dict]:
    """Convert flat list with structure codes to tree."""

    def get_parent_structure(structure: str) -> str | None:
        if not structure:
            return None
        parts = str(structure).split(".")
        return ".".join(parts[:-1]) if len(parts) > 1 else None

    nodes: dict[str, dict] = {}
    root_nodes: list[dict] = []

    for item in data:
        structure = item.get("structure")
        node = {
            "title": item.get("title"),
            "start_index": item.get("start_index"),
            "end_index": item.get("end_index"),
            "nodes": [],
        }

        nodes[structure] = node
        parent_structure = get_parent_structure(structure)

        if parent_structure and parent_structure in nodes:
            nodes[parent_structure]["nodes"].append(node)
        else:
            root_nodes.append(node)

    def clean_node(node: dict) -> dict:
        if not node["nodes"]:
            del node["nodes"]
        else:
            for child in node["nodes"]:
                clean_node(child)
        return node

    return [clean_node(node) for node in root_nodes]


def add_preface_if_needed(data: list[dict]) -> list[dict]:
    """Add preface node if document starts after page 1."""
    if not isinstance(data, list) or not data:
        return data

    if data[0].get("physical_index") is not None and data[0]["physical_index"] > 1:
        preface_node = {
            "structure": "0",
            "title": "Preface",
            "physical_index": 1,
        }
        data.insert(0, preface_node)
    return data


def post_processing(structure: list[dict], end_physical_index: int) -> list[dict]:
    """Post-process flat structure into tree with start/end indices."""
    for i, item in enumerate(structure):
        item["start_index"] = item.get("physical_index")
        if i < len(structure) - 1:
            if structure[i + 1].get("appear_start") == "yes":
                item["end_index"] = structure[i + 1]["physical_index"] - 1
            else:
                item["end_index"] = structure[i + 1]["physical_index"]
        else:
            item["end_index"] = end_physical_index

    tree = list_to_tree(structure)
    if tree:
        return tree

    for node in structure:
        node.pop("appear_start", None)
        node.pop("physical_index", None)
    return structure


def remove_page_number(data: Any) -> Any:
    """Remove page_number field from tree structure."""
    if isinstance(data, dict):
        data.pop("page_number", None)
        for key in list(data.keys()):
            if "nodes" in key:
                remove_page_number(data[key])
    elif isinstance(data, list):
        for item in data:
            remove_page_number(item)
    return data


def remove_structure_text(data: Any) -> Any:
    """Remove text field from tree structure."""
    if isinstance(data, dict):
        data.pop("text", None)
        if "nodes" in data:
            remove_structure_text(data["nodes"])
    elif isinstance(data, list):
        for item in data:
            remove_structure_text(item)
    return data


def remove_fields(data: Any, fields: list[str] | None = None) -> Any:
    """Remove specified fields from tree structure."""
    if fields is None:
        fields = ["text"]
    if isinstance(data, dict):
        return {k: remove_fields(v, fields) for k, v in data.items() if k not in fields}
    elif isinstance(data, list):
        return [remove_fields(item, fields) for item in data]
    return data


def reorder_dict(data: dict, key_order: list[str]) -> dict:
    """Reorder dictionary keys according to specified order."""
    if not key_order:
        return data
    return {key: data[key] for key in key_order if key in data}


def format_structure(structure: Any, order: list[str] | None = None) -> Any:
    """Format tree structure with specified key ordering."""
    if not order:
        return structure
    if isinstance(structure, dict):
        if "nodes" in structure:
            structure["nodes"] = format_structure(structure["nodes"], order)
        if not structure.get("nodes"):
            structure.pop("nodes", None)
        structure = reorder_dict(structure, order)
    elif isinstance(structure, list):
        structure = [format_structure(item, order) for item in structure]
    return structure


def create_clean_structure_for_description(structure: Any) -> Any:
    """Create clean structure for document description generation."""
    if isinstance(structure, dict):
        clean_node = {}
        for key in ["title", "node_id", "summary", "prefix_summary"]:
            if key in structure:
                clean_node[key] = structure[key]

        if structure.get("nodes"):
            clean_node["nodes"] = create_clean_structure_for_description(structure["nodes"])

        return clean_node
    elif isinstance(structure, list):
        return [create_clean_structure_for_description(item) for item in structure]
    return structure


def validate_and_truncate_physical_indices(
    toc_with_page_number: list[dict],
    page_list_length: int,
    start_index: int = 1,
    logger: Any = None,
) -> list[dict]:
    """Validate and truncate physical indices that exceed document length."""
    if not toc_with_page_number:
        return toc_with_page_number

    max_allowed_page = page_list_length + start_index - 1
    truncated_items = []

    for item in toc_with_page_number:
        if item.get("physical_index") is not None:
            original_index = item["physical_index"]
            if original_index > max_allowed_page:
                item["physical_index"] = None
                truncated_items.append(
                    {
                        "title": item.get("title", "Unknown"),
                        "original_index": original_index,
                    }
                )
                if logger:
                    logger.info(
                        f"Removed physical_index for '{item.get('title', 'Unknown')}' "
                        f"(was {original_index}, beyond document)"
                    )

    if truncated_items and logger:
        logger.info(f"Total removed items: {len(truncated_items)}")

    print(f"Document validation: {page_list_length} pages, max allowed index: {max_allowed_page}")
    if truncated_items:
        print(f"Truncated {len(truncated_items)} TOC items that exceeded document length")

    return toc_with_page_number
