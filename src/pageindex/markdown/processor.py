"""Markdown processing for PageIndex."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pageindex.config import PageIndexConfig
from pageindex.llm import LLMClient
from pageindex.tree import (
    create_clean_structure_for_description,
    format_structure,
    structure_to_list,
    write_node_id,
)

if TYPE_CHECKING:
    pass


def extract_nodes_from_markdown(markdown_content: str) -> tuple[list[dict], list[str]]:
    """Extract header nodes from markdown content.

    Args:
        markdown_content: Raw markdown text

    Returns:
        Tuple of (node_list, lines) where node_list contains header info
    """
    header_pattern = r"^(#{1,6})\s+(.+)$"
    code_block_pattern = r"^```"
    node_list = []

    lines = markdown_content.split("\n")
    in_code_block = False

    for line_num, line in enumerate(lines, 1):
        stripped_line = line.strip()

        if re.match(code_block_pattern, stripped_line):
            in_code_block = not in_code_block
            continue

        if not stripped_line:
            continue

        if not in_code_block:
            match = re.match(header_pattern, stripped_line)
            if match:
                title = match.group(2).strip()
                node_list.append({"node_title": title, "line_num": line_num})

    return node_list, lines


def extract_node_text_content(
    node_list: list[dict],
    markdown_lines: list[str],
) -> list[dict]:
    """Extract text content for each node.

    Args:
        node_list: List of nodes with line numbers
        markdown_lines: All lines from the markdown file

    Returns:
        List of nodes with text content and level info
    """
    all_nodes = []
    for node in node_list:
        line_content = markdown_lines[node["line_num"] - 1]
        header_match = re.match(r"^(#{1,6})", line_content)

        if header_match is None:
            print(
                f"Warning: Line {node['line_num']} does not contain a valid header: '{line_content}'"
            )
            continue

        processed_node = {
            "title": node["node_title"],
            "line_num": node["line_num"],
            "level": len(header_match.group(1)),
        }
        all_nodes.append(processed_node)

    for i, node in enumerate(all_nodes):
        start_line = node["line_num"] - 1
        if i + 1 < len(all_nodes):
            end_line = all_nodes[i + 1]["line_num"] - 1
        else:
            end_line = len(markdown_lines)

        node["text"] = "\n".join(markdown_lines[start_line:end_line]).strip()

    return all_nodes


def update_node_list_with_text_token_count(
    node_list: list[dict],
    config: PageIndexConfig,
) -> list[dict]:
    """Update nodes with token counts including children.

    Args:
        node_list: List of nodes
        config: PageIndex configuration

    Returns:
        Updated node list with text_token_count
    """
    from pageindex.llm import count_tokens

    def find_all_children(parent_index: int, parent_level: int, node_list: list[dict]) -> list[int]:
        children_indices = []
        for i in range(parent_index + 1, len(node_list)):
            current_level = node_list[i]["level"]
            if current_level <= parent_level:
                break
            children_indices.append(i)
        return children_indices

    result_list = node_list.copy()

    for i in range(len(result_list) - 1, -1, -1):
        current_node = result_list[i]
        current_level = current_node["level"]

        children_indices = find_all_children(i, current_level, result_list)

        node_text = current_node.get("text", "")
        total_text = node_text

        for child_index in children_indices:
            child_text = result_list[child_index].get("text", "")
            if child_text:
                total_text += "\n" + child_text

        result_list[i]["text_token_count"] = count_tokens(config, total_text)

    return result_list


def tree_thinning_for_index(
    node_list: list[dict],
    min_node_token: int,
    config: PageIndexConfig,
) -> list[dict]:
    """Apply tree thinning to merge small nodes.

    Args:
        node_list: List of nodes
        min_node_token: Minimum token threshold
        config: PageIndex configuration

    Returns:
        Thinned node list
    """
    from pageindex.llm import count_tokens

    def find_all_children(parent_index: int, parent_level: int, node_list: list[dict]) -> list[int]:
        children_indices = []
        for i in range(parent_index + 1, len(node_list)):
            current_level = node_list[i]["level"]
            if current_level <= parent_level:
                break
            children_indices.append(i)
        return children_indices

    result_list = node_list.copy()
    nodes_to_remove: set[int] = set()

    for i in range(len(result_list) - 1, -1, -1):
        if i in nodes_to_remove:
            continue

        current_node = result_list[i]
        current_level = current_node["level"]

        total_tokens = current_node.get("text_token_count", 0)

        if total_tokens < min_node_token:
            children_indices = find_all_children(i, current_level, result_list)

            children_texts = []
            for child_index in sorted(children_indices):
                if child_index not in nodes_to_remove:
                    child_text = result_list[child_index].get("text", "")
                    if child_text.strip():
                        children_texts.append(child_text)
                    nodes_to_remove.add(child_index)

            if children_texts:
                parent_text = current_node.get("text", "")
                merged_text = parent_text
                for child_text in children_texts:
                    if merged_text and not merged_text.endswith("\n"):
                        merged_text += "\n\n"
                    merged_text += child_text

                result_list[i]["text"] = merged_text
                result_list[i]["text_token_count"] = count_tokens(config, merged_text)

    for index in sorted(nodes_to_remove, reverse=True):
        result_list.pop(index)

    return result_list


def build_tree_from_nodes(node_list: list[dict]) -> list[dict]:
    """Build tree structure from flat node list.

    Args:
        node_list: Flat list of nodes with level info

    Returns:
        Hierarchical tree structure
    """
    if not node_list:
        return []

    stack: list[tuple[dict, int]] = []
    root_nodes: list[dict] = []
    node_counter = 1

    for node in node_list:
        current_level = node["level"]

        tree_node = {
            "title": node["title"],
            "node_id": str(node_counter).zfill(4),
            "text": node["text"],
            "line_num": node["line_num"],
            "nodes": [],
        }
        node_counter += 1

        while stack and stack[-1][1] >= current_level:
            stack.pop()

        if not stack:
            root_nodes.append(tree_node)
        else:
            parent_node, _ = stack[-1]
            parent_node["nodes"].append(tree_node)

        stack.append((tree_node, current_level))

    return root_nodes


def clean_tree_for_output(tree_nodes: list[dict]) -> list[dict]:
    """Clean tree structure for output.

    Args:
        tree_nodes: Tree structure

    Returns:
        Cleaned tree structure
    """
    cleaned_nodes = []

    for node in tree_nodes:
        cleaned_node = {
            "title": node["title"],
            "node_id": node["node_id"],
            "text": node["text"],
            "line_num": node["line_num"],
        }

        if node["nodes"]:
            cleaned_node["nodes"] = clean_tree_for_output(node["nodes"])

        cleaned_nodes.append(cleaned_node)

    return cleaned_nodes


async def get_node_summary(
    node: dict,
    summary_token_threshold: int,
    llm: LLMClient,
) -> str:
    """Get summary for a node, using text if short enough.

    Args:
        node: Node dictionary
        summary_token_threshold: Token threshold for summarization
        llm: LLM client

    Returns:
        Summary text
    """
    node_text = node.get("text", "")
    num_tokens = llm.count_tokens(node_text)

    if num_tokens < summary_token_threshold:
        return node_text
    else:
        return await generate_node_summary(node, llm)


async def generate_node_summary(node: dict, llm: LLMClient) -> str:
    """Generate a summary for a node using LLM.

    Args:
        node: Node dictionary with text
        llm: LLM client

    Returns:
        Generated summary
    """
    prompt = f"""You are given a part of a document.
Your task is to generate a description of what main points are covered in this partial document.

Partial Document Text: {node["text"]}

Directly return the description, do not include any other text."""

    return await llm.chat_async(prompt)


async def generate_summaries_for_structure_md(
    structure: list[dict],
    summary_token_threshold: int,
    llm: LLMClient,
) -> list[dict]:
    """Generate summaries for all nodes in markdown structure.

    Args:
        structure: Tree structure
        summary_token_threshold: Token threshold
        llm: LLM client

    Returns:
        Structure with summaries
    """
    nodes = structure_to_list(structure)
    tasks = [get_node_summary(node, summary_token_threshold, llm) for node in nodes]
    summaries = await asyncio.gather(*tasks)

    for node, summary in zip(nodes, summaries):
        if not node.get("nodes"):
            node["summary"] = summary
        else:
            node["prefix_summary"] = summary

    return structure


def generate_doc_description(structure: list[dict], llm: LLMClient) -> str:
    """Generate document description from structure.

    Args:
        structure: Tree structure
        llm: LLM client

    Returns:
        Document description
    """
    prompt = f"""You are an expert in generating descriptions for documents.
You are given a structure of a document.
Your task is to generate a one-sentence description for the document,
which makes it easy to distinguish the document from other documents.

Document Structure: {structure}

Directly return the description, do not include any other text."""

    return llm.chat(prompt)


async def md_to_tree(
    md_path: str | Path,
    config: PageIndexConfig | None = None,
    if_thinning: bool = False,
    min_token_threshold: int = 5000,
    if_add_node_summary: str = "no",
    summary_token_threshold: int = 200,
    if_add_doc_description: str = "no",
    if_add_node_text: str = "no",
    if_add_node_id: str = "yes",
) -> dict[str, Any]:
    """Convert markdown file to tree structure.

    Args:
        md_path: Path to markdown file
        config: PageIndex configuration (optional)
        if_thinning: Whether to apply tree thinning
        min_token_threshold: Minimum token threshold for thinning
        if_add_node_summary: Whether to add summaries ("yes"/"no")
        summary_token_threshold: Token threshold for summaries
        if_add_doc_description: Whether to add doc description ("yes"/"no")
        if_add_node_text: Whether to include text ("yes"/"no")
        if_add_node_id: Whether to add node IDs ("yes"/"no")

    Returns:
        Dictionary with doc_name and structure
    """
    if config is None:
        config = PageIndexConfig()

    llm = LLMClient(config)

    with open(md_path, encoding="utf-8") as f:
        markdown_content = f.read()

    print("Extracting nodes from markdown...")
    node_list, markdown_lines = extract_nodes_from_markdown(markdown_content)

    print("Extracting text content from nodes...")
    nodes_with_content = extract_node_text_content(node_list, markdown_lines)

    if if_thinning:
        nodes_with_content = update_node_list_with_text_token_count(nodes_with_content, config)
        print("Thinning nodes...")
        nodes_with_content = tree_thinning_for_index(
            nodes_with_content, min_token_threshold, config
        )

    print("Building tree from nodes...")
    tree_structure = build_tree_from_nodes(nodes_with_content)

    if if_add_node_id == "yes":
        write_node_id(tree_structure)

    print("Formatting tree structure...")

    if if_add_node_summary == "yes":
        tree_structure = format_structure(
            tree_structure,
            order=["title", "node_id", "summary", "prefix_summary", "text", "line_num", "nodes"],
        )

        print("Generating summaries for each node...")
        tree_structure = await generate_summaries_for_structure_md(
            tree_structure, summary_token_threshold, llm
        )

        if if_add_node_text == "no":
            tree_structure = format_structure(
                tree_structure,
                order=["title", "node_id", "summary", "prefix_summary", "line_num", "nodes"],
            )

        if if_add_doc_description == "yes":
            print("Generating document description...")
            clean_structure = create_clean_structure_for_description(tree_structure)
            doc_description = generate_doc_description(clean_structure, llm)
            return {
                "doc_name": Path(md_path).stem,
                "doc_description": doc_description,
                "structure": tree_structure,
            }
    else:
        if if_add_node_text == "yes":
            tree_structure = format_structure(
                tree_structure,
                order=[
                    "title",
                    "node_id",
                    "summary",
                    "prefix_summary",
                    "text",
                    "line_num",
                    "nodes",
                ],
            )
        else:
            tree_structure = format_structure(
                tree_structure,
                order=["title", "node_id", "summary", "prefix_summary", "line_num", "nodes"],
            )

    return {
        "doc_name": Path(md_path).stem,
        "structure": tree_structure,
    }
