"""Main PDF processing pipeline for PageIndex."""

from __future__ import annotations

import asyncio
import copy
import json
import random
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pageindex.config import ConfigLoader, PageIndexConfig
from pageindex.llm import LLMClient
from pageindex.pdf.parser import (
    add_node_text,
    get_page_tokens,
)
from pageindex.pdf.toc import (
    add_page_offset_to_toc_json,
    calculate_page_offset,
    check_title_appearance,
    check_title_appearance_in_start_concurrent,
    check_toc,
    extract_matching_page_pairs,
    page_list_to_group_text,
    toc_index_extractor,
    toc_transformer,
)
from pageindex.tree import (
    add_preface_if_needed,
    create_clean_structure_for_description,
    post_processing,
    remove_structure_text,
    structure_to_list,
    validate_and_truncate_physical_indices,
    write_node_id,
)
from pageindex.utils import (
    JsonLogger,
    convert_physical_index_to_int,
    extract_json,
    get_pdf_name,
)

if TYPE_CHECKING:
    pass


class PageIndexProcessor:
    """Main processor for building PageIndex from PDF documents."""

    def __init__(self, config: PageIndexConfig):
        self.config = config
        self.llm = LLMClient(config)

    def process(self, doc: str | Path | BytesIO) -> dict[str, Any]:
        """Process a PDF document and return the tree structure.

        Args:
            doc: Path to PDF file or BytesIO object

        Returns:
            Dictionary with doc_name, optional doc_description, and structure
        """
        return page_index_main(doc, self.config)


def page_index(
    doc: str | Path | BytesIO,
    project_id: str | None = None,
    location: str | None = None,
    model: str | None = None,
    toc_check_page_num: int | None = None,
    max_page_num_each_node: int | None = None,
    max_token_num_each_node: int | None = None,
    if_add_node_id: str | None = None,
    if_add_node_summary: str | None = None,
    if_add_doc_description: str | None = None,
    if_add_node_text: str | None = None,
) -> dict[str, Any]:
    """Process a PDF document and return the tree structure.

    This is the main entry point for the PageIndex library.

    Args:
        doc: Path to PDF file or BytesIO object
        project_id: Google Cloud project ID for Vertex AI
        location: Vertex AI location/region
        model: Gemini model to use
        toc_check_page_num: Number of pages to check for TOC
        max_page_num_each_node: Maximum pages per node
        max_token_num_each_node: Maximum tokens per node
        if_add_node_id: Whether to add node IDs ("yes"/"no")
        if_add_node_summary: Whether to generate summaries ("yes"/"no")
        if_add_doc_description: Whether to generate doc description ("yes"/"no")
        if_add_node_text: Whether to include full text ("yes"/"no")

    Returns:
        Dictionary with doc_name, optional doc_description, and structure
    """
    user_opt = {arg: value for arg, value in locals().items() if arg != "doc" and value is not None}
    config = ConfigLoader().load(user_opt)
    return page_index_main(doc, config)


def page_index_main(doc: str | Path | BytesIO, config: PageIndexConfig) -> dict[str, Any]:
    """Main processing function for PageIndex."""
    logger = JsonLogger(doc)
    llm = LLMClient(config)

    is_valid_pdf = (
        isinstance(doc, (str, Path)) and Path(doc).is_file() and str(doc).lower().endswith(".pdf")
    ) or isinstance(doc, BytesIO)

    if not is_valid_pdf:
        raise ValueError("Unsupported input type. Expected a PDF file path or BytesIO object.")

    print("Parsing PDF...")
    page_list = get_page_tokens(doc, config)

    logger.info({"total_page_number": len(page_list)})
    logger.info({"total_token": sum(page[1] for page in page_list)})

    async def page_index_builder() -> dict[str, Any]:
        structure = await tree_parser(page_list, config, llm, doc=doc, logger=logger)

        if config.if_add_node_id == "yes":
            write_node_id(structure)

        if config.if_add_node_text == "yes":
            add_node_text(structure, page_list)

        if config.if_add_node_summary == "yes":
            if config.if_add_node_text == "no":
                add_node_text(structure, page_list)
            await generate_summaries_for_structure(structure, llm)
            if config.if_add_node_text == "no":
                remove_structure_text(structure)

        if config.if_add_doc_description == "yes":
            clean_structure = create_clean_structure_for_description(structure)
            doc_description = generate_doc_description(clean_structure, llm)
            return {
                "doc_name": get_pdf_name(doc),
                "doc_description": doc_description,
                "structure": structure,
            }

        return {
            "doc_name": get_pdf_name(doc),
            "structure": structure,
        }

    return asyncio.run(page_index_builder())


async def tree_parser(
    page_list: list[tuple[str, int]],
    config: PageIndexConfig,
    llm: LLMClient,
    doc: Any = None,
    logger: Any = None,
) -> list[dict]:
    """Parse PDF pages into a tree structure."""
    check_toc_result = check_toc(page_list, config, llm)
    logger.info(check_toc_result)

    if (
        check_toc_result.get("toc_content")
        and check_toc_result["toc_content"].strip()
        and check_toc_result["page_index_given_in_toc"] == "yes"
    ):
        toc_with_page_number = await meta_processor(
            page_list,
            mode="process_toc_with_page_numbers",
            start_index=1,
            toc_content=check_toc_result["toc_content"],
            toc_page_list=check_toc_result["toc_page_list"],
            config=config,
            llm=llm,
            logger=logger,
        )
    else:
        toc_with_page_number = await meta_processor(
            page_list,
            mode="process_no_toc",
            start_index=1,
            config=config,
            llm=llm,
            logger=logger,
        )

    toc_with_page_number = add_preface_if_needed(toc_with_page_number)
    toc_with_page_number = await check_title_appearance_in_start_concurrent(
        toc_with_page_number, page_list, llm, logger
    )

    valid_toc_items = [
        item for item in toc_with_page_number if item.get("physical_index") is not None
    ]

    toc_tree = post_processing(valid_toc_items, len(page_list))
    tasks = [
        process_large_node_recursively(node, page_list, config, llm, logger) for node in toc_tree
    ]
    await asyncio.gather(*tasks)

    return toc_tree


async def meta_processor(
    page_list: list[tuple[str, int]],
    mode: str,
    start_index: int = 1,
    toc_content: str | None = None,
    toc_page_list: list[int] | None = None,
    config: PageIndexConfig | None = None,
    llm: LLMClient | None = None,
    logger: Any = None,
) -> list[dict]:
    """Process document based on TOC detection mode."""
    print(f"Processing mode: {mode}")
    print(f"Start index: {start_index}")

    if mode == "process_toc_with_page_numbers":
        toc_with_page_number = process_toc_with_page_numbers(
            toc_content, toc_page_list, page_list, config, llm, logger
        )
    elif mode == "process_toc_no_page_numbers":
        toc_with_page_number = process_toc_no_page_numbers(
            toc_content, toc_page_list, page_list, config, llm, logger
        )
    else:
        toc_with_page_number = process_no_toc(page_list, start_index, config, llm, logger)

    toc_with_page_number = [
        item for item in toc_with_page_number if item.get("physical_index") is not None
    ]

    toc_with_page_number = validate_and_truncate_physical_indices(
        toc_with_page_number, len(page_list), start_index=start_index, logger=logger
    )

    accuracy, incorrect_results = await verify_toc(
        page_list, toc_with_page_number, start_index, llm
    )

    logger.info(
        {
            "mode": mode,
            "accuracy": accuracy,
            "incorrect_results": incorrect_results,
        }
    )

    if accuracy == 1.0 and len(incorrect_results) == 0:
        return toc_with_page_number

    if accuracy > 0.6 and len(incorrect_results) > 0:
        toc_with_page_number, _ = await fix_incorrect_toc_with_retries(
            toc_with_page_number, page_list, incorrect_results, start_index, 3, llm, logger
        )
        return toc_with_page_number

    if mode == "process_toc_with_page_numbers":
        return await meta_processor(
            page_list,
            mode="process_toc_no_page_numbers",
            toc_content=toc_content,
            toc_page_list=toc_page_list,
            start_index=start_index,
            config=config,
            llm=llm,
            logger=logger,
        )
    elif mode == "process_toc_no_page_numbers":
        return await meta_processor(
            page_list,
            mode="process_no_toc",
            start_index=start_index,
            config=config,
            llm=llm,
            logger=logger,
        )
    else:
        raise Exception("Processing failed")


def process_toc_with_page_numbers(
    toc_content: str,
    toc_page_list: list[int],
    page_list: list[tuple[str, int]],
    config: PageIndexConfig,
    llm: LLMClient,
    logger: Any,
) -> list[dict]:
    """Process document with TOC that has page numbers."""
    toc_with_page_number = toc_transformer(toc_content, llm)
    logger.info(f"toc_with_page_number: {toc_with_page_number}")

    toc_no_page_number = copy.deepcopy(toc_with_page_number)
    for item in toc_no_page_number:
        item.pop("page_number", None)

    start_page_index = toc_page_list[-1] + 1
    main_content = ""
    for page_index in range(
        start_page_index, min(start_page_index + config.toc_check_page_num, len(page_list))
    ):
        main_content += f"<physical_index_{page_index + 1}>\n{page_list[page_index][0]}\n</physical_index_{page_index + 1}>\n\n"

    toc_with_physical_index = toc_index_extractor(toc_no_page_number, main_content, llm)
    logger.info(f"toc_with_physical_index: {toc_with_physical_index}")

    toc_with_physical_index = convert_physical_index_to_int(toc_with_physical_index)
    logger.info(f"toc_with_physical_index: {toc_with_physical_index}")

    matching_pairs = extract_matching_page_pairs(
        toc_with_page_number, toc_with_physical_index, start_page_index
    )
    logger.info(f"matching_pairs: {matching_pairs}")

    offset = calculate_page_offset(matching_pairs)
    logger.info(f"offset: {offset}")

    if offset is not None:
        toc_with_page_number = add_page_offset_to_toc_json(toc_with_page_number, offset)
    logger.info(f"toc_with_page_number: {toc_with_page_number}")

    toc_with_page_number = process_none_page_numbers(toc_with_page_number, page_list, 1, llm)
    logger.info(f"toc_with_page_number: {toc_with_page_number}")

    return toc_with_page_number


def process_toc_no_page_numbers(
    toc_content: str,
    toc_page_list: list[int],
    page_list: list[tuple[str, int]],
    config: PageIndexConfig,
    llm: LLMClient,
    logger: Any,
) -> list[dict]:
    """Process document with TOC but no page numbers."""
    page_contents = []
    token_lengths = []
    toc_content_json = toc_transformer(toc_content, llm)
    logger.info(f"toc_transformer: {toc_content_json}")

    for page_index in range(1, len(page_list) + 1):
        page_text = f"<physical_index_{page_index}>\n{page_list[page_index - 1][0]}\n</physical_index_{page_index}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(page_list[page_index - 1][1])

    group_texts = page_list_to_group_text(page_contents, token_lengths)
    logger.info(f"len(group_texts): {len(group_texts)}")

    toc_with_page_number = copy.deepcopy(toc_content_json)
    for group_text in group_texts:
        toc_with_page_number = add_page_number_to_toc(group_text, toc_with_page_number, llm)
    logger.info(f"add_page_number_to_toc: {toc_with_page_number}")

    toc_with_page_number = convert_physical_index_to_int(toc_with_page_number)
    logger.info(f"convert_physical_index_to_int: {toc_with_page_number}")

    return toc_with_page_number


def process_no_toc(
    page_list: list[tuple[str, int]],
    start_index: int,
    config: PageIndexConfig,
    llm: LLMClient,
    logger: Any,
) -> list[dict]:
    """Process document without TOC."""
    page_contents = []
    token_lengths = []
    for page_index in range(start_index, start_index + len(page_list)):
        page_text = f"<physical_index_{page_index}>\n{page_list[page_index - start_index][0]}\n</physical_index_{page_index}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(page_list[page_index - start_index][1])

    group_texts = page_list_to_group_text(page_contents, token_lengths)
    logger.info(f"len(group_texts): {len(group_texts)}")

    toc_with_page_number = generate_toc_init(group_texts[0], llm)
    for group_text in group_texts[1:]:
        toc_with_page_number_additional = generate_toc_continue(
            toc_with_page_number, group_text, llm
        )
        toc_with_page_number.extend(toc_with_page_number_additional)
    logger.info(f"generate_toc: {toc_with_page_number}")

    toc_with_page_number = convert_physical_index_to_int(toc_with_page_number)
    logger.info(f"convert_physical_index_to_int: {toc_with_page_number}")

    return toc_with_page_number


def generate_toc_init(part: str, llm: LLMClient) -> list[dict]:
    """Generate initial TOC from document text."""
    print("Generating initial TOC...")
    prompt = (
        """
You are an expert in extracting hierarchical tree structure.
Your task is to generate the tree structure of the document.

The structure variable is the numeric system representing the hierarchy section index.
For example, the first section has structure index 1, the first subsection has structure index 1.1, etc.

For the title, extract the original title from the text, only fix space inconsistency.

The provided text contains tags like <physical_index_X> to indicate the start and end of page X.

For the physical_index, extract the physical index of the start of the section. Keep the format.

The response should be in the following format:
[
    {
        "structure": "(string)",
        "title": "<title>",
        "physical_index": "<physical_index_X>" (keep the format)
    },
    ...
]

Directly return the final JSON structure. Do not output anything else.

Given text:
"""
        + part
    )

    response, finish_reason = llm.chat_with_finish_reason(prompt)

    if finish_reason == "finished":
        return extract_json(response)
    else:
        raise Exception(f"finish reason: {finish_reason}")


def generate_toc_continue(toc_content: list[dict], part: str, llm: LLMClient) -> list[dict]:
    """Continue generating TOC from additional document text."""
    print("Continuing TOC generation...")
    prompt = (
        """
You are an expert in extracting hierarchical tree structure.
You are given a tree structure of the previous part and the text of the current part.
Your task is to continue the tree structure from the previous part to include the current part.

The structure variable is the numeric system representing the hierarchy section index.

For the title, extract the original title from the text, only fix space inconsistency.

The provided text contains tags like <physical_index_X> to indicate the start and end of page X.

For the physical_index, extract the physical index of the start of the section. Keep the format.

The response should be in the following format:
[
    {
        "structure": "(string)",
        "title": "<title>",
        "physical_index": "<physical_index_X>" (keep the format)
    },
    ...
]

Directly return the additional part of the final JSON structure. Do not output anything else.

Given text:
"""
        + part
        + "\n\nPrevious tree structure:\n"
        + json.dumps(toc_content, indent=2)
    )

    response, finish_reason = llm.chat_with_finish_reason(prompt)

    if finish_reason == "finished":
        return extract_json(response)
    else:
        raise Exception(f"finish reason: {finish_reason}")


def add_page_number_to_toc(part: str, structure: list[dict], llm: LLMClient) -> list[dict]:
    """Add page numbers to TOC structure."""
    prompt = (
        """
You are given a JSON structure of a document and a partial part of the document.
Your task is to check if the title described in the structure starts in the partial document.

The provided text contains tags like <physical_index_X> to indicate the physical location of page X.

If the full target section starts in the partial document, insert "start": "yes" and "physical_index": "<physical_index_X>".
If the full target section does not start in the partial document, insert "start": "no", "physical_index": null.

The response should be in the following format:
[
    {
        "structure": "(string)",
        "title": "<title>",
        "start": "yes" or "no",
        "physical_index": "<physical_index_X>" (keep the format) or null
    },
    ...
]

The given structure contains the result of the previous part.
You need to fill the result of the current part, do not change the previous result.
Directly return the final JSON structure. Do not output anything else.

Current Partial Document:
"""
        + part
        + "\n\nGiven Structure:\n"
        + json.dumps(structure, indent=2)
    )

    current_json_raw = llm.chat(prompt)
    json_result = extract_json(current_json_raw)

    for item in json_result:
        if "start" in item:
            del item["start"]
    return json_result


def process_none_page_numbers(
    toc_items: list[dict],
    page_list: list[tuple[str, int]],
    start_index: int,
    llm: LLMClient,
) -> list[dict]:
    """Process TOC items that don't have page numbers."""
    for i, item in enumerate(toc_items):
        if "physical_index" not in item:
            prev_physical_index = 0
            for j in range(i - 1, -1, -1):
                if toc_items[j].get("physical_index") is not None:
                    prev_physical_index = toc_items[j]["physical_index"]
                    break

            next_physical_index = -1
            for j in range(i + 1, len(toc_items)):
                if toc_items[j].get("physical_index") is not None:
                    next_physical_index = toc_items[j]["physical_index"]
                    break

            page_contents = []
            for page_index in range(prev_physical_index, next_physical_index + 1):
                list_index = page_index - start_index
                if 0 <= list_index < len(page_list):
                    page_text = f"<physical_index_{page_index}>\n{page_list[list_index][0]}\n</physical_index_{page_index}>\n\n"
                    page_contents.append(page_text)

            item_copy = copy.deepcopy(item)
            item_copy.pop("page", None)
            result = add_page_number_to_toc("".join(page_contents), [item_copy], llm)
            if result and result[0].get("physical_index"):
                idx_str = result[0]["physical_index"]
                if isinstance(idx_str, str) and idx_str.startswith("<physical_index_"):
                    item["physical_index"] = int(
                        idx_str.replace("<physical_index_", "").replace(">", "").strip()
                    )
            item.pop("page", None)

    return toc_items


async def verify_toc(
    page_list: list[tuple[str, int]],
    list_result: list[dict],
    start_index: int,
    llm: LLMClient,
    n: int | None = None,
) -> tuple[float, list[dict]]:
    """Verify TOC accuracy by checking title appearances."""
    print("Verifying TOC...")

    last_physical_index = None
    for item in reversed(list_result):
        if item.get("physical_index") is not None:
            last_physical_index = item["physical_index"]
            break

    if last_physical_index is None or last_physical_index < len(page_list) / 2:
        return 0, []

    if n is None:
        print("Checking all items")
        sample_indices = list(range(len(list_result)))
    else:
        n = min(n, len(list_result))
        print(f"Checking {n} items")
        sample_indices = random.sample(range(len(list_result)), n)

    indexed_sample_list = []
    for idx in sample_indices:
        item = list_result[idx]
        if item.get("physical_index") is not None:
            item_with_index = item.copy()
            item_with_index["list_index"] = idx
            indexed_sample_list.append(item_with_index)

    tasks = [
        check_title_appearance(item, page_list, start_index, llm) for item in indexed_sample_list
    ]
    results = await asyncio.gather(*tasks)

    correct_count = 0
    incorrect_results = []
    for result in results:
        if result["answer"] == "yes":
            correct_count += 1
        else:
            incorrect_results.append(result)

    checked_count = len(results)
    accuracy = correct_count / checked_count if checked_count > 0 else 0
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy, incorrect_results


async def fix_incorrect_toc(
    toc_with_page_number: list[dict],
    page_list: list[tuple[str, int]],
    incorrect_results: list[dict],
    start_index: int,
    llm: LLMClient,
    logger: Any,
) -> tuple[list[dict], list[dict]]:
    """Fix incorrect TOC entries."""
    print(f"Fixing {len(incorrect_results)} incorrect TOC entries...")
    incorrect_indices = {result["list_index"] for result in incorrect_results}
    end_index = len(page_list) + start_index - 1

    async def process_and_check_item(incorrect_item: dict) -> dict:
        list_index = incorrect_item["list_index"]

        if list_index < 0 or list_index >= len(toc_with_page_number):
            return {
                "list_index": list_index,
                "title": incorrect_item["title"],
                "physical_index": incorrect_item.get("physical_index"),
                "is_valid": False,
            }

        prev_correct = None
        for i in range(list_index - 1, -1, -1):
            if i not in incorrect_indices and 0 <= i < len(toc_with_page_number):
                physical_index = toc_with_page_number[i].get("physical_index")
                if physical_index is not None:
                    prev_correct = physical_index
                    break
        if prev_correct is None:
            prev_correct = start_index - 1

        next_correct = None
        for i in range(list_index + 1, len(toc_with_page_number)):
            if i not in incorrect_indices and 0 <= i < len(toc_with_page_number):
                physical_index = toc_with_page_number[i].get("physical_index")
                if physical_index is not None:
                    next_correct = physical_index
                    break
        if next_correct is None:
            next_correct = end_index

        page_contents = []
        for page_index in range(prev_correct, next_correct + 1):
            list_idx = page_index - start_index
            if 0 <= list_idx < len(page_list):
                page_text = f"<physical_index_{page_index}>\n{page_list[list_idx][0]}\n</physical_index_{page_index}>\n\n"
                page_contents.append(page_text)

        content_range = "".join(page_contents)
        physical_index_int = single_toc_item_index_fixer(
            incorrect_item["title"], content_range, llm
        )

        check_item = incorrect_item.copy()
        check_item["physical_index"] = physical_index_int
        check_result = await check_title_appearance(check_item, page_list, start_index, llm)

        return {
            "list_index": list_index,
            "title": incorrect_item["title"],
            "physical_index": physical_index_int,
            "is_valid": check_result["answer"] == "yes",
        }

    tasks = [process_and_check_item(item) for item in incorrect_results]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    invalid_results = []
    for result in results:
        if isinstance(result, Exception):
            continue
        if result["is_valid"]:
            list_idx = result["list_index"]
            if 0 <= list_idx < len(toc_with_page_number):
                toc_with_page_number[list_idx]["physical_index"] = result["physical_index"]
        else:
            invalid_results.append(
                {
                    "list_index": result["list_index"],
                    "title": result["title"],
                    "physical_index": result["physical_index"],
                }
            )

    logger.info(f"invalid_results: {invalid_results}")
    return toc_with_page_number, invalid_results


async def fix_incorrect_toc_with_retries(
    toc_with_page_number: list[dict],
    page_list: list[tuple[str, int]],
    incorrect_results: list[dict],
    start_index: int,
    max_attempts: int,
    llm: LLMClient,
    logger: Any,
) -> tuple[list[dict], list[dict]]:
    """Fix incorrect TOC entries with retries."""
    print("Fixing incorrect TOC with retries...")
    fix_attempt = 0
    current_toc = toc_with_page_number
    current_incorrect = incorrect_results

    while current_incorrect:
        print(f"Fixing {len(current_incorrect)} incorrect results")
        current_toc, current_incorrect = await fix_incorrect_toc(
            current_toc, page_list, current_incorrect, start_index, llm, logger
        )
        fix_attempt += 1
        if fix_attempt >= max_attempts:
            logger.info("Maximum fix attempts reached")
            break

    return current_toc, current_incorrect


def single_toc_item_index_fixer(section_title: str, content: str, llm: LLMClient) -> int | None:
    """Fix a single TOC item's page index."""
    prompt = (
        """
You are given a section title and several pages of a document.
Your job is to find the physical index of the start page of the section.

The provided pages contain tags like <physical_index_X> to indicate the physical location of page X.

Reply in a JSON format:
{
    "thinking": "<which page contains the start of this section>",
    "physical_index": "<physical_index_X>" (keep the format)
}
Directly return the final JSON structure. Do not output anything else.

Section Title:
"""
        + section_title
        + "\n\nDocument pages:\n"
        + content
    )

    response = llm.chat(prompt)
    json_content = extract_json(response)
    return convert_physical_index_to_int(json_content.get("physical_index"))


async def process_large_node_recursively(
    node: dict,
    page_list: list[tuple[str, int]],
    config: PageIndexConfig,
    llm: LLMClient,
    logger: Any,
) -> dict:
    """Recursively process large nodes by splitting them."""
    node_page_list = page_list[node["start_index"] - 1 : node["end_index"]]
    token_num = sum(page[1] for page in node_page_list)

    if (
        node["end_index"] - node["start_index"] > config.max_page_num_each_node
        and token_num >= config.max_token_num_each_node
    ):
        print(
            f"Large node: {node['title']}, start: {node['start_index']}, "
            f"end: {node['end_index']}, tokens: {token_num}"
        )

        node_toc_tree = await meta_processor(
            node_page_list,
            mode="process_no_toc",
            start_index=node["start_index"],
            config=config,
            llm=llm,
            logger=logger,
        )
        node_toc_tree = await check_title_appearance_in_start_concurrent(
            node_toc_tree, page_list, llm, logger
        )

        valid_node_toc_items = [
            item for item in node_toc_tree if item.get("physical_index") is not None
        ]

        if (
            valid_node_toc_items
            and node["title"].strip() == valid_node_toc_items[0]["title"].strip()
        ):
            node["nodes"] = post_processing(valid_node_toc_items[1:], node["end_index"])
            node["end_index"] = (
                valid_node_toc_items[1]["start_index"]
                if len(valid_node_toc_items) > 1
                else node["end_index"]
            )
        else:
            node["nodes"] = post_processing(valid_node_toc_items, node["end_index"])
            node["end_index"] = (
                valid_node_toc_items[0]["start_index"]
                if valid_node_toc_items
                else node["end_index"]
            )

    if node.get("nodes"):
        tasks = [
            process_large_node_recursively(child_node, page_list, config, llm, logger)
            for child_node in node["nodes"]
        ]
        await asyncio.gather(*tasks)

    return node


async def generate_node_summary(node: dict, llm: LLMClient) -> str:
    """Generate a summary for a node."""
    prompt = f"""You are given a part of a document.
Your task is to generate a description of what main points are covered in this partial document.

Partial Document Text: {node["text"]}

Directly return the description, do not include any other text."""

    return await llm.chat_async(prompt)


async def generate_summaries_for_structure(structure: list[dict], llm: LLMClient) -> list[dict]:
    """Generate summaries for all nodes in the structure."""
    nodes = structure_to_list(structure)
    tasks = [generate_node_summary(node, llm) for node in nodes]
    summaries = await asyncio.gather(*tasks)

    for node, summary in zip(nodes, summaries):
        node["summary"] = summary

    return structure


def generate_doc_description(structure: list[dict], llm: LLMClient) -> str:
    """Generate a document description."""
    prompt = f"""You are an expert in generating descriptions for documents.
You are given a structure of a document.
Your task is to generate a one-sentence description for the document,
which makes it easy to distinguish the document from other documents.

Document Structure: {structure}

Directly return the description, do not include any other text."""

    return llm.chat(prompt)
