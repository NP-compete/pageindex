"""Table of Contents detection and extraction for PageIndex."""

from __future__ import annotations

import asyncio
import json
import math
import re
from typing import TYPE_CHECKING, Any

from pageindex.llm import LLMClient
from pageindex.utils import (
    convert_page_to_int,
    extract_json,
    get_json_content,
)

if TYPE_CHECKING:
    from pageindex.config import PageIndexConfig


def toc_detector_single_page(content: str, llm: LLMClient) -> str:
    """Detect if a page contains a table of contents."""
    prompt = f"""
Your job is to detect if there is a table of content provided in the given text.

Given text: {content}

return the following JSON format:
{{
    "thinking": "<your reasoning>",
    "toc_detected": "yes" or "no"
}}

Directly return the final JSON structure. Do not output anything else.
Please note: abstract, summary, notation list, figure list, table list, etc. are not table of contents."""

    response = llm.chat(prompt)
    json_content = extract_json(response)
    return json_content.get("toc_detected", "no")


def find_toc_pages(
    start_page_index: int,
    page_list: list[tuple[str, int]],
    config: PageIndexConfig,
    llm: LLMClient,
    logger: Any = None,
) -> list[int]:
    """Find pages containing table of contents."""
    print("Finding TOC pages...")
    last_page_is_yes = False
    toc_page_list = []
    i = start_page_index

    while i < len(page_list):
        if i >= config.toc_check_page_num and not last_page_is_yes:
            break
        detected_result = toc_detector_single_page(page_list[i][0], llm)
        if detected_result == "yes":
            if logger:
                logger.info(f"Page {i} has TOC")
            toc_page_list.append(i)
            last_page_is_yes = True
        elif detected_result == "no" and last_page_is_yes:
            if logger:
                logger.info(f"Found the last page with TOC: {i - 1}")
            break
        i += 1

    if not toc_page_list and logger:
        logger.info("No TOC found")

    return toc_page_list


def detect_page_index(toc_content: str, llm: LLMClient) -> str:
    """Detect if TOC contains page numbers."""
    print("Detecting page index in TOC...")
    prompt = f"""
You will be given a table of contents.

Your job is to detect if there are page numbers/indices given within the table of contents.

Given text: {toc_content}

Reply format:
{{
    "thinking": "<your reasoning>",
    "page_index_given_in_toc": "yes" or "no"
}}
Directly return the final JSON structure. Do not output anything else."""

    response = llm.chat(prompt)
    json_content = extract_json(response)
    return json_content.get("page_index_given_in_toc", "no")


def toc_extractor(
    page_list: list[tuple[str, int]],
    toc_page_list: list[int],
    llm: LLMClient,
) -> dict[str, Any]:
    """Extract TOC content from pages."""

    def transform_dots_to_colon(text: str) -> str:
        text = re.sub(r"\.{5,}", ": ", text)
        text = re.sub(r"(?:\. ){5,}\.?", ": ", text)
        return text

    toc_content = ""
    for page_index in toc_page_list:
        toc_content += page_list[page_index][0]
    toc_content = transform_dots_to_colon(toc_content)
    has_page_index = detect_page_index(toc_content, llm)

    return {
        "toc_content": toc_content,
        "page_index_given_in_toc": has_page_index,
    }


def check_if_toc_transformation_is_complete(content: str, toc: str, llm: LLMClient) -> str:
    """Check if TOC transformation is complete."""
    prompt = f"""
You are given a raw table of contents and a table of contents.
Your job is to check if the table of contents is complete.

Reply format:
{{
    "thinking": "<your reasoning>",
    "completed": "yes" or "no"
}}
Directly return the final JSON structure. Do not output anything else.

Raw Table of contents:
{content}

Cleaned Table of contents:
{toc}"""

    response = llm.chat(prompt)
    json_content = extract_json(response)
    return json_content.get("completed", "no")


def toc_transformer(toc_content: str, llm: LLMClient) -> list[dict]:
    """Transform TOC content into structured JSON."""
    print("Transforming TOC to JSON...")
    init_prompt = """
You are given a table of contents. Your job is to transform the whole table of content into a JSON format.

structure is the numeric system which represents the index of the hierarchy section in the table of contents.
For example, the first section has structure index 1, the first subsection has structure index 1.1, etc.

The response should be in the following JSON format:
{
    "table_of_contents": [
        {
            "structure": "(string)",
            "title": "<title>",
            "page": <page number>
        },
        ...
    ]
}
You should transform the full table of contents in one go.
Directly return the final JSON structure, do not output anything else."""

    prompt = init_prompt + "\nGiven table of contents:\n" + toc_content
    last_complete, finish_reason = llm.chat_with_finish_reason(prompt)

    if_complete = check_if_toc_transformation_is_complete(toc_content, last_complete, llm)
    if if_complete == "yes" and finish_reason == "finished":
        last_complete_json = extract_json(last_complete)
        return convert_page_to_int(last_complete_json.get("table_of_contents", []))

    last_complete = get_json_content(last_complete)
    while not (if_complete == "yes" and finish_reason == "finished"):
        position = last_complete.rfind("}")
        if position != -1:
            last_complete = last_complete[: position + 2]

        prompt = f"""
Your task is to continue the table of contents json structure.
Directly output the remaining part of the json structure.

The raw table of contents is:
{toc_content}

The incomplete transformed table of contents json structure is:
{last_complete}

Please continue the json structure."""

        new_complete, finish_reason = llm.chat_with_finish_reason(prompt)

        if new_complete.startswith("```json"):
            new_complete = get_json_content(new_complete)
        last_complete = last_complete + new_complete

        if_complete = check_if_toc_transformation_is_complete(toc_content, last_complete, llm)

    last_complete_json = json.loads(last_complete)
    return convert_page_to_int(last_complete_json.get("table_of_contents", []))


def toc_index_extractor(
    toc: list[dict],
    content: str,
    llm: LLMClient,
) -> list[dict]:
    """Extract physical page indices for TOC items."""
    print("Extracting TOC indices...")
    prompt = (
        """
You are given a table of contents in JSON format and several pages of a document.
Your job is to add the physical_index to the table of contents.

The provided pages contain tags like <physical_index_X> to indicate the physical location of page X.

The structure variable is the numeric system representing the hierarchy section index.

The response should be in the following JSON format:
[
    {
        "structure": "(string)",
        "title": "<title>",
        "physical_index": "<physical_index_X>" (keep the format)
    },
    ...
]

Only add the physical_index to sections that are in the provided pages.
If the section is not in the provided pages, do not add the physical_index.
Directly return the final JSON structure. Do not output anything else.

Table of contents:
"""
        + str(toc)
        + "\n\nDocument pages:\n"
        + content
    )

    response = llm.chat(prompt)
    return extract_json(response)


def page_list_to_group_text(
    page_contents: list[str],
    token_lengths: list[int],
    max_tokens: int = 20000,
    overlap_page: int = 1,
) -> list[str]:
    """Group pages into text chunks respecting token limits."""
    num_tokens = sum(token_lengths)

    if num_tokens <= max_tokens:
        return ["".join(page_contents)]

    subsets = []
    current_subset: list[str] = []
    current_token_count = 0

    expected_parts_num = math.ceil(num_tokens / max_tokens)
    average_tokens_per_part = math.ceil(((num_tokens / expected_parts_num) + max_tokens) / 2)

    for i, (page_content, page_tokens) in enumerate(zip(page_contents, token_lengths)):
        if current_token_count + page_tokens > average_tokens_per_part:
            subsets.append("".join(current_subset))
            overlap_start = max(i - overlap_page, 0)
            current_subset = list(page_contents[overlap_start:i])
            current_token_count = sum(token_lengths[overlap_start:i])

        current_subset.append(page_content)
        current_token_count += page_tokens

    if current_subset:
        subsets.append("".join(current_subset))

    print(f"Divided page_list into {len(subsets)} groups")
    return subsets


def extract_matching_page_pairs(
    toc_page: list[dict],
    toc_physical_index: list[dict],
    start_page_index: int,
) -> list[dict]:
    """Extract matching pairs between TOC pages and physical indices."""
    pairs = []
    for phy_item in toc_physical_index:
        for page_item in toc_page:
            if phy_item.get("title") == page_item.get("title"):
                physical_index = phy_item.get("physical_index")
                if physical_index is not None and int(physical_index) >= start_page_index:
                    pairs.append(
                        {
                            "title": phy_item.get("title"),
                            "page": page_item.get("page"),
                            "physical_index": physical_index,
                        }
                    )
    return pairs


def calculate_page_offset(pairs: list[dict]) -> int | None:
    """Calculate the offset between logical and physical page numbers."""
    differences = []
    for pair in pairs:
        try:
            physical_index = pair["physical_index"]
            page_number = pair["page"]
            difference = physical_index - page_number
            differences.append(difference)
        except (KeyError, TypeError):
            continue

    if not differences:
        return None

    difference_counts: dict[int, int] = {}
    for diff in differences:
        difference_counts[diff] = difference_counts.get(diff, 0) + 1

    return max(difference_counts.items(), key=lambda x: x[1])[0]


def add_page_offset_to_toc_json(data: list[dict], offset: int) -> list[dict]:
    """Add page offset to convert logical pages to physical indices."""
    for item in data:
        if item.get("page") is not None and isinstance(item["page"], int):
            item["physical_index"] = item["page"] + offset
            del item["page"]
    return data


def check_toc(
    page_list: list[tuple[str, int]],
    config: PageIndexConfig,
    llm: LLMClient,
) -> dict[str, Any]:
    """Check for and extract table of contents from document."""
    toc_page_list = find_toc_pages(
        start_page_index=0,
        page_list=page_list,
        config=config,
        llm=llm,
    )

    if len(toc_page_list) == 0:
        print("No TOC found")
        return {"toc_content": None, "toc_page_list": [], "page_index_given_in_toc": "no"}

    print("TOC found")
    toc_json = toc_extractor(page_list, toc_page_list, llm)

    if toc_json["page_index_given_in_toc"] == "yes":
        print("Page index found in TOC")
        return {
            "toc_content": toc_json["toc_content"],
            "toc_page_list": toc_page_list,
            "page_index_given_in_toc": "yes",
        }

    current_start_index = toc_page_list[-1] + 1

    while (
        toc_json["page_index_given_in_toc"] == "no"
        and current_start_index < len(page_list)
        and current_start_index < config.toc_check_page_num
    ):
        additional_toc_pages = find_toc_pages(
            start_page_index=current_start_index,
            page_list=page_list,
            config=config,
            llm=llm,
        )

        if len(additional_toc_pages) == 0:
            break

        additional_toc_json = toc_extractor(page_list, additional_toc_pages, llm)
        if additional_toc_json["page_index_given_in_toc"] == "yes":
            print("Page index found in TOC")
            return {
                "toc_content": additional_toc_json["toc_content"],
                "toc_page_list": additional_toc_pages,
                "page_index_given_in_toc": "yes",
            }

        current_start_index = additional_toc_pages[-1] + 1

    print("Page index not found in TOC")
    return {
        "toc_content": toc_json["toc_content"],
        "toc_page_list": toc_page_list,
        "page_index_given_in_toc": "no",
    }


async def check_title_appearance(
    item: dict,
    page_list: list[tuple[str, int]],
    start_index: int,
    llm: LLMClient,
) -> dict:
    """Check if a title appears on its indicated page."""
    title = item["title"]
    if "physical_index" not in item or item["physical_index"] is None:
        return {
            "list_index": item.get("list_index"),
            "answer": "no",
            "title": title,
            "page_number": None,
        }

    page_number = item["physical_index"]
    page_text = page_list[page_number - start_index][0]

    prompt = f"""
Your job is to check if the given section appears or starts in the given page_text.

Note: do fuzzy matching, ignore any space inconsistency in the page_text.

The given section title is {title}.
The given page_text is {page_text}.

Reply format:
{{
    "thinking": "<your reasoning>",
    "answer": "yes" or "no" (yes if the section appears or starts in the page_text, no otherwise)
}}
Directly return the final JSON structure. Do not output anything else."""

    response = await llm.chat_async(prompt)
    response_json = extract_json(response)
    answer = response_json.get("answer", "no")

    return {
        "list_index": item.get("list_index"),
        "answer": answer,
        "title": title,
        "page_number": page_number,
    }


async def check_title_appearance_in_start(
    title: str,
    page_text: str,
    llm: LLMClient,
    logger: Any = None,
) -> str:
    """Check if a title appears at the start of a page."""
    prompt = f"""
You will be given the current section title and the current page_text.
Your job is to check if the current section starts in the beginning of the given page_text.
If there are other contents before the current section title, then it does not start at the beginning.
If the current section title is the first content, then it starts at the beginning.

Note: do fuzzy matching, ignore any space inconsistency in the page_text.

The given section title is {title}.
The given page_text is {page_text}.

Reply format:
{{
    "thinking": "<your reasoning>",
    "start_begin": "yes" or "no"
}}
Directly return the final JSON structure. Do not output anything else."""

    response = await llm.chat_async(prompt)
    response_json = extract_json(response)
    if logger:
        logger.info(f"Response: {response_json}")
    return response_json.get("start_begin", "no")


async def check_title_appearance_in_start_concurrent(
    structure: list[dict],
    page_list: list[tuple[str, int]],
    llm: LLMClient,
    logger: Any = None,
) -> list[dict]:
    """Check title appearances concurrently."""
    if logger:
        logger.info("Checking title appearance in start concurrently")

    for item in structure:
        if item.get("physical_index") is None:
            item["appear_start"] = "no"

    tasks = []
    valid_items = []
    for item in structure:
        if item.get("physical_index") is not None:
            page_text = page_list[item["physical_index"] - 1][0]
            tasks.append(check_title_appearance_in_start(item["title"], page_text, llm, logger))
            valid_items.append(item)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item, result in zip(valid_items, results):
        if isinstance(result, Exception):
            if logger:
                logger.error(f"Error checking start for {item['title']}: {result}")
            item["appear_start"] = "no"
        else:
            item["appear_start"] = result

    return structure
