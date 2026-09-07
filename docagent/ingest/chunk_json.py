"""JSON 切片策略：结构数据按元素切——一个数组元素 / 一个顶层键 = 一个知识块。

json 与 excel 同属"结构数据"，但层级比表格自由，切法按顶层形态分三种：
- 顶层数组（最常见：商品列表/记录集合）→ 每个元素一块，metadata 带路径 "$[0]"
- 顶层对象 → 每个顶层键一组（如 {"商品": {...}, "售后": {...}} 两个主题块）
- 顶层标量 → 整块（罕见，兜底）

对象值统一扁平化成 "a.b：值" 多行文本：检索时键路径与值都在同一块文本里，
命中"价格"或命中"199"都能回到同一块；键路径保留结构语义，比只拼值更可读。

溯源：json_path 存到 section 字段（"$[2].specs"），答案出处能精确到元素。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .chunk_models import MAX_CHUNK_CHARS, DocumentChunk, split_oversized_text


def chunk_json_file(file_path: Path, source_file: str, document_title: str) -> list[DocumentChunk]:
    """读取 .json（UTF-8/带 BOM），按结构元素切成知识块。"""
    raw_text = file_path.read_text(encoding="utf-8-sig", errors="replace")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as decode_error:
        # 探测阶段已认 JSON，这里仍可能解析失败（如截断文件）——抛给编排层记 FAIL
        raise ValueError(f"JSON 解析失败（第 {decode_error.lineno} 行）：{decode_error.msg}")

    if isinstance(data, list):
        elements = [(f"$[{index}]", element) for index, element in enumerate(data)]
    elif isinstance(data, dict):
        elements = [(f"$.{key}", value) for key, value in data.items()]
    else:
        elements = [("$", data)]

    chunks: list[DocumentChunk] = []
    for json_path, element in elements:
        chunks.extend(_chunk_one_element(element, json_path, source_file, document_title))
    return chunks


def _chunk_one_element(
    element: Any,
    json_path: str,
    source_file: str,
    document_title: str,
) -> list[DocumentChunk]:
    """一个元素切成 1..n 块。

    对象数组（最常见：记录集合）→ 逐条成块，路径带下标；
    标量/混合数组 → 整组一块（元素太碎，逐条入库无检索价值且难溯源）。
    """
    if isinstance(element, list):
        if element and all(isinstance(item, dict) for item in element):
            sub_chunks: list[DocumentChunk] = []
            for index, item in enumerate(element):
                sub_chunks.extend(_chunk_one_element(item, f"{json_path}[{index}]", source_file, document_title))
            return sub_chunks
        element_text = str(element)
        if len(element_text) > MAX_CHUNK_CHARS:
            return [
                _build_chunk(sub_text, json_path, source_file, document_title)
                for sub_text in split_oversized_text(element_text)
            ]
        return [_build_chunk(element_text, json_path, source_file, document_title)]

    if isinstance(element, dict):
        text = _flatten_dict_to_text(element, json_path)
    else:
        text = str(element)

    if len(text) > MAX_CHUNK_CHARS:
        return [
            _build_chunk(sub_text, json_path, source_file, document_title)
            for sub_text in split_oversized_text(text)
        ]
    return [_build_chunk(text, json_path, source_file, document_title)]


def _flatten_dict_to_text(data: dict, root_path: str) -> str:
    """字典扁平化成 "键路径：值" 多行文本；嵌套键用点路径（specs.weight）。"""
    lines: list[str] = []

    def collect(current: Any, path_prefix: str) -> None:
        if isinstance(current, dict):
            for key, value in current.items():
                collect(value, f"{path_prefix}.{key}" if path_prefix else str(key))
        elif isinstance(current, list):
            for index, item in enumerate(current):
                collect(item, f"{path_prefix}[{index}]")
        else:
            lines.append(f"{path_prefix}：{current}")

    collect(data, root_path)
    return "\n".join(lines)


def _build_chunk(text: str, json_path: str, source_file: str, document_title: str) -> DocumentChunk:
    metadata = {
        "doc_id": None,
        "title": document_title,
        "section": json_path,
        "block_type": "json_item",
        "source_file": source_file,
        "format": "json",
    }
    return DocumentChunk(text=text, metadata=metadata)
