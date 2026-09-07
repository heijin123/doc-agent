"""markdown 结构切块——md/docx 两类格式的核心切片策略。

原理：标题行是文档作者切好的"天然章节边界"。逐行扫描，遇标题行即开新块，
标题行本身并入块首（块带题，检索命中即知道所属章节）。
- section 溯源字段 = 当前标题链（"第 2 章 / 2.1 安装"），展示出处比单标题更精确
- 代码围栏(```)内的 # 不算标题（防代码行污染章节）
- 无标题的连续长文：空行分段；单段超 MAX_CHUNK_CHARS 按句子二次切并带重叠
- 不去重合并相邻同题块（demo1 期 MarkdownHeaderTextSplitter 同标题合并坑：73→20，
  SKU 张冠李戴）——宁可多块，不让内容错位

txt 是另一套段落切（行首 # 可能是普通文本），见 chunk_plain_text.py。
"""
from __future__ import annotations

import re

from .chunk_models import (
    MAX_CHUNK_CHARS,
    DocumentChunk,
    split_oversized_text,
)

_HEADING_LINE_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")


def chunk_markdown_text(markdown_text: str, source_file: str, document_title: str) -> list[DocumentChunk]:
    """按标题结构把整篇 markdown 切成块。docx 先归一化成 markdown 文本再调本函数。"""
    chunks: list[DocumentChunk] = []
    current_section_headings: list[str] = []   # 标题链栈：按标题级别裁剪
    current_text_parts: list[str] = []
    current_section_title = ""                  # 当前块的 section 值（快照，防后续栈变化影响）
    in_code_fence = False

    def flush_current_block() -> None:
        """把累积的文本按块大小规则落成 1..n 个 DocumentChunk。"""
        nonlocal current_text_parts
        if not current_text_parts:
            return
        full_text = "\n".join(current_text_parts).strip()
        current_text_parts = []
        if not full_text:
            return
        if len(full_text) <= MAX_CHUNK_CHARS:
            chunks.append(_make_chunk(full_text, current_section_title, "section", source_file, document_title))
            return
        # 长块二次切：优先按段落（空行）聚合到目标大小，无段落则按句子硬切
        for sub_text in split_oversized_text(full_text):
            chunks.append(_make_chunk(sub_text, current_section_title, "paragraph", source_file, document_title))

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            current_text_parts.append(line)
            continue
        if in_code_fence:
            current_text_parts.append(line)
            continue

        heading_match = _HEADING_LINE_PATTERN.match(stripped)
        if heading_match:
            flush_current_block()
            heading_level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            # 标题链按级别裁剪：同级/更高级标题到来，旧的低级标题出栈
            current_section_headings = current_section_headings[: heading_level - 1]
            current_section_headings.append(heading_text)
            current_section_title = " / ".join(current_section_headings)
            # 标题行进入块首，保证块自带标题可检索
            current_text_parts.append(heading_text)
            continue

        current_text_parts.append(line)

    flush_current_block()
    return chunks


def _make_chunk(
    text: str,
    section_title: str,
    block_type: str,
    source_file: str,
    document_title: str,
) -> DocumentChunk:
    metadata = {
        "doc_id": None,           # 落库时由摄取编排生成（每文档一个）
        "title": document_title,  # 文档标题（默认文件名去扩展名）
        "section": section_title or None,
        "block_type": block_type,
        "source_file": source_file,
        "format": "markdown",
    }
    return DocumentChunk(text=text, metadata=metadata)
