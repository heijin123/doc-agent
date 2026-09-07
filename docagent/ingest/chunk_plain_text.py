"""纯文本(.txt) 切片策略：段落为语义单元，段落过长按句子二次切。

不复用 markdown 切块器：txt 里行首 # 是普通文本（shell 注释、C# 字样、
会议纪要的编号），误当标题会把正文切碎。段落（空行分隔）是 txt 里
作者唯一留下的结构信号，也是检索时最自然的"问题单元"。

无标题 → section 溯源字段为空（纯文本没有章节概念，不强造）。
"""
from __future__ import annotations

from pathlib import Path

from .chunk_models import MAX_CHUNK_CHARS, DocumentChunk, split_oversized_text


def chunk_plain_text_file(file_path: Path, source_file: str, document_title: str) -> list[DocumentChunk]:
    """读取 .txt，按段落切成知识块。"""
    raw_text = file_path.read_text(encoding="utf-8", errors="replace")
    chunks: list[DocumentChunk] = []
    current_paragraph_parts: list[str] = []

    def flush_paragraph() -> None:
        nonlocal current_paragraph_parts
        paragraph_text = "\n".join(current_paragraph_parts).strip()
        current_paragraph_parts = []
        if not paragraph_text:
            return
        if len(paragraph_text) <= MAX_CHUNK_CHARS:
            chunks.append(_make_chunk(paragraph_text, source_file, document_title))
        else:
            for sub_text in split_oversized_text(paragraph_text):
                chunks.append(_make_chunk(sub_text, source_file, document_title))

    for raw_line in raw_text.splitlines():
        if raw_line.strip():
            current_paragraph_parts.append(raw_line.strip())
        else:
            flush_paragraph()
    flush_paragraph()
    return chunks


def _make_chunk(text: str, source_file: str, document_title: str) -> DocumentChunk:
    metadata = {
        "doc_id": None,
        "title": document_title,
        "page": None,
        "section": None,
        "block_type": "paragraph",
        "source_file": source_file,
        "format": "txt",
    }
    return DocumentChunk(text=text, metadata=metadata)
