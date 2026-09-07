"""PDF 切片策略：一页 = 一个知识块（页内超长再按段落/句子二次切）。

PDF 没有可靠的标题层级（文字层只是坐标+字体），页边界是唯一稳定结构——
现实文档一页通常恰是"一个主题"（产品规格一页、发票一页、论文一页一个要点），
且检索时"答案在第几页"是溯源最常用的定位，page 进 metadata。

快路径语义（v1）：
- 只处理有文字层的页；纯图/扫描页 get_text 为空 → 跳过（质量门判级 + VLM
  转录在 M2 补全节点接入，本切块器不越权做 OCR）
- 双栏/乱码等复杂版面问题由摄取编排的质量门标黄/红后走慢路径，这里不做
  启发式猜测（解析实验已证：规则猜版面既贵又错）

溯源：page 存人类页码（1 起始）；页内二次切的子块仍带同页号。
"""
from __future__ import annotations

from pathlib import Path

import pymupdf

from .chunk_models import MAX_CHUNK_CHARS, DocumentChunk, split_oversized_text


def chunk_pdf_file(file_path: Path, source_file: str, document_title: str) -> list[DocumentChunk]:
    """读取 PDF，按页切成知识块。"""
    chunks: list[DocumentChunk] = []
    with pymupdf.open(str(file_path)) as document:
        for page_index, page in enumerate(document):
            page_text = page.get_text().strip()
            if not page_text:
                continue  # 纯图/扫描页：留给质量门/VLM 慢路径
            page_number = page_index + 1  # 人类页码 1 起始
            if len(page_text) > MAX_CHUNK_CHARS:
                for sub_text in split_oversized_text(page_text):
                    chunks.append(_build_chunk(sub_text, page_number, source_file, document_title))
            else:
                chunks.append(_build_chunk(page_text, page_number, source_file, document_title))
    return chunks


def _build_chunk(text: str, page_number: int, source_file: str, document_title: str) -> DocumentChunk:
    metadata = {
        "doc_id": None,
        "title": document_title,
        "page": page_number,
        "section": f"第 {page_number} 页",
        "block_type": "page",
        "source_file": source_file,
        "format": "pdf",
    }
    return DocumentChunk(text=text, metadata=metadata)
