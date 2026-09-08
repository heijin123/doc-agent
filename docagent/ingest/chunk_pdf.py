"""PDF 切片策略：一页 = 一个知识块（页内超长再按段落/句子二次切）。

PDF 没有可靠的标题层级（文字层只是坐标+字体），页边界是唯一稳定结构——
现实文档一页通常恰是"一个主题"（产品规格一页、发票一页、论文一页一个要点），
且检索时"答案在第几页"是溯源最常用的定位，page 进 metadata。

图片引用（2026-09-08 起）：
- 页面经 get_text("dict") 取 text/image 两类版面块，按阅读顺序（y 分桶 + x）
  合并为"文本流"——图片在原位写成 [IMAGE:{image_id}] 占位符（随块入库，
  展示层召回后替换为 <img>）；图字节收集进 ChunkingResult.images 待仓库落盘。
- 纯图/扫描页（无文本块）仍跳过——图内信息留给 M2 VLM 转录兜底，本层不越权。
- 双栏/乱码等复杂版面仍由质量门标黄/红后走慢路径，不做启发式猜测。

文本渲染注意：span 是同一行内按样式切分的片段，直接拼接会丢词间空格
（"foo bar" 在 span 边界 → "foobar"）——按 span 间隙 ≥ 0.35×字号 补空格。

溯源：page 存人类页码（1 起始）；页内二次切的子块仍带同页号。
"""
from __future__ import annotations

from pathlib import Path

import pymupdf

from .chunk_models import (
    MAX_CHUNK_CHARS,
    DocumentChunk,
    ExtractedImage,
    make_image_id,
    split_oversized_text,
)

_SPAN_GAP_RATIO = 0.35  # 相邻 span 间隙 ≥ 0.35×字号 → 视为词间空格
_LINE_BUCKET_POINTS = 8  # y 坐标分桶容差（同一视觉行的文本/图算同排）


def chunk_pdf_file(
    file_path: Path,
    source_file: str,
    document_title: str,
    vlm_transcripts: dict[int, str] | None = None,
) -> tuple[list[DocumentChunk], list[ExtractedImage]]:
    """读取 PDF，按页切成知识块；同时提取页内图片（占位符已写入块文本）。

    vlm_transcripts（M2 慢路径注入）：{页号: 转录文本}——红页原文文本不可信，
    不进库；转录文本作为 figure_transcript 块**追加到文档块流尾部**（老块 id
    全部不动 → 无 block_seq 平移、无孤儿；幂等靠转录文本 md5）。历史库中
    已入库的乱码块由显式 delete_doc_blocks 重建清理（业务层职责）。默认
    None = 纯快路径（含红页原文，M1 行为，供无 Key 降级）。
    """
    source_stem = Path(source_file).stem
    chunks: list[DocumentChunk] = []
    all_images: list[ExtractedImage] = []
    seen_image_ids: set[str] = set()
    transcripts = vlm_transcripts or {}
    tail_transcript_chunks: list[tuple[str, int]] = []  # (转录文本, 页号) 待尾部追加

    with pymupdf.open(str(file_path)) as document:
        for page_index, page in enumerate(document):
            page_number = page_index + 1  # 人类页码 1 起始
            transcript_text = transcripts.get(page_number)
            if transcript_text is not None:
                # 红页：原文不产块；转录文本统一收集到尾部（避免本页块数变化平移后续 id）
                tail_transcript_chunks.append((transcript_text, page_number))
                continue
            page_text, page_images = _extract_page_stream(page, document, source_stem, seen_image_ids)
            if not page_text.strip():
                continue  # 纯图/扫描页：留给质量门/VLM 慢路径
            all_images.extend(page_images)
            if len(page_text) > MAX_CHUNK_CHARS:
                for sub_text in split_oversized_text(page_text):
                    chunks.append(_build_chunk(sub_text, page_number, "page", source_file, document_title))
            else:
                chunks.append(_build_chunk(page_text, page_number, "page", source_file, document_title))

    # 转录块尾部追加（block_seq 由分发层统一连续编号，老块序号不受影响）
    for transcript_text, page_number in tail_transcript_chunks:
        if len(transcript_text) > MAX_CHUNK_CHARS:
            for sub_text in split_oversized_text(transcript_text):
                chunks.append(_build_chunk(sub_text, page_number, "figure_transcript", source_file, document_title))
        else:
            chunks.append(_build_chunk(transcript_text, page_number, "figure_transcript", source_file, document_title))
    return chunks, all_images


def _extract_page_stream(page, document, source_stem: str, seen_image_ids: set[str]):
    """把一页转成"文本 + 图片占位"交错的阅读流；返回 (page_text, 新增图片列表)。"""
    page_blocks = page.get_text("dict")["blocks"]
    text_blocks = [block for block in page_blocks if block["type"] == 0]
    image_infos = page.get_image_info(xrefs=True)  # 每项含 bbox/xref/width/height

    # 按阅读顺序合并文本块与图块：y 分桶（同排）后再按 x 从左到右
    items: list[tuple[str, object]] = []
    for block in text_blocks:
        items.append(("text", block))
    for info in image_infos:
        items.append(("image", info))
    items.sort(key=lambda kind_and_payload: _reading_order_key(kind_and_payload[1]))

    text_parts: list[str] = []
    fresh_images: list[ExtractedImage] = []
    for kind, payload in items:
        if kind == "text":
            text_parts.append(_render_text_block(payload))
        else:
            xref = payload.get("xref") or 0
            if not xref:
                continue
            extracted = document.extract_image(xref)  # 可能抛异常（损坏流）→ 外层兜
            image_bytes = extracted["image"]
            image_ext = extracted.get("ext") or "png"
            image_id = make_image_id(source_stem, image_bytes)
            text_parts.append(f"[IMAGE:{image_id}]")
            if image_id not in seen_image_ids:
                seen_image_ids.add(image_id)
                fresh_images.append(ExtractedImage(image_id=image_id, data=image_bytes, ext=image_ext))
    return "\n".join(text_parts), fresh_images


def _reading_order_key(payload: dict) -> tuple[int, float]:
    """版面块排序键：y 分桶（同行容差）优先，再按 x。"""
    bounding_box = payload["bbox"]
    return (round(bounding_box[1] / _LINE_BUCKET_POINTS), bounding_box[0])


def _render_text_block(block: dict) -> str:
    """文本版面块 → 多行文本（span 间隙补空格，行间保留换行）。"""
    lines = []
    for line in block.get("lines", []):
        rendered_line = _render_line_spans(line.get("spans", []))
        if rendered_line:
            lines.append(rendered_line)
    return "\n".join(lines)


def _render_line_spans(spans: list[dict]) -> str:
    """一行内多个 span：相邻 span 间隙 ≥ 0.35×字号 时补空格，防词被拼死。"""
    parts: list[str] = []
    previous_end_x: float | None = None
    for span in spans:
        span_text = span.get("text", "")
        span_bounding = span.get("bbox", (0, 0, 0, 0))
        font_size = span.get("size", 10)
        if previous_end_x is not None and span_bounding[0] - previous_end_x > _SPAN_GAP_RATIO * font_size:
            parts.append(" ")
        parts.append(span_text)
        previous_end_x = span_bounding[2]
    return "".join(parts).strip()


def _build_chunk(
    text: str,
    page_number: int,
    block_type: str,
    source_file: str,
    document_title: str,
) -> DocumentChunk:
    metadata = {
        "doc_id": None,
        "title": document_title,
        "page": page_number,
        "section": f"第 {page_number} 页",
        "block_type": block_type,
        "source_file": source_file,
        "format": "pdf",
    }
    return DocumentChunk(text=text, metadata=metadata)
