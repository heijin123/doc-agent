"""PPT(.pptx) 切片策略：一页幻灯片 = 一个知识块。

PPT 信息密度低但结构强：每页是一个独立主题（标题即主题），天然对齐
"问题单元"。正文通常稀疏，单页极少超长，页边界切分足够。

实现：
- 每页取标题（若有）作 section，正文取所有含文字形状的文本拼合
- 纯图页无文字 → 跳过（M2 VLM 补全节点转录）
- 页内文本超长（信息密集页）按句子二次切，子块仍带页号

v1 取舍：嵌套形状组(group)内的文本、表格形状、备注页不取——演示语料
为常规商务页（标题+要点）；真实复杂 PPT 需要展开 shape 树，M2 再做。
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from .chunk_models import MAX_CHUNK_CHARS, DocumentChunk, split_oversized_text


def chunk_pptx_file(file_path: Path, source_file: str, document_title: str) -> list[DocumentChunk]:
    """读取 .pptx，每页切成一个（或多个）知识块。"""
    presentation = Presentation(str(file_path))
    chunks: list[DocumentChunk] = []
    for slide_index, slide in enumerate(presentation.slides):
        slide_number = slide_index + 1
        slide_title = _extract_slide_title(slide)
        body_text = _extract_slide_body_text(slide)
        if not body_text.strip():
            continue  # 纯图页：留给 VLM 慢路径

        section_title = slide_title or f"第 {slide_number} 页"
        if len(body_text) > MAX_CHUNK_CHARS:
            for sub_text in split_oversized_text(body_text):
                chunks.append(_build_chunk(sub_text, slide_number, section_title, source_file, document_title))
        else:
            chunks.append(_build_chunk(body_text, slide_number, section_title, source_file, document_title))
    return chunks


def _extract_slide_title(slide) -> str | None:
    """取占位符标题文字；无标题版式的页面返回 None。"""
    if slide.shapes.title is not None and slide.shapes.title.has_text_frame:
        title_text = slide.shapes.title.text.strip()
        if title_text:
            return title_text
    return None


def _extract_slide_body_text(slide) -> str:
    """拼合页面所有形状的文字（标题外的正文/文本框），按形状顺序换行分隔。"""
    text_parts: list[str] = []
    for shape in slide.shapes:
        if shape == slide.shapes.title:
            continue  # 标题单独进 section，不重复进正文
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            continue  # v1 取舍：组内文本框不展开（见模块 docstring）
        if shape.has_text_frame:
            shape_text = shape.text_frame.text.strip()
            if shape_text:
                text_parts.append(shape_text)
    return "\n".join(text_parts)


def _build_chunk(
    text: str,
    slide_number: int,
    section_title: str,
    source_file: str,
    document_title: str,
) -> DocumentChunk:
    metadata = {
        "doc_id": None,
        "title": document_title,
        "page": slide_number,  # 借用 page 存幻灯片页码，溯源显示"第 N 页"
        "section": section_title,
        "block_type": "slide",
        "source_file": source_file,
        "format": "slides",
    }
    return DocumentChunk(text=text, metadata=metadata)
