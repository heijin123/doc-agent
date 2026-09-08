"""PPT(.pptx) 切片策略：一页幻灯片 = 一个知识块。

PPT 信息密度低但结构强：每页是一个独立主题（标题即主题），天然对齐
"问题单元"。正文通常稀疏，单页极少超长，页边界切分足够。

实现：
- 每页取标题（若有）作 section，正文按形状顺序拼合文本与图片占位符
- 图片（picture 形状）以 [IMAGE:{image_id}] 占位符进正文流（随形状顺序），
  字节收集进 images 待仓库落盘——"图不丢、召回后可展示"（2026-09-08 起）
- 纯图页无文字 → 跳过（图内信息留给 M2 VLM 转录兜底，不产纯占位块）

v1 取舍：嵌套形状组(group)内的文本与图片、表格形状、备注页不取——
演示语料为常规商务页（标题+要点）；真实复杂 PPT 需展开 shape 树，M2 再做。
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from .chunk_models import MAX_CHUNK_CHARS, DocumentChunk, ExtractedImage, make_image_id, split_oversized_text


def chunk_pptx_file(
    file_path: Path, source_file: str, document_title: str
) -> tuple[list[DocumentChunk], list[ExtractedImage]]:
    """读取 .pptx，每页切成一个（或多个）知识块；提取 picture 图片。"""
    source_stem = Path(source_file).stem
    presentation = Presentation(str(file_path))
    chunks: list[DocumentChunk] = []
    images: list[ExtractedImage] = []
    seen_image_ids: set[str] = set()

    for slide_index, slide in enumerate(presentation.slides):
        slide_number = slide_index + 1
        slide_title = _extract_slide_title(slide)
        body_text, fresh_images = _extract_slide_body_stream(
            slide, slide_title, source_stem, seen_image_ids
        )
        if not body_text.strip():
            continue  # 纯图页：留给 VLM 慢路径
        images.extend(fresh_images)

        section_title = slide_title or f"第 {slide_number} 页"
        if len(body_text) > MAX_CHUNK_CHARS:
            for sub_text in split_oversized_text(body_text):
                chunks.append(_build_chunk(sub_text, slide_number, section_title, source_file, document_title))
        else:
            chunks.append(_build_chunk(body_text, slide_number, section_title, source_file, document_title))
    return chunks, images


def _extract_slide_title(slide) -> str | None:
    """取占位符标题文字；无标题版式的页面返回 None。"""
    if slide.shapes.title is not None and slide.shapes.title.has_text_frame:
        title_text = slide.shapes.title.text.strip()
        if title_text:
            return title_text
    return None


def _extract_slide_body_stream(
    slide, slide_title: str | None, source_stem: str, seen_image_ids: set[str]
) -> tuple[str, list[ExtractedImage]]:
    """按形状顺序拼合页面正文：文本框文字与 picture 图占位符交错出现。"""
    text_parts: list[str] = []
    fresh_images: list[ExtractedImage] = []

    for shape in slide.shapes:
        if shape == slide.shapes.title:
            continue  # 标题单独进 section，不重复进正文
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            continue  # v1 取舍：组内形状不展开（见模块 docstring）
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            placeholder, image = _extract_picture_shape(shape, source_stem, seen_image_ids)
            if image is not None:
                fresh_images.append(image)
            if placeholder:
                text_parts.append(placeholder)
            continue
        if shape.has_text_frame:
            shape_text = shape.text_frame.text.strip()
            if shape_text:
                text_parts.append(shape_text)
    return "\n".join(text_parts), fresh_images


def _extract_picture_shape(
    shape, source_stem: str, seen_image_ids: set[str]
) -> tuple[str | None, ExtractedImage | None]:
    """单个 picture 形状 → (占位符行, 新图)；损坏/外链图跳过不阻断整页。"""
    try:
        image_part = shape.image  # 无 image 属性或损坏时抛异常
        image_bytes = image_part.blob
    except Exception:
        return None, None
    if not image_bytes:
        return None, None

    image_ext = getattr(image_part, "ext", None) or "png"
    image_id = make_image_id(source_stem, image_bytes)
    if image_id in seen_image_ids:
        return f"[IMAGE:{image_id}]", None
    seen_image_ids.add(image_id)
    return f"[IMAGE:{image_id}]", ExtractedImage(image_id=image_id, data=image_bytes, ext=image_ext)


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
