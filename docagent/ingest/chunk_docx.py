"""Word(.docx) 切片策略：标题结构切 + 表格转 markdown 保留二维结构。

实现路线 = 归一化复用：把 docx 的正文元素按出现顺序转成 markdown 文本流
（Heading 样式 → # 标题行；表格 → markdown 管道表格；普通段落原样），
再交给 chunk_markdown.chunk_markdown_text 做标题结构切。

为什么这样归一化而不是给 docx 单独写切块器：
- 标题层级（Heading 1/2/3，中文 Word 显示为"标题 1"）与 markdown 的 # 同构，
  归一化后天然获得"按标题切 + section 标题链"能力，不重复实现；
- 表格在正文里常夹在段落之间，python-docx 的 doc.paragraphs 会漏掉表格内文字，
  必须按 body 元素顺序遍历（iter_block_items 惯用法），转成 markdown 表格后
  表格文本不会被切碎（表格是"不可分割块"，切进两个块就都不可用）。

v1 取舍：图片不转录（占位注释说明图内容缺失），VLM 转录在 M2 补全节点接入。
"""
from __future__ import annotations

from pathlib import Path

import docx
from docx.document import Document as _Document
from docx.table import Table as _Table
from docx.text.paragraph import Paragraph as _Paragraph

from .chunk_markdown import chunk_markdown_text
from .chunk_models import DocumentChunk

# 归一化成 markdown 时的表格列分隔与标题标记
_TABLE_CELL_SEPARATOR = "|"
_TABLE_HEADER_SEPARATOR_ROW = "---"
_HEADING_PREFIX_BY_LEVEL = {1: "#", 2: "##", 3: "###"}  # 更深标题并入三级，不无限细分


def chunk_docx_file(file_path: Path, source_file: str, document_title: str) -> list[DocumentChunk]:
    """读取 .docx，按标题层级与表格结构切成知识块。"""
    document = docx.Document(str(file_path))
    markdown_lines = _convert_document_to_markdown_lines(document)
    markdown_text = "\n".join(markdown_lines)
    if not markdown_text.strip():
        return []
    return chunk_markdown_text(markdown_text, source_file, document_title)


def _convert_document_to_markdown_lines(document: _Document) -> list[str]:
    """把 docx 正文元素按出现顺序转成 markdown 行流（标题带 #、表格带 |）。"""
    markdown_lines: list[str] = []
    for block in _iter_block_items(document):
        if isinstance(block, _Paragraph):
            heading_level = _paragraph_heading_level(block)
            paragraph_text = block.text.strip()
            if not paragraph_text:
                continue
            if heading_level:
                prefix = _HEADING_PREFIX_BY_LEVEL.get(heading_level, "###")
                markdown_lines.append(f"{prefix} {paragraph_text}")
            else:
                markdown_lines.append(paragraph_text)
        elif isinstance(block, _Table):
            markdown_lines.extend(_render_table_as_markdown(block))
    return markdown_lines


def _iter_block_items(document: _Document):
    """按 body 元素顺序产出 段落/表格——python-docx 官方惯用法。

    为什么必须用这个：doc.paragraphs 只给正文段落、doc.tables 只给表格，
    两者在文档中的真实交错顺序会丢失（先表格后段落会排到段落后面）。
    """
    parent_element = document.element.body
    for child in parent_element.iterchildren():
        if child.tag.endswith("}p"):  # w:p 段落
            yield _Paragraph(child, document)
        elif child.tag.endswith("}tbl"):  # w:tbl 表格
            yield _Table(child, document)


def _paragraph_heading_level(paragraph: _Paragraph) -> int | None:
    """按段落样式名判断标题级别；中英文样式名都认（Heading 1 / 标题 1）。"""
    style_name = (paragraph.style.name if paragraph.style else "") or ""
    if style_name in ("Heading 1", "标题 1"):
        return 1
    if style_name in ("Heading 2", "标题 2"):
        return 2
    if style_name in ("Heading 3", "标题 3"):
        return 3
    return None


def _render_table_as_markdown(table: _Table) -> list[str]:
    """表格转 markdown 管道表格：首行当表头，内容单元格竖线转义防破坏列结构。"""
    rows_as_lists = [
        [cell.text.replace("\n", " ").replace("|", "\\|").strip() for cell in row.cells]
        for row in table.rows
    ]
    # 过滤全空行（合并单元格跨行时其余行为空）
    rows_as_lists = [row for row in rows_as_lists if any(cell for cell in row)]

    markdown_lines: list[str] = []
    for row_index, row_cells in enumerate(rows_as_lists):
        markdown_lines.append(_TABLE_CELL_SEPARATOR + _TABLE_CELL_SEPARATOR.join(row_cells) + _TABLE_CELL_SEPARATOR)
        if row_index == 0:  # 首行后紧跟分隔行，markdown 才识别为表格
            separators = [_TABLE_HEADER_SEPARATOR_ROW] * len(row_cells)
            markdown_lines.append(
                _TABLE_CELL_SEPARATOR + _TABLE_CELL_SEPARATOR.join(separators) + _TABLE_CELL_SEPARATOR
            )
    return markdown_lines
