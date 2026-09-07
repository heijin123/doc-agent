"""摄取切块分发入口：探测文件类型 → 路由到对应格式的定制切片策略。

编排层（后续 M 的摄取流水线）只认识这一个函数：
    chunk_document(file_path) -> ChunkingResult
内部完成两件事：
1. 格式探测（file_type_detection，扩展名 + 内容嗅探）
2. 按 DocumentType 路由到分格式切块器，并统一补齐溯源元数据
   （block_seq 连续编号 / source_file / title / format）

格式策略一览（详见各 chunk_*.py 模块 docstring）：
    pdf     页为块（页内超长二次切）          —— chunk_pdf
    word    标题结构切（表格转 md 保留二维）   —— chunk_docx
    markdown 标题结构切（section=标题链）      —— chunk_markdown
    txt     段落为块                        —— chunk_plain_text
    excel   一行 = 一条记录                  —— chunk_excel
    json    一个数组元素/顶层键 = 一块        —— chunk_json
    slides  一页 = 一块                     —— chunk_ppt

不支持/无法识别的文件抛 UnsupportedFileError / ValueError，
由摄取编排 catch 后记入该文档的 FAIL 报告（demo1 教训：单文档失败不拖垮全批）。
"""
from __future__ import annotations

from pathlib import Path

from .chunk_docx import chunk_docx_file
from .chunk_excel import chunk_excel_file
from .chunk_json import chunk_json_file
from .chunk_markdown import chunk_markdown_text
from .chunk_models import ChunkingResult, DocumentChunk
from .chunk_pdf import chunk_pdf_file
from .chunk_plain_text import chunk_plain_text_file
from .chunk_ppt import chunk_pptx_file
from .file_type_detection import DocumentType, detect_file_type

# 各格式在溯源元数据里的 format 值（覆盖切块器内部默认，统一由探测结果决定）
_CHUNK_FORMAT_BY_TYPE = {
    DocumentType.PDF: "pdf",
    DocumentType.WORD: "word",
    DocumentType.MARKDOWN: "markdown",
    DocumentType.EXCEL: "excel",
    DocumentType.JSON: "json",
    DocumentType.PLAIN_TEXT: "txt",
    DocumentType.SLIDES: "slides",
}


def chunk_document(file_path: Path) -> ChunkingResult:
    """探测并切分单个文档，返回带溯源元数据的块列表。"""
    file_path = Path(file_path)
    detection = detect_file_type(file_path)
    source_file = file_path.name
    document_title = file_path.stem

    document_type = detection.document_type
    if document_type == DocumentType.PDF:
        chunks = chunk_pdf_file(file_path, source_file, document_title)
    elif document_type == DocumentType.WORD:
        chunks = chunk_docx_file(file_path, source_file, document_title)
    elif document_type == DocumentType.MARKDOWN:
        markdown_text = file_path.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_markdown_text(markdown_text, source_file, document_title)
    elif document_type == DocumentType.EXCEL:
        chunks = chunk_excel_file(file_path, source_file, document_title)
    elif document_type == DocumentType.JSON:
        chunks = chunk_json_file(file_path, source_file, document_title)
    elif document_type == DocumentType.PLAIN_TEXT:
        chunks = chunk_plain_text_file(file_path, source_file, document_title)
    elif document_type == DocumentType.SLIDES:
        chunks = chunk_pptx_file(file_path, source_file, document_title)
    else:
        raise ValueError(f"未支持的文档类型：{document_type}")

    chunks = _complete_chunk_metadata(chunks, source_file, document_title, document_type)
    return ChunkingResult(
        source_file=source_file,
        document_type=document_type.value,
        detected_by=detection.detected_by,
        chunks=chunks,
    )


def _complete_chunk_metadata(
    chunks: list[DocumentChunk],
    source_file: str,
    document_title: str,
    document_type: DocumentType,
) -> list[DocumentChunk]:
    """统一补齐溯源键：block_seq 连续编号、format、缺失的 source_file/title。"""
    format_value = _CHUNK_FORMAT_BY_TYPE[document_type]
    for block_index, chunk in enumerate(chunks):
        metadata = chunk.metadata
        metadata["block_seq"] = block_index  # 0 起始连续编号，doc_id 前缀后成库内唯一 id
        metadata["format"] = format_value
        metadata["source_file"] = source_file
        metadata["title"] = document_title
    return chunks
