"""摄取切块分发入口：探测文件类型 → 路由到对应格式的定制切片策略。

编排层（后续 M 的摄取流水线）只认识这一个函数：
    chunk_document(file_path) -> ChunkingResult
内部完成两件事：
1. 格式探测（file_type_detection，扩展名 + 内容嗅探）
2. 按 DocumentType 路由到分格式切块器，并统一补齐溯源元数据
   （block_seq 连续编号 / source_file / title / format / image_ids）

图片（2026-09-08 起）：pdf/docx/slides 切块器返回 (chunks, images)——图片字节
不在此落盘（保持本层纯函数可测），收集进 ChunkingResult.images 交编排层
save_images 落盘登记；块文本中的 [IMAGE:{id}] 占位符统一解析进 metadata.image_ids
（逗号串，Chroma 标量约束），检索命中后展示层按 id 取图填充。

格式策略一览（详见各 chunk_*.py 模块 docstring）：
    pdf     页为块（页内超长二次切）+ 版面块定位插图  —— chunk_pdf
    word    标题结构切（表格转 md 保留二维）+ 内联图   —— chunk_docx
    markdown 标题结构切（section=标题链）              —— chunk_markdown
    txt     段落为块                                 —— chunk_plain_text
    excel   一行 = 一条记录                           —— chunk_excel
    json    一个数组元素/顶层键 = 一块                 —— chunk_json
    slides  一页 = 一块 + picture 图                  —— chunk_ppt

不支持/无法识别的文件抛 UnsupportedFileError / ValueError，
由摄取编排 catch 后记入该文档的 FAIL 报告（demo1 教训：单文档失败不拖垮全批）。
"""
from __future__ import annotations

from pathlib import Path

from .chunk_docx import chunk_docx_file
from .chunk_excel import chunk_excel_file
from .chunk_json import chunk_json_file
from .chunk_markdown import chunk_markdown_text
from .chunk_models import (
    IMAGE_PLACEHOLDER_PATTERN,
    ChunkingResult,
    DocumentChunk,
    ExtractedImage,
    extract_doc_date,
)
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


def chunk_document(file_path: Path, vlm_transcripts: dict[int, str] | None = None) -> ChunkingResult:
    """探测并切分单个文档，返回带溯源元数据的块列表与提取的图片。

    vlm_transcripts（M2 慢路径注入，仅 PDF 生效）：{页号: VLM 转录文本}——
    对应红页用转录文本成块（figure_transcript），其余页快路径不变。
    """
    file_path = Path(file_path)
    detection = detect_file_type(file_path)
    source_file = file_path.name
    document_title = file_path.stem

    document_type = detection.document_type
    chunks: list[DocumentChunk]
    images: list[ExtractedImage]
    if document_type == DocumentType.PDF:
        chunks, images = chunk_pdf_file(file_path, source_file, document_title, vlm_transcripts)
    elif document_type == DocumentType.WORD:
        chunks, images = chunk_docx_file(file_path, source_file, document_title)
    elif document_type == DocumentType.SLIDES:
        chunks, images = chunk_pptx_file(file_path, source_file, document_title)
    elif document_type == DocumentType.MARKDOWN:
        markdown_text = file_path.read_text(encoding="utf-8", errors="replace")
        chunks, images = chunk_markdown_text(markdown_text, source_file, document_title), []
    elif document_type == DocumentType.EXCEL:
        chunks, images = chunk_excel_file(file_path, source_file, document_title), []
    elif document_type == DocumentType.JSON:
        chunks, images = chunk_json_file(file_path, source_file, document_title), []
    elif document_type == DocumentType.PLAIN_TEXT:
        chunks, images = chunk_plain_text_file(file_path, source_file, document_title), []
    else:
        raise ValueError(f"未支持的文档类型：{document_type}")

    _complete_chunk_metadata(chunks, source_file, document_title, document_type)
    return ChunkingResult(
        source_file=source_file,
        document_type=document_type.value,
        detected_by=detection.detected_by,
        chunks=chunks,
        images=images,
    )


def _complete_chunk_metadata(
    chunks: list[DocumentChunk],
    source_file: str,
    document_title: str,
    document_type: DocumentType,
) -> None:
    """统一补齐溯源键：block_seq 连续编号、format、缺失的 source_file/title、image_ids。

    文档日期（doc_date/doc_year）也从标题在此统一注入（2026-09-08 起，防跨年报表
    检索张冠李戴）：年份解析不到（手册/论文等无日期文档）就不写键，问答年份过滤
    时这类块被排除属预期。ingested_at 由 store 层写（落库时刻，此处不管）。
    """
    format_value = _CHUNK_FORMAT_BY_TYPE[document_type]
    doc_date = extract_doc_date(document_title)
    for block_index, chunk in enumerate(chunks):
        metadata = chunk.metadata
        metadata["block_seq"] = block_index  # 0 起始连续编号，doc_id 前缀后成库内唯一 id
        metadata["format"] = format_value
        metadata["source_file"] = source_file
        metadata["title"] = document_title
        if doc_date:
            metadata["doc_date"] = doc_date
            metadata["doc_year"] = int(doc_date[:4])
        image_ids = IMAGE_PLACEHOLDER_PATTERN.findall(chunk.text)
        if image_ids:
            # 去重保序（同块同图多处引用只记一次）；逗号串满足 Chroma 标量约束
            metadata["image_ids"] = ",".join(dict.fromkeys(image_ids))
