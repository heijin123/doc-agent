"""文件类型探测：识别上传文档属于哪一类，驱动后续"分格式定制切片"。

两类信号（由快到慢、由准到兜底）：
1. 扩展名——用户正常命名时最直接；扩展名命中即信任（内容嗅探只做兜底）
2. 内容嗅探——扩展名缺失 / 未知 / 与内容不一致时，按文件头判定：
   - PDF        : 前 4 字节 `%PDF`
   - Office 新版 : ZIP 容器（PK 头）→ 列内部目录区分 docx(word/)/xlsx(xl/)/pptx(ppt/)
   - Office 旧版 : OLE 复合文档头（D0 CF 11 E0）→ 明确拒绝并提示转存新格式
   - 文本类     : 解码尝试 → JSON（首字符 {/[ + 可解析）/ markdown（行首 # 标题）/
                  纯文本兜底

不支持格式不静默跳过：抛 UnsupportedFileError，调用方转成摄取报告里的
"该文件无法入库 + 原因"（demo1 教训：单文档失败要隔离，不能拖垮全批）。
"""
from __future__ import annotations

import json
import re
import zipfile
from enum import Enum
from pathlib import Path

# 文件头魔数（十六进制字节）——内容嗅探用
_OLE_HEADER = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # 旧版 Office(.doc/.xls/.ppt) 复合文档
_ZIP_HEADER = b"PK\x03\x04"  # docx/xlsx/pptx 都是 zip 容器
_PDF_HEADER = b"%PDF"


class DocumentType(Enum):
    """摄取管线支持的文档大类——每个类型映射一份定制切片策略。"""

    PDF = "pdf"
    WORD = "word"          # .docx
    MARKDOWN = "markdown"  # .md / .markdown
    EXCEL = "excel"        # .xlsx / .xlsm
    JSON = "json"
    PLAIN_TEXT = "txt"
    SLIDES = "slides"      # .pptx


# 扩展名 → 类型（探测第一信号）；同一类型可有多个扩展名
EXTENSION_TO_TYPE: dict[str, DocumentType] = {
    ".pdf": DocumentType.PDF,
    ".docx": DocumentType.WORD,
    ".md": DocumentType.MARKDOWN,
    ".markdown": DocumentType.MARKDOWN,
    ".xlsx": DocumentType.EXCEL,
    ".xlsm": DocumentType.EXCEL,
    ".json": DocumentType.JSON,
    ".txt": DocumentType.PLAIN_TEXT,
    ".text": DocumentType.PLAIN_TEXT,
    ".pptx": DocumentType.SLIDES,
}

# 认识但不支持的老格式：报错信息给出补救动作，比笼统"不支持"有用
_UNSUPPORTED_LEGACY_EXTENSIONS: dict[str, str] = {
    ".doc": "旧版 Word(.doc) 暂不支持，请在 Office 中另存为 .docx 后上传",
    ".xls": "旧版 Excel(.xls) 暂不支持，请在 Office 中另存为 .xlsx 后上传",
    ".ppt": "旧版 PowerPoint(.ppt) 暂不支持，请在 Office 中另存为 .pptx 后上传",
}

# ZIP 容器内部目录特征 → 判断是哪一种 Office 文档
_ZIP_DIRECTORY_MARKERS: dict[str, DocumentType] = {
    "word/": DocumentType.WORD,
    "xl/": DocumentType.EXCEL,
    "ppt/": DocumentType.SLIDES,
}

# 文本解码尝试顺序：UTF-8 最普遍（带 BOM 变体），UTF-16 兜底（Windows 记事本另存），
# GBK 兜底（中文遗留编码）。解码成功 + 可打印占比达标才算"文本文件"。
_TEXT_ENCODINGS = ("utf-8-sig", "utf-16", "gbk")
_PRINTABLE_RATIO_THRESHOLD = 0.8

_HEADING_LINE_PATTERN = re.compile(r"^#{1,6}\s+\S")  # markdown 标题行，如 "## 安装说明"


class UnsupportedFileError(Exception):
    """文件无法按当前支持的格式解析。message 面向使用者，直接进摄取报告。"""


class DetectionResult:
    """探测结论：类型 + 依据（哪个信号判出来的，进摄取报告便于排查）。"""

    def __init__(self, document_type: DocumentType, detected_by: str, file_name: str):
        self.document_type = document_type
        self.detected_by = detected_by  # "extension" | "content_sniffing:xxx"
        self.file_name = file_name

    def __repr__(self) -> str:
        return f"DetectionResult({self.document_type.value}, by={self.detected_by})"


def detect_file_type(file_path: Path) -> DetectionResult:
    """判断文件属于哪种文档类型。

    抛出：
        UnsupportedFileError —— 扩展名命中已知但不支持的旧格式，或内容无法识别。
    """
    file_path = Path(file_path)
    file_name = file_path.name

    detected_by_extension = _detect_by_extension(file_path)
    if detected_by_extension is not None:
        return DetectionResult(detected_by_extension, "extension", file_name)

    # 扩展名没给出结论：可能是未知扩展名、无扩展名，或扩展名与内容不符
    return _detect_by_content_sniffing(file_path)


def _detect_by_extension(file_path: Path) -> DocumentType | None:
    extension = file_path.suffix.lower()
    if extension in EXTENSION_TO_TYPE:
        return EXTENSION_TO_TYPE[extension]
    if extension in _UNSUPPORTED_LEGACY_EXTENSIONS:
        raise UnsupportedFileError(_UNSUPPORTED_LEGACY_EXTENSIONS[extension])
    return None  # 未知扩展名 → 交给内容嗅探


def _detect_by_content_sniffing(file_path: Path) -> DetectionResult:
    with open(file_path, "rb") as file_handle:
        head_bytes = file_handle.read(4096)

    if head_bytes.startswith(_PDF_HEADER):
        return _result(DocumentType.PDF, "content_sniffing:pdf_header", file_path)
    if head_bytes.startswith(_ZIP_HEADER):
        return _result(_detect_office_zip(file_path), "content_sniffing:zip_structure", file_path)
    if head_bytes.startswith(_OLE_HEADER):
        raise UnsupportedFileError(
            "该文件是旧版 Office 复合文档（.doc/.xls/.ppt），请在 Office 中另存为新格式后上传"
        )

    decoded_text = _try_decode_as_text(head_bytes)
    if decoded_text is not None:
        document_type = _classify_decoded_text(decoded_text, file_path)
        return _result(document_type, "content_sniffing:text", file_path)

    raise UnsupportedFileError("无法识别文件类型（不是已知的文档/文本格式）")


def _result(document_type: DocumentType, detected_by: str, file_path: Path) -> DetectionResult:
    return DetectionResult(document_type, detected_by, file_path.name)


def _detect_office_zip(file_path: Path) -> DocumentType:
    """ZIP 容器内按目录特征判定是哪一种 Office 文档（docx/xlsx/pptx）。"""
    try:
        with zipfile.ZipFile(file_path) as archive:
            member_names = archive.namelist()
    except zipfile.BadZipFile:
        raise UnsupportedFileError("文件是损坏的 ZIP 容器，无法识别为 Office 文档")
    for directory_marker, document_type in _ZIP_DIRECTORY_MARKERS.items():
        if any(name.startswith(directory_marker) for name in member_names):
            return document_type
    raise UnsupportedFileError("是 ZIP 容器但不是受支持的 Office 文档（docx/xlsx/pptx）")


def _try_decode_as_text(head_bytes: bytes) -> str | None:
    """按编码顺序尝试解码文件头；可打印字符占比过低说明是二进制，判非文本。"""
    for encoding in _TEXT_ENCODINGS:
        try:
            decoded = head_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        if _printable_ratio(decoded) >= _PRINTABLE_RATIO_THRESHOLD:
            return decoded
    return None


def _printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable_count = sum(1 for character in text if character.isprintable() or character.isspace())
    return printable_count / len(text)


def _classify_decoded_text(decoded_text: str, file_path: Path) -> DocumentType:
    """文本内容进一步区分：JSON（结构数据）→ markdown（标题特征）→ 纯文本兜底。

    JSON 判定要读全文件：文件头 4KB 可能是截断的 JSON 片段，json.loads 必然失败。
    文件头特征像 JSON（{/[ 开头）才值得为它做全量解析（低频操作，开销可接受）。
    """
    stripped = decoded_text.lstrip("\ufeff \t\r\n")
    if stripped.startswith(("{", "[")):
        try:
            full_text = file_path.read_text(encoding="utf-8-sig", errors="replace")
            json.loads(full_text)  # 可完整解析才算 JSON，避免误伤散文以 { 开头
            return DocumentType.JSON
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass  # 只是以 { 开头的普通文本，落入下方判断
    if _HEADING_LINE_PATTERN.search(decoded_text):
        return DocumentType.MARKDOWN
    return DocumentType.PLAIN_TEXT
