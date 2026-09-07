"""doc-agent 探测 + 分格式切片 确定性验证（无需 Redis / LLM Key）。

覆盖两条主路径：
1. 探测：扩展名命中 / 内容嗅探（改名 PDF/JSON/文本）/ 旧版 Office 拒绝 / 未知二进制拒绝
2. 切块：7 种格式样本 → 溯源键齐全 + block_seq 连续 + 单块不超 MAX
   + 各格式判别词命中（验证"定制策略"真的生效，而非统一套模板）

运行：.venv/Scripts/python.exe scripts/verify_chunking.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

# 脚本在 scripts/ 下运行，docagent 包在项目根
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docagent.ingest.chunk_models import MAX_CHUNK_CHARS, TRACEABILITY_KEYS, DocumentChunk
from docagent.ingest.chunking import chunk_document
from docagent.ingest.file_type_detection import (
    DocumentType,
    UnsupportedFileError,
    detect_file_type,
)

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"
GENERATED_DIR = SAMPLES_DIR / "generated"

passed_cases: list[str] = []
failed_cases: list[str] = []


def check(condition: bool, case_name: str, detail: str = "") -> None:
    if condition:
        passed_cases.append(case_name)
        print(f"  PASS  {case_name}")
    else:
        failed_cases.append(case_name)
        print(f"  FAIL  {case_name}  {detail}")


# ==================== 一、探测路径 ====================
def verify_detection() -> None:
    print("\n== 探测 ==")
    md_result = detect_file_type(GENERATED_DIR / "sample_manual.md")
    check(md_result.document_type == DocumentType.MARKDOWN and md_result.detected_by == "extension",
          "探测-扩展名判 md")

    pdf_result = detect_file_type(SAMPLES_DIR / "3_invoice_table_cn.pdf")
    check(pdf_result.document_type == DocumentType.PDF, "探测-扩展名判 pdf")

    # 内容嗅探：PDF 去掉扩展名仍应识别（文件头 %PDF）
    renamed_pdf = GENERATED_DIR / "renamed_pdf_copy"
    shutil.copyfile(SAMPLES_DIR / "3_invoice_table_cn.pdf", renamed_pdf)
    sniffed = detect_file_type(renamed_pdf)
    check(sniffed.document_type == DocumentType.PDF and sniffed.detected_by.startswith("content_sniffing"),
          "嗅探-无扩展名 PDF", f"got {sniffed.detected_by}")

    # 内容嗅探：JSON 改成未知扩展名仍应识别
    renamed_json = GENERATED_DIR / "products.data"
    shutil.copyfile(GENERATED_DIR / "sample_products.json", renamed_json)
    sniffed_json = detect_file_type(renamed_json)
    check(sniffed_json.document_type == DocumentType.JSON, "嗅探-未知扩展名 JSON", f"got {sniffed_json.document_type}")

    # 内容嗅探：纯文本改名未知扩展名 → 纯文本兜底
    renamed_txt = GENERATED_DIR / "notes.unknown"
    shutil.copyfile(GENERATED_DIR / "sample_notes.txt", renamed_txt)
    sniffed_txt = detect_file_type(renamed_txt)
    check(sniffed_txt.document_type == DocumentType.PLAIN_TEXT, "嗅探-文本兜底", f"got {sniffed_txt.document_type}")

    # 拒绝路径：旧版 Office 扩展名
    try:
        detect_file_type(GENERATED_DIR / "nonexistent.doc")  # 文件不存在也会先查扩展名表
        check(False, "拒绝-旧版 .doc 扩展名")
    except UnsupportedFileError:
        check(True, "拒绝-旧版 .doc 扩展名")

    # 拒绝路径：OLE 二进制头（老 Office 内容但无扩展名）
    fake_ole = GENERATED_DIR / "legacy_binary"
    fake_ole.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64)
    try:
        detect_file_type(fake_ole)
        check(False, "拒绝-OLE 二进制头")
    except UnsupportedFileError:
        check(True, "拒绝-OLE 二进制头")

    # 拒绝路径：未知二进制
    fake_binary = GENERATED_DIR / "random_binary.bin"
    fake_binary.write_bytes(bytes(range(256)) * 4)
    try:
        detect_file_type(fake_binary)
        check(False, "拒绝-未知二进制")
    except UnsupportedFileError:
        check(True, "拒绝-未知二进制")

    for temporary_file in (renamed_pdf, renamed_json, renamed_txt, fake_ole, fake_binary):
        temporary_file.unlink(missing_ok=True)


# ==================== 二、切块路径 ====================
def verify_chunking() -> None:
    print("\n== 切块 ==")
    cases = [
        ("markdown", GENERATED_DIR / "sample_manual.md"),
        ("txt", GENERATED_DIR / "sample_notes.txt"),
        ("json", GENERATED_DIR / "sample_products.json"),
        ("excel", GENERATED_DIR / "xlsx_products.xlsx"),
        ("word", GENERATED_DIR / "docx_spec.docx"),
        ("slides", GENERATED_DIR / "pptx_intro.pptx"),
        ("pdf", SAMPLES_DIR / "3_invoice_table_cn.pdf"),
    ]
    for format_name, sample_path in cases:
        verify_one_document(format_name, sample_path)


def verify_one_document(format_name: str, sample_path: Path) -> None:
    result = chunk_document(sample_path)
    check(result.document_type == format_name, f"[{format_name}] 类型标注", f"got {result.document_type}")
    check(len(result.chunks) > 0, f"[{format_name}] 切出块", f"chunks={len(result.chunks)}")

    all_text = ""
    for block_index, chunk in enumerate(result.chunks):
        all_text += chunk.text
        check(_chunk_well_formed(chunk, block_index), f"[{format_name}] 块#{block_index} 溯源键与序号",
              f"keys={sorted(chunk.metadata.keys())}")

    expected_keywords = {
        "markdown": ["文档智能助手", "pip install docagent", "DASHSCOPE_API_KEY"],
        "txt": ["引用溯源", "七种格式"],
        "json": ["蓝牙耳机", "199"],
        "excel": ["蓝牙耳机", "价格：199"],
        "word": ["DA-500", "399 元", "离线问答"],
        "slides": ["引用溯源问答"],
        "pdf": ["发票"],
    }[format_name]
    for keyword in expected_keywords:
        check(keyword in all_text, f"[{format_name}] 判别词 {keyword!r} 未丢", f"text={all_text[:80]}...")

    check(_all_chunks_within_size_limit(result.chunks), f"[{format_name}] 所有块 ≤ MAX_CHUNK_CHARS")
    _verify_format_specific(format_name, result.chunks)


def _chunk_well_formed(chunk: DocumentChunk, expected_seq: int) -> bool:
    text_ok = bool(chunk.text.strip())
    metadata = chunk.metadata
    keys_ok = set(metadata.keys()) <= set(TRACEABILITY_KEYS)
    required_keys = {"doc_id", "title", "block_type", "block_seq", "source_file", "format"}
    required_ok = required_keys <= set(metadata.keys())
    seq_ok = metadata.get("block_seq") == expected_seq
    size_ok = len(chunk.text) <= MAX_CHUNK_CHARS
    return text_ok and keys_ok and required_ok and seq_ok and size_ok


def _all_chunks_within_size_limit(chunks: list[DocumentChunk]) -> bool:
    return all(len(chunk.text) <= MAX_CHUNK_CHARS for chunk in chunks)


def _verify_format_specific(format_name: str, chunks: list[DocumentChunk]) -> None:
    """各格式"定制策略真的生效"的专项断言。"""
    if format_name == "markdown":
        sections = {chunk.metadata.get("section") for chunk in chunks}
        check("产品简介" in str(sections), "[markdown] 一级标题进 section")
        check("安装说明 / 环境变量" in str(sections), "[markdown] 标题链 section（二级/三级拼接）")
        no_code_heading = all(
            (chunk.metadata.get("section") or "").find("这行在代码块里") == -1 for chunk in chunks
        )
        check(no_code_heading, "[markdown] 代码块内 # 不误判为标题")
        paragraph_chunks = [chunk for chunk in chunks if chunk.metadata.get("block_type") == "paragraph"]
        check(len(paragraph_chunks) >= 2, "[markdown] 长段落触发二次切", f"paragraph_chunks={len(paragraph_chunks)}")

    elif format_name == "json":
        json_paths = sorted(chunk.metadata.get("section") for chunk in chunks)
        check(json_paths == ["$[0]", "$[1]"], "[json] 对象数组逐元素成块", f"paths={json_paths}")

    elif format_name == "excel":
        row_pages = sorted(chunk.metadata.get("page") for chunk in chunks)
        # 原 4 行含 1 空行：记录行号应为 1/2/4（空行被跳过，保留原行号）
        check(row_pages == [1, 2, 4], "[excel] 一行一记录 + 空行跳过 + 原行号保留", f"pages={row_pages}")
        all_sections = {chunk.metadata.get("section") for chunk in chunks}
        check(all_sections == {"工作表 商品清单"}, "[excel] sheet 名进 section")

    elif format_name == "word":
        table_block = next((chunk for chunk in chunks if "DA-500" in chunk.text), None)
        check(table_block is not None and "型号" in table_block.text,
              "[word] 表格转 markdown 保留表头与数据", f"text={table_block.text[:60] if table_block else '无'}")

    elif format_name == "slides":
        slide_sections = {chunk.metadata.get("section") for chunk in chunks}
        check("DocAgent 介绍" in slide_sections and "核心能力" in slide_sections,
              "[slides] 每页标题进 section", f"sections={slide_sections}")
        slide_pages = {chunk.metadata.get("page") for chunk in chunks}
        check(slide_pages == {1, 2}, "[slides] 每页一块（页号标注）", f"pages={slide_pages}")

    elif format_name == "pdf":
        check(any(chunk.metadata.get("page") == 1 for chunk in chunks), "[pdf] 页码溯源存在")


if __name__ == "__main__":
    verify_detection()
    verify_chunking()

    print(f"\n结果：{len(passed_cases)} 通过 / {len(failed_cases)} 失败")
    if failed_cases:
        print("失败项：", failed_cases)
        sys.exit(1)
    print("ALL_PASS")
