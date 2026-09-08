"""图片引用链路 确定性验证（无 LLM Key；图片目录/db 注入临时路径）。

覆盖（2026-09-08 图片引用设计）：
1. 切块提取：docx/pptx/pdf 含图样本 → 每格式提取到图（bytes 可解码）
   + 块文本出现 [IMAGE:{id}] 占位符 + metadata.image_ids 落块（逗号串）
2. 溯源白名单：加 image_ids 后块 metadata 仍 ⊆ TRACEABILITY_KEYS
3. id 语义：同内容 md5 尾一致、doc 前缀区分跨文档同图
4. 仓库幂等：save_images 两遍 → 第二遍全 skipped；文件与 sqlite 各只一份

运行：.venv/Scripts/python.exe scripts/verify_images_ref.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

# 注入临时图片目录/db 必须在 import config 之前（config 模块级读 env）
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="verify_images_ref_"))
os.environ["DOCAGENT_IMAGES_DIR"] = str(_TMP_ROOT / "images")
os.environ["DOCAGENT_IMAGES_DB"] = str(_TMP_ROOT / "image_files.db")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docagent.images import repo as image_repo
from docagent.ingest.chunk_models import (
    IMAGE_PLACEHOLDER_PATTERN,
    TRACEABILITY_KEYS,
    ExtractedImage,
)
from docagent.ingest.chunking import chunk_document

GENERATED_DIR = Path(__file__).resolve().parents[1] / "data" / "samples" / "generated"

passed_cases: list[str] = []
failed_cases: list[str] = []


def check(condition: bool, case_name: str, detail: str = "") -> None:
    if condition:
        passed_cases.append(case_name)
        print(f"  PASS  {case_name}")
    else:
        failed_cases.append(case_name)
        print(f"  FAIL  {case_name}  {detail}")


def _extract_image_ids_from_chunk(chunk) -> list[str]:
    raw = chunk.metadata.get("image_ids") or ""
    return [image_id for image_id in raw.split(",") if image_id]


def verify_image_extraction() -> None:
    print("\n== 图片提取（docx/pptx/pdf 含图样本）==")
    expected_formats = {
        "docx_spec.docx": "word",
        "pptx_intro.pptx": "slides",
        "pdf_with_image.pdf": "pdf",
    }
    all_image_ids_by_document: dict[str, list[str]] = {}

    for file_name, expected_format in expected_formats.items():
        result = chunk_document(GENERATED_DIR / file_name)
        check(result.document_type == expected_format, f"[{file_name}] 类型标注")

        extracted_count = len(result.images)
        check(extracted_count >= 1, f"[{file_name}] 提取到图", f"images={extracted_count}")
        if result.images:
            first_image = result.images[0]
            check(first_image.data[:4] == b"\x89PNG", f"[{file_name}] 图片字节可解码(png头)",
                  f"head={first_image.data[:4]!r}")
            check(first_image.ext == "png", f"[{file_name}] 扩展名推断")

        placeholder_total = sum(len(IMAGE_PLACEHOLDER_PATTERN.findall(chunk.text)) for chunk in result.chunks)
        check(placeholder_total >= 1, f"[{file_name}] 块文本含 [IMAGE] 占位符", f"占位={placeholder_total}")

        chunks_with_image_ids = [_extract_image_ids_from_chunk(chunk) for chunk in result.chunks]
        metadata_ids = sorted({image_id for ids in chunks_with_image_ids for image_id in ids})
        check(len(metadata_ids) == extracted_count,
              f"[{file_name}] metadata.image_ids 与提取图一一对应",
              f"metadata={metadata_ids} vs images={extracted_count}")
        all_image_ids_by_document[file_name] = metadata_ids

        for chunk in result.chunks:
            keys_ok = set(chunk.metadata.keys()) <= set(TRACEABILITY_KEYS)
            if not keys_ok:
                check(False, f"[{file_name}] 溯源键白名单", f"keys={sorted(chunk.metadata.keys())}")
                break
        else:
            check(True, f"[{file_name}] 溯源键白名单（含 image_ids 合法）")

    # 同图跨文档：md5 尾一致 + doc_id 前缀区分
    docx_id = all_image_ids_by_document["docx_spec.docx"][0]
    pptx_id = all_image_ids_by_document["pptx_intro.pptx"][0]
    pdf_id = all_image_ids_by_document["pdf_with_image.pdf"][0]
    suffixes = {image_id.rsplit("-", 1)[1] for image_id in (docx_id, pptx_id, pdf_id)}
    check(len(suffixes) == 1, "同内容图 md5 尾一致", f"suffixes={suffixes}")
    check(docx_id.startswith("docx_spec-") and pptx_id.startswith("pptx_intro-") and pdf_id.startswith("pdf_with_image-"),
          "doc 前缀区分跨文档同图", f"ids={docx_id}/{pptx_id}/{pdf_id}")


def verify_repository_idempotency() -> None:
    print("\n== 图片仓库幂等 ==")
    result = chunk_document(GENERATED_DIR / "docx_spec.docx")
    extracted_images: list[ExtractedImage] = result.images

    first_outcome = image_repo.save_images("docx_spec", extracted_images)
    check(first_outcome["saved"] == len(extracted_images) and first_outcome["skipped"] == 0,
          "首次落盘全 saved", f"{first_outcome}")

    second_outcome = image_repo.save_images("docx_spec", extracted_images)
    check(second_outcome["saved"] == 0 and second_outcome["skipped"] == len(extracted_images),
          "重复落盘全 skipped", f"{second_outcome}")

    registered = image_repo.count_images()
    check(registered == len(extracted_images), "sqlite 只登记一份", f"count={registered}")

    image_id = extracted_images[0].image_id
    record = image_repo.lookup(image_id)
    check(record is not None and record["doc_id"] == "docx_spec", "lookup 命中登记", f"{record}")
    resolved = image_repo.resolve_file_path(image_id)
    check(resolved is not None and resolved.exists() and resolved.stat().st_size == len(extracted_images[0].data),
          "文件落盘且可解析", f"{resolved}")


if __name__ == "__main__":
    verify_image_extraction()
    verify_repository_idempotency()

    shutil.rmtree(_TMP_ROOT, ignore_errors=True)
    print(f"\n结果：{len(passed_cases)} 通过 / {len(failed_cases)} 失败")
    if failed_cases:
        print("失败项：", failed_cases)
        sys.exit(1)
    print("ALL_PASS")
