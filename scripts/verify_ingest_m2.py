"""M2 VLM 转录线 确定性验证（mock VLM 注入；无真实 LLM 调用）。

覆盖（2026-09-08 VLM 设计 v2——红页转录语义）：
1. 无 Key 降级：红页不转录（原文入库，M1 行为），报告标注，不崩
2. 显式重建：delete_doc_blocks 清零（孤儿清理入口，业务层显式调用）
3. 转录替换（v2 核心）：注入转录后红页原文**不产块**（文本不可信不进库），
   转录文本作为 figure_transcript 块追加尾部（老块 id 不动 → 无平移无孤儿）
4. 幂等：转录输出稳定时重跑全 skip

运行：.venv/Scripts/python.exe scripts/verify_ingest_m2.py
依赖真实语料：data/samples/2_react_paper_en.pdf（P2 等 5 个坏字体红页）
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

# 环境前置（import docagent 前）：mock 向量 + 临时 chroma/图片目录
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="verify_ingest_m2_"))
os.environ["DOCAGENT_CHROMA_DIR"] = str(_TMP_ROOT / "chroma")
os.environ["DOCAGENT_IMAGES_DIR"] = str(_TMP_ROOT / "images")
os.environ["DOCAGENT_IMAGES_DB"] = str(_TMP_ROOT / "image_files.db")
os.environ["EMBEDDING_PROVIDER"] = "mock"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import docagent.config as config  # noqa: E402
from docagent.ingest import vlm_transcribe  # noqa: E402
from docagent.ingest.pdf_quality import assess_pdf  # noqa: E402
from docagent.ingest.pipeline import run_document  # noqa: E402
from docagent.vectorstore import store as vector_store  # noqa: E402

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"
REACT_PDF = SAMPLES_DIR / "2_react_paper_en.pdf"  # 含 5 个坏字体红页

passed_cases: list[str] = []
failed_cases: list[str] = []


def check(condition: bool, case_name: str, detail: str = "") -> None:
    if condition:
        passed_cases.append(case_name)
        print(f"  PASS  {case_name}")
    else:
        failed_cases.append(case_name)
        print(f"  FAIL  {case_name}  {detail}")


def _collection() -> object:
    return vector_store.get_collection()


def _doc_count(doc_id: str) -> int:
    return len(_collection().get(where={"doc_id": doc_id})["ids"])


def _page_blocks(doc_id: str, page_number: int) -> list[dict]:
    found = _collection().get(
        where={"$and": [{"doc_id": doc_id}, {"page": page_number}]},
        include=["documents", "metadatas"],
    )
    return [
        {"text": found["documents"][index], "metadata": found["metadatas"][index]}
        for index in range(len(found["ids"]))
    ]


def _fake_transcripts_factory(text_template: str = "VLM-MOCK-P{page} Apple Remote Front Row keyboard"):
    def fake_transcribe(pdf_path, page_numbers, max_pages=None):
        limited = page_numbers[:max_pages] if max_pages else page_numbers
        transcripts = {page_number: text_template.format(page=page_number) for page_number in limited}
        usage = {"prompt_tokens": 1000 * len(transcripts), "completion_tokens": 200 * len(transcripts)}
        return transcripts, usage, []
    return fake_transcribe


def verify_no_key_degradation(red_pages: list[int]) -> None:
    print("\n== 1. 无 Key 降级（红页原文入库 = M1 行为，报告标注，不崩）==")
    original_key = config.DASHSCOPE_API_KEY
    config.DASHSCOPE_API_KEY = ""
    try:
        with mock.patch.object(vlm_transcribe, "transcribe_pages") as fake_transcribe:
            report = run_document(REACT_PDF)
            fake_transcribe.assert_not_called()  # 无 Key 根本不该发起转录
    finally:
        config.DASHSCOPE_API_KEY = original_key

    check(report["status"] != "failed", "无 Key 文档仍成功入库", f"status={report['status']}")
    check(report["vlm_pages"] == 0, "无 Key 转录页数 0")
    check("未配" in (report.get("vlm_note") or ""), "报告标注未配 Key", f"note={report.get('vlm_note')}")

    red_blocks = sum(len(_page_blocks(REACT_PDF.stem, page_number)) for page_number in red_pages[:3])
    check(red_blocks >= 1, "无 Key：红页原文块在库（M1 兼容）", f"前3红页块数={red_blocks}")


def verify_rebuild_and_transcribe(red_pages: list[int]) -> None:
    print("\n== 2/3. 显式重建 + 转录替换（红页原文不产块，转录尾部追加，无孤儿）==")
    doc_id = REACT_PDF.stem
    original_count = _doc_count(doc_id)  # 无 Key 阶段的全量库

    deleted = vector_store.delete_doc_blocks(doc_id)
    check(deleted == original_count and _doc_count(doc_id) == 0,
          "delete_doc_blocks 显式清零（孤儿清理入口）", f"删 {deleted} → 余 {_doc_count(doc_id)}")

    config.DASHSCOPE_API_KEY = "test-key"  # 注入转录前提（transcribe 被 mock，不真调）
    config.EMBEDDING_PROVIDER = "dashscope"  # 绕过 vlm_node 的 mock 环境抑制（embedding 缓存仍 mock）
    with mock.patch.object(vlm_transcribe, "transcribe_pages", side_effect=_fake_transcripts_factory()):
        report = run_document(REACT_PDF)

    check(report["status"] == "ok" and report["stored"] > 0, "注入转录后文档入库", f"stored={report['stored']}")
    check(report["vlm_pages"] == len(red_pages), "转录页数 = 红页数", f"{report['vlm_pages']} vs {len(red_pages)}")
    check(report["vlm_usage"]["prompt_tokens"] > 0, "usage 有 token 成本", f"{report['vlm_usage']}")

    # 每个红页：库中该页只剩转录块（无乱码原文块）
    for page_number in red_pages[:3]:
        page_blocks = _page_blocks(doc_id, page_number)
        is_transcript_only = (
            len(page_blocks) == 1
            and page_blocks[0]["metadata"].get("block_type") == "figure_transcript"
            and "VLM-MOCK" in (page_blocks[0]["text"] or "")
        )
        check(is_transcript_only, f"红页 P{page_number} 仅转录块（原文不进库）",
              f"块数={len(page_blocks)}")

    # 非红页块数不变：重建后总数 = 转录块数 + 非红页原文块数
    transcribe_result_chunks = report["chunks"]
    non_red_page_count = transcribe_result_chunks - len(red_pages)  # 转录块在尾部占红页数个
    check(_doc_count(doc_id) == transcribe_result_chunks, "库内总块 = 本次切块数（无孤儿残留）",
          f"库={_doc_count(doc_id)} vs 切块={transcribe_result_chunks}")
    check(non_red_page_count > 0, "非红页块照常入库", f"非红页块={non_red_page_count}")


def verify_idempotent_rerun(red_pages: list[int]) -> None:
    print("\n== 4. 转录稳定时幂等重跑全 skip ==")
    with mock.patch.object(vlm_transcribe, "transcribe_pages", side_effect=_fake_transcripts_factory()):
        rerun_report = run_document(REACT_PDF)
    check(rerun_report["status"] == "skipped" and rerun_report["stored"] == 0 and rerun_report["skipped"] > 0,
          "转录稳定重跑全 skip（幂等）", f"stored={rerun_report['stored']} skipped={rerun_report['skipped']}")


if __name__ == "__main__":
    red_pages = assess_pdf(REACT_PDF)["red_pages"]
    check(len(red_pages) >= 1, "前置：react 判出红页（活体料成立）", f"red_pages={red_pages}")

    verify_no_key_degradation(red_pages)
    verify_rebuild_and_transcribe(red_pages)
    verify_idempotent_rerun(red_pages)

    shutil.rmtree(_TMP_ROOT, ignore_errors=True)
    print(f"\n结果：{len(passed_cases)} 通过 / {len(failed_cases)} 失败")
    if failed_cases:
        print("失败项：", failed_cases)
        sys.exit(1)
    print("ALL_PASS")
