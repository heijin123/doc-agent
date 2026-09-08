"""M3 问答层 确定性验证（无需 LLM Key；向量库注入临时空目录）。

覆盖：
1. 纯函数：幽灵引用剔除（越界 [9] 删除、有效 [1] 保留、句子不删、去重保序）
2. 问候不检索：greet 路径不碰 search_docs / 不调 LLM
3. 无 Key 降级：文档问题空库全链不崩，answer 给"未找到/未配置"可读答复
4. 端到端（mock 注入 hits + mock chat）：有效引用 → sources 组装契约
   （document_name/page_number/snippet 与前端对齐）；越界角标被剔除
5. server chat 端点契约：问候 POST → final_answer/sources 字段齐

运行：.venv/Scripts/python.exe scripts/verify_docqa_m3.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

# 环境前置（import docagent 前）：mock 向量 + 临时 chroma/图片目录
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="verify_docqa_"))
os.environ["DOCAGENT_CHROMA_DIR"] = str(_TMP_ROOT / "chroma")
os.environ["DOCAGENT_IMAGES_DIR"] = str(_TMP_ROOT / "images")
os.environ["DOCAGENT_IMAGES_DB"] = str(_TMP_ROOT / "image_files.db")
os.environ["EMBEDDING_PROVIDER"] = "mock"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import docagent.config as config  # noqa: E402
from docagent import qa as qa_module  # noqa: E402
from docagent.qa import run_question, strip_out_of_range_citations  # noqa: E402
from docagent.vectorstore import store as vector_store  # noqa: E402

passed_cases: list[str] = []
failed_cases: list[str] = []


def check(condition: bool, case_name: str, detail: str = "") -> None:
    if condition:
        passed_cases.append(case_name)
        print(f"  PASS  {case_name}")
    else:
        failed_cases.append(case_name)
        print(f"  FAIL  {case_name}  {detail}")


def _fake_hits() -> list[dict]:
    return [
        {
            "id": "sample:0",
            "text": "DocAgent 支持七种文档格式的摄取，包括 PDF、Word、Excel。",
            "metadata": {"doc_id": "sample", "title": "产品手册", "page": 3, "block_type": "page"},
            "distance": 0.5,
        },
        {
            "id": "sample:1",
            "text": "引用溯源会按句标注出处，答案带文档与页码。",
            "metadata": {"doc_id": "sample", "title": "产品手册", "page": 5, "block_type": "page"},
            "distance": 0.7,
        },
    ]


def verify_pure_citation_stripping() -> None:
    print("\n== 1. 幽灵引用剔除（纯函数）==")
    cleaned, referenced = strip_out_of_range_citations("根据[1]与[9]可知，该功能支持[1]引用。[99]越界。", valid_count=2)
    check(cleaned == "根据[1]与可知，该功能支持[1]引用。越界。",
          "越界 [9][99] 删除、有效 [1] 保留、句子不删", f"cleaned={cleaned!r}")
    check(referenced == [1], "有效引用去重保序", f"refs={referenced}")

    cleaned, referenced = strip_out_of_range_citations("完全越界[3]，无有效引用。", valid_count=2)
    check(cleaned == "完全越界，无有效引用。", "全越界时仅删角标")
    check(referenced == [], "全越界无有效引用")

    cleaned, referenced = strip_out_of_range_citations("正常句子，无角标。", valid_count=3)
    check(cleaned == "正常句子，无角标。" and referenced == [], "无角标原文保留")


def verify_greeting_does_not_search() -> None:
    print("\n== 2. 问候不检索（不碰 search_docs / 不调 LLM）==")
    original_key = config.DASHSCOPE_API_KEY
    config.DASHSCOPE_API_KEY = ""  # 即使无 Key 问候也应正常
    try:
        with mock.patch.object(vector_store, "search_docs", side_effect=AssertionError("问候不应检索")) as fake_search, \
             mock.patch.object(qa_module.llm, "chat_completion", side_effect=AssertionError("问候不应调 LLM")) as fake_llm:
            outcome = run_question("你好，你是谁？")
            fake_search.assert_not_called()
            fake_llm.assert_not_called()
    finally:
        config.DASHSCOPE_API_KEY = original_key
    check("doc-agent 文档助手" in outcome["answer"] and outcome["sources"] == [],
          "问候返回固定助手介绍", f"answer={outcome['answer'][:40]!r}")


def verify_no_key_degradation() -> None:
    print("\n== 3. 无 Key + 空库降级（全链不崩）==")
    original_key = config.DASHSCOPE_API_KEY
    config.DASHSCOPE_API_KEY = ""  # chat 不可用 → answer 节点降级文本
    try:
        outcome = run_question("振华重工 2024 年报的营业收入是多少？")
    finally:
        config.DASHSCOPE_API_KEY = original_key
    check(outcome["answer"] != "" and "未找到" in outcome["answer"],
          "空库/无 Key 给可读答复", f"answer={outcome['answer'][:50]!r}")
    check(outcome["degraded"] is True, "降级标记", f"degraded={outcome['degraded']}")


def verify_end_to_end_citations() -> None:
    print("\n== 4. 端到端（mock hits + mock chat）：引用组装与越界剔除 ==")
    original_key = config.DASHSCOPE_API_KEY
    config.DASHSCOPE_API_KEY = "test-key"  # llm 被 patch，不真调
    try:
        fake_chat = mock.Mock(return_value={
            "text": "支持七种格式[1]。详情见手册[2]与[9]。",
            "usage": {"prompt_tokens": 120, "completion_tokens": 30},
        })
        with mock.patch.object(vector_store, "search_docs", return_value=_fake_hits()), \
             mock.patch.object(qa_module.llm, "chat_completion", fake_chat):
            outcome = run_question("doc-agent 支持哪些格式？")
    finally:
        config.DASHSCOPE_API_KEY = original_key

    check("支持七种格式[1]。详情见手册[2]与。" in outcome["answer"],
          "越界 [9] 被剔除、[1][2] 保留", f"answer={outcome['answer']!r}")
    check(len(outcome["sources"]) == 2, "sources 只含有效引用 2 条", f"n={len(outcome['sources'])}")
    first_source = outcome["sources"][0]
    contract_ok = (
        first_source.get("document_name") == "产品手册"
        and first_source.get("page_number") == 3
        and "PDF" in (first_source.get("snippet") or "")
    )
    check(contract_ok, "sources 契约字段与前端对齐", f"{first_source}")


def verify_server_chat_contract() -> None:
    print("\n== 5. server chat 端点契约（问候路径，无副作用）==")
    from fastapi.testclient import TestClient  # noqa: E402
    from docagent.api.server import app  # noqa: E402

    client = TestClient(app)
    health = client.get("/health")
    check(health.status_code == 200 and health.json().get("status") == "ok", "health ok")

    response = client.post("/api/v1/chat", json={"message": "你好"})
    body = response.json()
    check(response.status_code == 200, "chat HTTP 200")
    check(body.get("final_answer") and body.get("sources") == [] and body.get("thread_id"),
          "chat 响应字段契约（final_answer/sources/thread_id）", f"keys={sorted(body.keys())}")


if __name__ == "__main__":
    verify_pure_citation_stripping()
    verify_greeting_does_not_search()
    verify_no_key_degradation()
    verify_end_to_end_citations()
    verify_server_chat_contract()

    shutil.rmtree(_TMP_ROOT, ignore_errors=True)
    print(f"\n结果：{len(passed_cases)} 通过 / {len(failed_cases)} 失败")
    if failed_cases:
        print("失败项：", failed_cases)
        sys.exit(1)
    print("ALL_PASS")
