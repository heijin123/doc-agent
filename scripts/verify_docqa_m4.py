"""M4 评估引擎 确定性验证（无需 LLM Key；注入临时 chroma + mock embedding）。

覆盖：
1. golden 结构校验：缺字段 / id 重复 → 报错清单；合法用例通过
2. anchor 定位（纯函数）：空白/大小写不敏感命中正确块；无命中给提示；
   anchor 跨文档命中触发跨 doc 提示（anchor 太泛预警）
3. 命中判定与指标聚合（纯函数）：rank / hit / miss；手工小样本 recall 与 MRR
4. 门槛判定：mock → SKIP；真向量 recall 达标/不达标 → PASS/FAIL；
   答案层可回查率与带引用用例数的分支
5. run_eval 全流程（mock + 临时库 + 临时 golden）：不崩、报告结构齐、
   anchor 缺陷用例被跳过、报告落盘到注入路径
6. run_answer_eval 无 Key 降级：不调 chat、degraded 标记、不崩

运行：.venv/Scripts/python.exe scripts/verify_docqa_m4.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

# 环境前置（import docagent 前）：mock 向量 + 临时 chroma/images/报告目录
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="verify_docqa_m4_"))
os.environ["DOCAGENT_CHROMA_DIR"] = str(_TMP_ROOT / "chroma")
os.environ["DOCAGENT_IMAGES_DIR"] = str(_TMP_ROOT / "images")
os.environ["DOCAGENT_IMAGES_DB"] = str(_TMP_ROOT / "image_files.db")
os.environ["EMBEDDING_PROVIDER"] = "mock"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docagent import eval as eval_module  # noqa: E402
from docagent.eval import (  # noqa: E402
    aggregate_metrics,
    compute_case_outcome,
    eval_gate,
    load_golden,
    locate_expected_blocks,
    run_answer_eval,
    run_eval,
    run_retrieval_eval,
    validate_golden,
)
from docagent.ingest.chunk_models import DocumentChunk  # noqa: E402
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


# ---------------- 测试数据 ----------------

def _seed_library() -> None:
    """向临时向量库注入 2 文档 4 块（mock embedding，仅供定位/链路测试，无语义）。"""
    doc_a_blocks = [
        DocumentChunk("Hermes Agent 是基于 Nous Research 开源框架的实战指南。", {
            "doc_id": "doc_a", "title": "Hermes 手册", "page": 1,
            "section": "封面", "block_type": "page", "block_seq": 0,
        }),
        DocumentChunk("Hermes 把外部能力做成「一文件一后端」的插件交付。", {
            "doc_id": "doc_a", "title": "Hermes 手册", "page": 41,
            "section": "架构", "block_type": "page", "block_seq": 1,
        }),
    ]
    doc_b_blocks = [
        DocumentChunk("乐鑫科技回购方案实施期限是 2026 年 8 月 18 日至 2026 年 11 月 17 日。", {
            "doc_id": "doc_b", "title": "乐鑫回购公告", "page": 1,
            "section": "摘要", "block_type": "page", "block_seq": 0,
        }),
        DocumentChunk("截至 2026 年 9 月 4 日累计回购 1,420,836 股。Hermes 与其无关。", {
            "doc_id": "doc_b", "title": "乐鑫回购公告", "page": 2,
            "section": "进展", "block_type": "page", "block_seq": 1,
        }),
    ]
    vector_store.upsert_doc_chunks(doc_a_blocks, "doc_a")
    vector_store.upsert_doc_chunks(doc_b_blocks, "doc_b")


def _write_golden(cases: list[dict]) -> Path:
    golden_path = _TMP_ROOT / "golden.json"
    golden_path.write_text(json.dumps({"meta": {}, "cases": cases}), encoding="utf-8")
    return golden_path


_VALID_CASES = [
    {"id": "g01", "query": "Hermes 手册基于什么开源框架？", "expect_doc": "doc_a", "anchor": "Nous Research"},
    {"id": "g02", "query": "乐鑫回购了多少股？", "expect_doc": "doc_b", "anchor": "1,420,836"},
]


# ---------------- 用例 ----------------

def verify_golden_validation() -> None:
    print("\n== 1. golden 结构校验 ==")
    errors = validate_golden(_VALID_CASES)
    check(errors == [], "合法用例零错误", f"errors={errors}")

    bad = [
        {"id": "x1", "query": "没有 anchor 字段", "expect_doc": "doc_a"},
        {"id": "x2", "query": "短", "expect_doc": "doc_a", "anchor": "ab"},
        {"id": "x1", "query": "重复 id 用例", "expect_doc": "doc_b", "anchor": "足够长的锚文本内容"},
    ]
    errors = validate_golden(bad)
    check(len(errors) == 3, "缺字段/过短/重复 id 均被报出", f"errors={errors}")


def verify_anchor_locating() -> None:
    print("\n== 2. anchor 定位（空白/大小写不敏感 + 缺陷提示）==")
    expected, warnings = locate_expected_blocks("nous research 开源", "doc_a")
    check(expected == ["doc_a:0"], "空白不敏感命中正确块", f"expected={expected}")
    check(warnings == [], "同文档命中无跨 doc 警告")

    expected, _ = locate_expected_blocks("1,420,836", "doc_b")
    check(expected == ["doc_b:1"], "数字 anchor 精确命中", f"expected={expected}")

    expected, warnings = locate_expected_blocks("hermes", "doc_a")
    check("doc_b:1" in expected and warnings, "跨文档命中触发太泛警告", f"warn={warnings}")

    expected, warnings = locate_expected_blocks("库里不存在的锚文本xyz", "doc_a")
    check(expected == [] and len(warnings) == 1, "无命中给缺陷提示", f"warn={warnings}")


def verify_case_outcome_and_metrics() -> None:
    print("\n== 3. 命中判定 + 指标聚合（纯函数）==")
    hit = compute_case_outcome(["doc_b:1", "doc_a:0"], ["doc_a:0"])
    check(hit == {"hit": True, "rank": 2}, "期望块在 rank2 命中", f"{hit}")

    miss = compute_case_outcome(["doc_b:0", "doc_b:1"], ["doc_a:1"])
    check(miss == {"hit": False, "rank": None}, "未命中返回 miss", f"{miss}")

    metrics = aggregate_metrics([
        {"hit": True, "rank": 1},
        {"hit": True, "rank": 3},
        {"hit": False, "rank": None},
    ])
    # recall = 2/3 ≈ 0.6667；MRR = (1/1 + 1/3)/2 = 0.6667
    check(metrics["recall"] == round(2 / 3, 4) and metrics["mrr"] == round(2 / 3, 4),
          "小样本 recall/MRR 手算一致", f"{metrics}")

    empty = aggregate_metrics([])
    check(empty["recall"] == 0.0 and empty["cases"] == 0, "空结果安全返回 0")

    merged = eval_module._merge_usage({}, {"prompt_tokens": 10, "completion_tokens": 5})
    check(merged == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
          "usage 聚合自动补 total_tokens", f"{merged}")
    merged_twice = eval_module._merge_usage(dict(merged), {"prompt_tokens": 2})
    check(merged_twice["total_tokens"] == 17 and merged_twice["completion_tokens"] == 5,
          "usage 二次累加正确", f"{merged_twice}")


def verify_gate_logic() -> None:
    print("\n== 4. 门槛判定分支 ==")
    skip = eval_gate({"recall": 0.5, "top_k": 5, "cases": 10}, "mock")
    check(skip["passed"] is None and "跳过" in skip["checks"][0], "mock 下 SKIP", f"{skip['checks']}")

    ok = eval_gate({"recall": 0.9, "top_k": 5, "cases": 10}, "dashscope")
    check(ok["passed"] is True, "真向量 recall≥0.8 PASS")

    bad = eval_gate({"recall": 0.5, "top_k": 5, "cases": 10}, "dashscope")
    check(bad["passed"] is False, "真向量 recall<0.8 FAIL")

    answer_ok = eval_gate({"recall": 0.9, "top_k": 5, "cases": 10}, "dashscope",
                          {"degraded": False, "traceable_rate": 1.0, "referenced_cases": 3})
    check(answer_ok["passed"] is True, "答案层可回查率 100% 过闸")

    answer_bad = eval_gate({"recall": 0.9, "top_k": 5, "cases": 10}, "dashscope",
                           {"degraded": False, "traceable_rate": 0.8, "referenced_cases": 3})
    check(answer_bad["passed"] is False, "答案层可回查率 <1 FAIL")

    answer_none = eval_gate({"recall": 0.9, "top_k": 5, "cases": 10}, "dashscope",
                            {"degraded": False, "traceable_rate": None, "referenced_cases": 0})
    check(answer_none["passed"] is False and "带引用用例" in answer_none["checks"][1],
          "0 个带引用用例触发 FAIL 检查", f"{answer_none['checks']}")


def verify_run_eval_end_to_end() -> None:
    print("\n== 5. run_eval 全流程（mock + 临时库）==")
    # 含 1 条 anchor 定位失败的缺陷用例（应被跳过，不计分母）
    defect_case = {"id": "g99", "query": "查不到的内容提问", "expect_doc": "doc_a", "anchor": "不存在锚文本xyz"}
    cases = _VALID_CASES + [defect_case]
    golden_path = _write_golden(cases)
    report_path = _TMP_ROOT / "reports" / "eval_report.json"

    report = run_eval(golden_path=golden_path, report_path=report_path)

    check(report["meta"]["cases_total"] == 3, "报告含全部用例计数")
    check(report["meta"]["provider"] == "mock", "provider=mock 标注")
    check(report["retrieval"]["metrics"]["cases"] == 2, "缺陷用例不计指标分母",
          f"cases={report['retrieval']['metrics']['cases']}")
    check(len(report["retrieval"]["defects"]) >= 1, "缺陷提示入报告")
    check(report["gate"]["passed"] is None, "mock 门槛 SKIP")
    check(report_path.exists(), "报告落盘到注入路径")
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    check(saved["gate"]["passed"] is None and saved["retrieval"]["metrics"]["cases"] == 2,
          "落盘 JSON 与内存报告一致")


def verify_answer_eval_degraded() -> None:
    print("\n== 6. run_answer_eval 无 Key 降级（chat 返回 None → degraded，不崩）==")
    original_key = os.environ.get("DASHSCOPE_API_KEY", "")
    import docagent.config as config  # noqa: E402

    config.DASHSCOPE_API_KEY = ""  # llm.chat_completion 动态读 key，无 Key 返 None（不真发请求）
    try:
        summary = run_answer_eval(_VALID_CASES)
    finally:
        config.DASHSCOPE_API_KEY = original_key

    check(summary["degraded"] is True, "降级标记")
    check(summary["cases_total"] == 2, "仍逐条产出结果")
    check(summary["referenced_cases"] == 0, "无 Key 无引用用例（不参与可回查率）")


if __name__ == "__main__":
    _seed_library()
    verify_golden_validation()
    verify_anchor_locating()
    verify_case_outcome_and_metrics()
    verify_gate_logic()
    verify_run_eval_end_to_end()
    verify_answer_eval_degraded()

    shutil.rmtree(_TMP_ROOT, ignore_errors=True)
    print(f"\n结果：{len(passed_cases)} 通过 / {len(failed_cases)} 失败")
    if failed_cases:
        print("失败项：", failed_cases)
        sys.exit(1)
    print("ALL_PASS")
