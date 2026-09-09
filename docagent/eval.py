"""M4 问答评估引擎：golden 检索层指标 + 可选答案层引用可回查。

对应需求 §5.3 / R4（回归门槛 recall@5 ≥ 0.8，引用可回查率 100%）：
- golden 数据：data/golden/qa_golden.json（query → expect_doc + anchor 原文片段）
- 期望块定位：anchor 在向量库全量文本中做归一化子串匹配（空白/大小写不敏感），
  命中块即"答案应出自的块集合"——golden 标注成本低，不依赖人工抄 block_id
- 检索层：逐条 query 走 search_docs(top_k) → recall@k / MRR（每用例 0/1，
  至少一个期望块进 top-k 即命中；anchor 定位失败 = golden 缺陷，单独统计不计分母）
- 答案层（--answers 可选，需 Key）：run_question 全链 → 每条 sources 的
  block_id 可回查率（M3 幽灵引用已代码层剔除，此处验证结果层无假出处）
- 门槛判定：真向量(dashscope)下 recall@5 >= 0.8 过闸；mock 无语义，跳过门槛
  只保链路；--answers 且有引用用例时引用可回查率必须 100%

报告：控制台表格 + data/reports/eval_report_latest.json（含耗时与 chat 成本聚合，
为后续"与上次对比"留底）。
"""
from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

from . import config
from . import qa as qa_module
from .vectorstore import store as vector_store
from .vectorstore import embedding as embed_module

DEFAULT_GOLDEN_PATH = config.BASE_DIR / "data" / "golden" / "qa_golden.json"
REPORT_DIR = config.BASE_DIR / "data" / "reports"
LATEST_REPORT = REPORT_DIR / "eval_report_latest.json"

RETRIEVAL_GATE_RECALL = 0.8  # R4 门槛：至少一期望块进 top-k 的用例占比


# ---------------- golden 加载与校验 ----------------

def load_golden(path: str | Path | None = None) -> list[dict]:
    """读取 golden 用例列表；meta 块与 cases 并存的文件取 cases。"""
    golden_path = Path(path) if path else DEFAULT_GOLDEN_PATH
    payload = json.loads(golden_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return list(payload["cases"])
    return list(payload)


def validate_golden(cases: list[dict]) -> list[str]:
    """结构性校验：必填字段齐全、id 唯一、query/anchor 非空。

    返回错误清单（空 = 通过）；评估引擎遇错误用例直接跳过并计入 defects。
    """
    errors: list[str] = []
    seen_ids: set[str] = set()
    for index, case in enumerate(cases):
        case_id = (case.get("id") or "").strip()
        if not case_id:
            errors.append(f"#{index} 缺字段: ['id']")
            continue
        if case_id in seen_ids:  # id 去重独立于字段校验：缺字段的用例也要占位
            errors.append(f"id 重复: {case_id}")
        seen_ids.add(case_id)

        missing = [
            field for field in ("query", "expect_doc", "anchor")
            if not (case.get(field) or "").strip()
        ]
        if missing:
            errors.append(f"{case_id} 缺字段: {missing}")
        elif len(case.get("query") or "") < 4 or len(case.get("anchor") or "") < 4:
            # 字段齐备但太短无法定位（缺字段已报，不双报）。下限 4 字符：
            # CJK 锚句 5 字已可唯一定位（如"安装前审查"），6 字下限曾误报合法短锚
            errors.append(f"{case_id} query/anchor 过短，无法定位")
    return errors


# ---------------- 期望块定位（anchor → 块 id 集合） ----------------

def _normalize(text: str) -> str:
    """归一化：去所有空白 + 小写（PDF 抽取的单词间空白不可靠，大小写不敏感匹配）。"""
    return re.sub(r"\s+", "", text or "").lower()


def _block_index() -> tuple[list[str], list[dict], list[str]]:
    """全库块 id / metadata / 归一化文本（每次评估现拉，块级小库 <1s；大库可后续缓存）。"""
    collection_data = vector_store.get_collection().get(
        include=["metadatas", "documents"]
    )
    block_ids = collection_data["ids"]
    metadatas = collection_data["metadatas"]
    normed_texts = [_normalize(text) for text in collection_data["documents"]]
    return block_ids, metadatas, normed_texts


def locate_expected_blocks(anchor: str, expect_doc: str) -> tuple[list[str], list[str]]:
    """anchor 全库归一化子串匹配 → 期望块 id 列表（按库序）+ 提示清单。

    命中块都算期望（同一信息常出现在摘要与正文两块）；跨文档命中说明 anchor
    太泛（如页眉共现的公司名），记入提示供 golden 修订，不影响定位结果。
    """
    block_ids, metadatas, normed_texts = _block_index()
    target = _normalize(anchor)
    expected: list[str] = []
    warnings: list[str] = []
    for block_id, metadata, normed in zip(block_ids, metadatas, normed_texts):
        if target in normed:
            expected.append(block_id)
            if (metadata or {}).get("doc_id") != expect_doc:
                warnings.append(f"{block_id} 命中但 doc_id≠期望({expect_doc})")
    if not expected:
        warnings.append(f"anchor 无命中（golden 缺陷，需修订 anchor 或重灌文档）")
    return expected, warnings


# ---------------- 指标计算（纯函数，verify 直接单测） ----------------

def compute_case_outcome(retrieved_ids: list[str], expected_ids: list[str]) -> dict:
    """单个用例命中判定：期望块集合与检索 top-k 的交集。

    返回 {hit: bool, rank: int|None}——rank 取首个期望块在检索序中的位置(1 起)。
    """
    expected_set = set(expected_ids)
    for rank, block_id in enumerate(retrieved_ids, start=1):
        if block_id in expected_set:
            return {"hit": True, "rank": rank}
    return {"hit": False, "rank": None}


def aggregate_metrics(case_outcomes: list[dict]) -> dict:
    """检索层指标聚合：recall@k = 命中用例占比；MRR = 1/首个命中 rank 的均值。"""
    if not case_outcomes:
        return {"cases": 0, "recall": 0.0, "mrr": 0.0}
    hit_count = sum(1 for outcome in case_outcomes if outcome["hit"])
    reciprocal_ranks = [
        1.0 / outcome["rank"] for outcome in case_outcomes if outcome["hit"] and outcome["rank"]
    ]
    return {
        "cases": len(case_outcomes),
        "recall": round(hit_count / len(case_outcomes), 4),
        "mrr": round(statistics.mean(reciprocal_ranks), 4) if reciprocal_ranks else 0.0,
    }


def _merge_usage(total: dict, usage: dict) -> dict:
    """累加 chat usage 并补 total_tokens。

    llm.chat_completion 只返回 prompt/completion_tokens（见 llm.py），
    total_tokens 由本层求和补出，保证报告成本聚合字段齐全。
    """
    for key in ("prompt_tokens", "completion_tokens"):
        value = (usage or {}).get(key)
        if value is not None:
            total[key] = total.get(key, 0) + int(value)
    total["total_tokens"] = total.get("prompt_tokens", 0) + total.get("completion_tokens", 0)
    return total


# ---------------- 两层评估 ----------------

def run_retrieval_eval(cases: list[dict], top_k: int = 5) -> dict:
    """检索层：逐条 query 真检索（embedding 成本小），产出指标与逐条明细。"""
    start = time.perf_counter()
    provider = embed_module.info()
    expected_cache: dict[str, list[str]] = {}
    anchor_warnings: dict[str, list[str]] = {}
    rows: list[dict] = []
    valid_outcomes: list[dict] = []

    for case in cases:
        case_id = case["id"]
        expected, warnings = locate_expected_blocks(case["anchor"], case["expect_doc"])
        expected_cache[case_id] = expected
        anchor_warnings[case_id] = warnings

    for case in cases:
        case_id = case["id"]
        expected = expected_cache[case_id]
        if not expected:  # golden 缺陷：不计分母，防缺陷拉低真实指标
            rows.append({"id": case_id, "skipped": True, "reason": "anchor 未定位到期望块"})
            continue
        hits = vector_store.search_docs([case["query"]], top_k=top_k)
        retrieved_ids = [hit["id"] for hit in hits]
        outcome = compute_case_outcome(retrieved_ids, expected)
        valid_outcomes.append(outcome)
        rows.append({
            "id": case_id,
            "expected": expected,
            "retrieved": retrieved_ids,
            "hit": outcome["hit"],
            "rank": outcome["rank"],
            "top1_doc": (hits[0]["metadata"] or {}).get("doc_id") if hits else None,
            "top1_distance": round(hits[0]["distance"], 4) if hits else None,
        })

    metrics = aggregate_metrics(valid_outcomes)
    metrics["top_k"] = top_k
    defects = [warning for warnings in anchor_warnings.values() for warning in warnings]
    return {
        "provider": provider["provider"],
        "degraded": provider["degraded"],
        "elapsed_s": round(time.perf_counter() - start, 2),
        "metrics": metrics,
        "rows": rows,
        "defects": defects,
        "usage": {},  # 检索层无 LLM chat 成本；embedding 成本不在 usage 内
    }


def run_answer_eval(cases: list[dict]) -> dict:
    """答案层（可选，需真 Key）：run_question 全链 → 引用可回查率。

    每条 source 带 block_id（qa._build_source 契约）；可回查 = block_id 非空且
    存在于向量库 id 集合。幽灵引用已由 citations 节点代码层剔除，本层验证结果。
    """
    start = time.perf_counter()
    block_ids, _, _ = _block_index()
    existing_ids = set(block_ids)

    rows: list[dict] = []
    usage_total: dict = {}
    degraded = False
    for case in cases:
        outcome = qa_module.run_question(case["query"])
        _merge_usage(usage_total, outcome.get("usage"))
        sources = outcome.get("sources") or []
        if outcome.get("degraded"):
            degraded = True
        untraceable = [
            source.get("block_id") for source in sources
            if not source.get("block_id") or source.get("block_id") not in existing_ids
        ]
        rows.append({
            "id": case["id"],
            "answer": outcome.get("answer") or "",
            "source_count": len(sources),
            "traceable": not untraceable,
            "untraceable_ids": untraceable,
            "note": outcome.get("note"),
        })

    referenced_cases = [row for row in rows if row["source_count"] > 0]
    traceable_cases = [row for row in referenced_cases if row["traceable"]]
    return {
        "elapsed_s": round(time.perf_counter() - start, 2),
        "degraded": degraded,
        "cases_total": len(rows),
        "referenced_cases": len(referenced_cases),
        "traceable_rate": round(len(traceable_cases) / len(referenced_cases), 4) if referenced_cases else None,
        "usage": usage_total,
        "rows": rows,
    }


# ---------------- 门槛判定 ----------------

def eval_gate(retrieval_metrics: dict, provider: str, answer_summary: dict | None = None) -> dict:
    """回归门槛（R4）：真向量下 recall@5 >= 0.8；答案层有引用时可回查率 100%。

    返回 {passed: bool|None, checks: [str]}——passed=None 表示门槛不适用
    （mock 无语义 / 无可评估用例），链路本身无错不算失败。
    """
    checks: list[str] = []
    passed: bool | None = True

    if provider == "mock":
        return {"passed": None, "checks": ["mock 向量无语义，跳过 recall 门槛（仅链路验证）"]}

    recall = retrieval_metrics.get("recall", 0.0)
    case_count = retrieval_metrics.get("cases", 0)
    if case_count == 0:
        return {"passed": None, "checks": ["golden 全部 anchor 定位失败，无法评估（需修 golden）"]}

    checks.append(f"recall@{retrieval_metrics['top_k']} = {recall}（门槛 ≥ {RETRIEVAL_GATE_RECALL}，{case_count} 用例）")
    if recall < RETRIEVAL_GATE_RECALL:
        passed = False

    if answer_summary is not None and not answer_summary.get("degraded"):
        traceable_rate = answer_summary.get("traceable_rate")
        referenced_cases = answer_summary.get("referenced_cases", 0)
        if referenced_cases == 0:
            checks.append("答案层 0 个带引用用例（模型未产出引用，需查提示词）")
            passed = False
        else:
            checks.append(f"引用可回查率 = {traceable_rate}（门槛 1.0，{referenced_cases} 条带引用）")
            if traceable_rate != 1.0:
                passed = False
    return {"passed": passed, "checks": checks}


# ---------------- 顶层入口 ----------------

def run_eval(
    top_k: int = 5,
    with_answers: bool = False,
    golden_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict:
    """一键评估：golden → 检索层 + 可选答案层 → 指标 → 门槛 → 落盘报告。

    report_path 可注入（verify 脚本写临时目录，避免污染真实 data/reports）。
    """
    cases = load_golden(golden_path)
    validation_errors = validate_golden(cases)
    if validation_errors:
        # golden 结构性问题不阻断（engine 会跳过缺陷用例），但报告里显眼提示
        pass

    retrieval = run_retrieval_eval(cases, top_k=top_k)
    answer = run_answer_eval(cases) if with_answers else None
    gate = eval_gate(retrieval["metrics"], retrieval["provider"], answer)

    report = {
        "meta": {
            "golden": str(Path(golden_path) if golden_path else DEFAULT_GOLDEN_PATH),
            "cases_total": len(cases),
            "golden_errors": validation_errors,
            "provider": retrieval["provider"],
            "degraded": retrieval["degraded"],
            "top_k": top_k,
            "with_answers": with_answers,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "retrieval": {
            "metrics": retrieval["metrics"],
            "elapsed_s": retrieval["elapsed_s"],
            "defects": retrieval["defects"],
            "rows": retrieval["rows"],
        },
        "answer": answer,
        "gate": gate,
    }
    target_path = Path(report_path) if report_path else LATEST_REPORT
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


# ---------------- 控制台渲染 ----------------

def print_report(report: dict) -> None:
    """报告行渲染：检索表格 + 答案层 + 门槛判定（中文，一眼可读）。"""
    meta = report["meta"]
    provider_tag = f"{meta['provider']}" + ("（mock 降级）" if meta["degraded"] else "")
    print(f"评估报告  golden {report['retrieval']['metrics']['cases']}/{meta['cases_total']} 用例"
          f"  provider={provider_tag}  top_k={meta['top_k']}")
    print("-" * 96)

    rows = [row for row in report["retrieval"]["rows"] if not row.get("skipped")]
    for row in rows:
        expected_short = ", ".join(block_id.split(":")[-1] for block_id in row["expected"][:3])
        mark = "✔" if row["hit"] else "✘"
        rank_text = f"rank={row['rank']}" if row["hit"] else "miss"
        print(f"  {row['id']:<5} {mark}  {rank_text:<9} 期望块[{expected_short}]"
              f"  top1={row['top1_doc']}")
    skipped = [row for row in report["retrieval"]["rows"] if row.get("skipped")]
    if skipped:
        print(f"  跳过(golden 缺陷): {[row['id'] for row in skipped]}")

    metrics = report["retrieval"]["metrics"]
    print("-" * 96)
    print(f"检索指标  recall@{metrics['top_k']} = {metrics['recall']}   MRR = {metrics['mrr']}"
          f"   耗时 {report['retrieval']['elapsed_s']}s")

    answer = report["answer"]
    if answer:
        usage = answer["usage"]
        print(f"答案指标  带引用 {answer['referenced_cases']}/{answer['cases_total']} 条"
              f"   可回查率 {answer['traceable_rate']}"
              + (f"   chat tokens {usage.get('total_tokens', 0)}" if usage else "")
              + ("   [降级: 无 Key 未产出引用]" if answer["degraded"] else "")
              + f"   耗时 {answer['elapsed_s']}s")
        untraceable_rows = [row for row in answer["rows"] if row["source_count"] and not row["traceable"]]
        if untraceable_rows:
            print(f"  不可回查: {[(row['id'], row['untraceable_ids']) for row in untraceable_rows]}")

    gate = report["gate"]
    print("-" * 96)
    for check in gate["checks"]:
        print(f"  · {check}")
    if gate["passed"] is True:
        print("门槛判定: PASS")
    elif gate["passed"] is None:
        print("门槛判定: SKIP（门槛不适用）")
    else:
        print("门槛判定: FAIL")
