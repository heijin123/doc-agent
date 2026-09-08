"""日期元数据（doc_date/doc_year/ingested_at）确定性验证（无需 LLM Key）。

背景（2026-09-08）：md5 幂等只算块文本，metadata 新键不触发重嵌——本脚本验证
新键的解析、注入与检索使用三个环节：
1. extract_doc_date 纯函数：文件名 → 文档日期（完整日期/仅年份/无日期/非法回退）
2. extract_year_filter 纯函数：问题显式年份 → doc_year 过滤条件（防跨年报表张冠李戴）
3. 问答契约：上下文块头带"文档日期"提示；sources 透传 doc_date
4. 端到端（临时库 + mock embedding）：upsert 落库 ingested_at 注入；
   search_docs where 年份过滤只回该年份块（无日期块被排除）
5. 年份过滤无命中 → 自动降级为不限年份重查（note 记录，不把可答问题答成未找到）

运行：.venv/Scripts/python.exe scripts/verify_date_metadata.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

# 环境前置（import docagent 前）：mock 向量 + 临时 chroma 目录
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="verify_date_"))
os.environ["DOCAGENT_CHROMA_DIR"] = str(_TMP_ROOT / "chroma")
os.environ["DOCAGENT_IMAGES_DIR"] = str(_TMP_ROOT / "images")
os.environ["DOCAGENT_IMAGES_DB"] = str(_TMP_ROOT / "image_files.db")
os.environ["EMBEDDING_PROVIDER"] = "mock"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import docagent.config as config  # noqa: E402
from docagent import qa as qa_module  # noqa: E402
from docagent.ingest.chunk_models import DocumentChunk, extract_doc_date  # noqa: E402
from docagent.qa import extract_year_filter, run_question  # noqa: E402
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


def _fake_hit(block_id: str = "tianli:0", doc_date: str | None = "2024") -> dict:
    return {
        "id": block_id,
        "text": "营业收入为 1,234 万元，归属于上市公司股东的净利润为 567 万元。",
        "metadata": {"doc_id": "tianli", "title": "天力锂能：天力锂能集团股份有限公司2024年年度报告（更正后）",
                     "page": 3, "block_type": "page", "doc_date": doc_date},
        "distance": 0.5,
    }


# ---------------- 1. extract_doc_date 纯函数 ----------------

def verify_extract_doc_date() -> None:
    print("\n== 1. extract_doc_date：文件名 → 文档日期 ==")
    cases = [
        # (文件名, 期望)
        ("振华重工：振华重工2025年年度报告（英文版）", "2025"),                       # 仅年份（年报）
        ("天力锂能：天力锂能集团股份有限公司2024年年度报告（更正后）", "2024"),       # 仅年份（更正后后缀不影响）
        ("2025-04-29_回购进展公告", "2025-04-29"),                                    # 连字符完整日期
        ("2025.4.9_运营简报", "2025-04-09"),                                          # 点分隔 + 月日补零
        ("2025年4月29日回购进展公告", "2025-04-29"),                                   # 中文完整日期
        ("Annual_Report_2025_ZH", "2025"),                                            # 英文年报（年份在后）
        ("1_hermes_agent_manual_cn", None),                                           # 手册：无日期 → None
        ("2_react_paper_en", None),                                                   # 论文：数字前缀不误判
        ("3_invoice_table_cn", None),                                                 # 发票样本
        ("乐鑫科技：乐鑫科技关于以集中竞价交易方式回购公司股份比例累计达1%暨回购进展公告", None),  # 公告标题无年份
        ("项目规划_2025-99-99_v2", "2025"),                                            # 非法月日：放弃整段、年份兜底
    ]
    for file_name, expected in cases:
        got = extract_doc_date(file_name)
        check(got == expected, f"extract_doc_date({file_name[:24]}…) = {got!r}", f"期望 {expected!r}")


# ---------------- 2. extract_year_filter 纯函数 ----------------

def verify_extract_year_filter() -> None:
    print("\n== 2. extract_year_filter：问题显式年份 → where 条件 ==")
    cases = [
        ("天力锂能 2024 年年度报告的公告日期是什么时候？", {"doc_year": {"$in": [2024]}}),
        ("2024年报与2025年报的营收对比如何？", {"doc_year": {"$in": [2024, 2025]}}),
        ("振华重工 Annual Report 2025 的股票代码？", {"doc_year": {"$in": [2025]}}),
        ("乐鑫科技本次回购了多少股？", None),                 # 无显式年份不过滤
        ("累计回购 1,420,836 股占总股本比例？", None),        # 数字串非年份
        ("订单号 20251234 的物流状态", None),                 # 年份后接数字：4 位年号边界防误判
    ]
    for question, expected in cases:
        got = extract_year_filter(question)
        check(got == expected, f"extract_year_filter({question[:20]}…) = {got!r}", f"期望 {expected!r}")


# ---------------- 3. 问答契约：日期提示 + sources 透传 ----------------

def verify_qa_date_contract() -> None:
    print("\n== 3. 问答契约：上下文带日期提示、sources 透传 doc_date ==")
    messages = qa_module._build_answer_messages("天力锂能 2024 年营收？", [_fake_hit()])
    system_prompt = messages[0]["content"]
    check("文档日期 2024" in system_prompt, "上下文块头带文档日期提示", "见 system_prompt")

    hit_without_date = _fake_hit(block_id="manual:0", doc_date=None)
    messages_no_date = qa_module._build_answer_messages("手册安装前要做什么？", [hit_without_date])
    check("文档日期" not in messages_no_date[0]["content"], "无日期文档不写日期行（提示不编造）")

    source = qa_module._build_source(_fake_hit())
    check(source.get("doc_date") == "2024", "sources 契约透传 doc_date", f"{source}")
    source_no_date = qa_module._build_source(hit_without_date)
    check(source_no_date.get("doc_date") is None, "无日期文档 doc_date=None", f"{source_no_date}")


# ---------------- 4. 端到端（临时库 + mock embedding） ----------------

def _make_chunk(title: str, text: str, doc_year: int | None = None, doc_date: str | None = None) -> DocumentChunk:
    metadata: dict = {"title": title, "block_type": "paragraph"}
    if doc_year is not None:
        metadata["doc_year"] = doc_year
        metadata["doc_date"] = doc_date or str(doc_year)
    return DocumentChunk(text=text, metadata=metadata)


def verify_upsert_and_where_filter() -> None:
    print("\n== 4. 端到端：ingested_at 落库 + search_docs 年份过滤 ==")
    chunks_2024 = [_make_chunk("某公司2024年年度报告", "2024 年营业收入为 10 亿元。", 2024),
                   _make_chunk("某公司2024年年度报告", "2024 年净利润为 1 亿元。", 2024)]
    chunks_2025 = [_make_chunk("某公司2025年年度报告", "2025 年营业收入为 12 亿元。", 2025)]
    chunks_manual = [_make_chunk("操作手册", "安装前请先审查第三方插件的来源。")]

    outcome = vector_store.upsert_doc_chunks(chunks_2024 + chunks_2025 + chunks_manual, "corp")
    check(outcome["stored"] == 4 and outcome["skipped"] == 0, "4 块全部落库", f"{outcome}")

    data = vector_store.get_collection().get(include=["metadatas"])
    ingested_at_values = {(meta or {}).get("ingested_at") for meta in data["metadatas"]}
    check(len(ingested_at_values) == 1 and all(value for value in ingested_at_values),
          "ingested_at 已注入且同批一致", f"{ingested_at_values}")

    hits_2024 = vector_store.search_docs(["2024 年营业收入"], top_k=5, where={"doc_year": {"$in": [2024]}})
    check(hits_2024 and all((hit["metadata"].get("doc_year")) == 2024 for hit in hits_2024),
          "where 年份过滤：只回 2024 块", f"hit 年份={[(h['metadata'].get('doc_year'), h['metadata'].get('doc_date')) for h in hits_2024]}")

    hits_manual_only = vector_store.search_docs(["安装审查"], top_k=5, where={"doc_year": {"$in": [2024]}})
    check(all((hit["metadata"].get("doc_year")) == 2024 for hit in hits_manual_only),
          "无日期（doc_year 缺失）块被年份过滤排除", f"n={len(hits_manual_only)}")

    hits_no_where = vector_store.search_docs(["营业收入"], top_k=10)
    check(len(hits_no_where) == 4, "不过滤时全量候选（保持原行为）", f"n={len(hits_no_where)}")

    # 幂等：md5 相同重跑全 skip，ingested_at 保持首次值（不刷新）
    ingested_before = sorted(value for value in ingested_at_values)
    outcome2 = vector_store.upsert_doc_chunks(chunks_2024 + chunks_2025 + chunks_manual, "corp")
    check(outcome2["stored"] == 0 and outcome2["skipped"] == 4, "md5 幂等：重跑全 skip", f"{outcome2}")
    data2 = vector_store.get_collection().get(include=["metadatas"])
    ingested_after = sorted({(meta or {}).get("ingested_at") for meta in data2["metadatas"]})
    check(ingested_before == ingested_after, "skip 块 ingested_at 保留首入时间（不刷新）")


# ---------------- 5. 年份两段式检索：护栏 / 跑题回退 / 降级 note ----------------

def _run_with_fake_search(side_effect, question: str, chat_text: str = "答复[1]。",
                          chat_capture: list | None = None) -> tuple[dict, mock.Mock]:
    """以注入式 search_docs（按序返回）+ mock chat 跑一次问答，返回 (outcome, fake_search)。

    chat_capture：若给 list，则把每次调用发给模型的 system prompt 追加进去
    （用于断言"模型实际看到的是哪些块"，而非 mock 出来的答复文本）。
    """
    original_key = config.DASHSCOPE_API_KEY
    config.DASHSCOPE_API_KEY = "test-key"  # llm 被 patch，不真调
    try:
        fake_search = mock.Mock(side_effect=side_effect)

        def _fake_chat(messages, temperature=0.2):
            if chat_capture is not None:
                chat_capture.append(messages[0]["content"])
            return {"text": chat_text, "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

        with mock.patch.object(vector_store, "search_docs", fake_search), \
             mock.patch.object(qa_module.llm, "chat_completion", side_effect=_fake_chat):
            outcome = run_question(question)
        return outcome, fake_search
    finally:
        config.DASHSCOPE_API_KEY = original_key


def verify_search_year_filter_and_fallback() -> None:
    print("\n== 5. 年份两段式检索：护栏 / 跑题回退 / 无年份单查 ==")

    # 场景 A：目标年份有该主题 → 过滤结果启用（护栏生效），无 note
    year_hit = _fake_hit()  # doc_id=tianli、doc_year=2024
    outcome_a, search_a = _run_with_fake_search(
        [[_fake_hit()], [_fake_hit()]],  # semantic 先、year 过滤后（按调用序）
        "天力锂能 2024 年报的营业收入是多少？",
    )
    calls_a = search_a.call_args_list
    check(len(calls_a) == 2 and calls_a[0].kwargs.get("where") is None
          and calls_a[1].kwargs.get("where") == {"doc_year": {"$in": [2024]}},
          "年份 query → 先语义后年份过滤两次检索", f"{[c.kwargs.get('where') for c in calls_a]}")
    check(outcome_a.get("note") is None, "主题年份一致 → 无降级 note", f"{outcome_a.get('note')}")

    # 场景 B：目标年份无该主题（过滤空）→ 回退语义结果 + note
    outcome_b, search_b = _run_with_fake_search(
        [[_fake_hit()], []],  # semantic 有 tianli 块；2026 年过滤为空
        "乐鑫科技 2026 年公告里累计回购了多少股？",
        chat_text="乐鑫公告里回购了 1,420,836 股[1]。",
    )
    check(len(search_b.call_args_list) == 2, "两段都执行（过滤空也做语义基准）")
    check(outcome_b.get("note") and "2026" in outcome_b["note"] and "不限年份" in outcome_b["note"],
          "过滤空 → note 明示无该年文档、已按语义检索", f"{outcome_b.get('note')}")
    check("1,420,836" in outcome_b.get("answer", ""), "回退后仍正常作答（不答成未找到）",
          f"{outcome_b.get('answer')[:40]!r}")

    # 场景 C：目标年份命中但跑题主题（问振华 2024、库里只有振华 2025 + 天力 2024）
    #         → 语义 top1 是振华 2025、年份过滤命中全是天力 → doc_id 不一致 → 回退语义，
    #           模型看到的是振华块（可答"无 2024，2025 年报代码 600320"），而非跑题天力块
    zhenhua_hit = {
        "id": "zhenhua:0",
        "text": "Zhenhua Heavy Industries stock code is 600320, listed on the Shanghai Stock Exchange.",
        "metadata": {"doc_id": "zhenhua", "title": "振华重工：振华重工2025年年度报告（英文版）",
                     "page": 1, "block_type": "page", "doc_year": 2025, "doc_date": "2025"},
        "distance": 0.3,
    }
    tianli_hit = _fake_hit()  # 2024 年份过滤命中 = 天力锂能（跑题主题）
    captured_prompts: list[str] = []
    outcome_c, search_c = _run_with_fake_search(
        [[zhenhua_hit], [tianli_hit]],  # 调用序：semantic → year 过滤
        "振华重工 2024 年报的股票代码是什么？",
        chat_text="知识库暂无振华重工 2024 年报；其 2025 年报股票代码为 600320[1]。",
        chat_capture=captured_prompts,
    )
    check(len(search_c.call_args_list) == 2, "跑题场景两段都执行")
    check(outcome_c.get("note") and "2024" in outcome_c["note"] and "不限年份" in outcome_c["note"],
          "过滤跑题 → 回退语义并 note 提示", f"{outcome_c.get('note')}")
    visible_prompt = captured_prompts[0] if captured_prompts else ""
    check("振华重工：振华重工2025年年度报告" in visible_prompt and "天力锂能" not in visible_prompt,
          "回退后模型看到语义命中的振华块（而非跑题的天力块）",
          f"prompt 块头={[line[:36] for line in visible_prompt.splitlines() if line.startswith('[')]}")

    # 场景 D：无显式年份 → 只查一次（无 where），保持原成本
    outcome_d, search_d = _run_with_fake_search(
        [[_fake_hit()]],
        "天力锂能年报的营业收入是多少？",
    )
    search_d.assert_called_once()
    check(search_d.call_args.kwargs.get("where") is None, "无年份 query → 单次语义检索", "见 call")


if __name__ == "__main__":
    verify_extract_doc_date()
    verify_extract_year_filter()
    verify_qa_date_contract()
    verify_upsert_and_where_filter()
    verify_search_year_filter_and_fallback()

    shutil.rmtree(_TMP_ROOT, ignore_errors=True)
    print(f"\n结果：{len(passed_cases)} 通过 / {len(failed_cases)} 失败")
    if failed_cases:
        print("失败项：", failed_cases)
        sys.exit(1)
    print("ALL_PASS")
