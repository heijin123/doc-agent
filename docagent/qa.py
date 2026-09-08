"""问答编排（M3 消费侧）：检索 → 生成 → 引用校验，LangGraph 状态机。

与摄取流水线同框架两种用法（需求 §4.2 核心叙事）：
- 摄取图 = 数据流水线（探测/切块/落库，确定性为主）
- 问答图 = 消费侧（检索确定性 + 生成模型），节点少而薄

节点链：
    route ──文档问题──▶ search ──▶ answer ──▶ citations ──▶ END
       │
       └──问候/闲聊──▶ greet（不检索，固定简短回应）──▶ END

引用溯源约束（需求 §5.2 核心）：
1. search 返回 top-k 块，每块按顺序编号 [1..k]，完整进上下文；
2. answer 提示词：只允许引用上下文里的块，格式 [n]；
3. citations 节点代码层校验：答案中 [n] 若越界（n>k 或非数字）→ 删角标
   不删句子（幽灵引用剔除，demo1 strip_ghost 同款——不赌模型）；
   有效引用按编号去重组装成前端契约 sources。
4. 无有效引用 → 答案追加"综合回答、未找到直接文档出处"提示。

MVP 简答（需求 §5.2 拍板）：单轮检索问答，thread 历史记忆不消费
（demo1 已证多轮，此处不做演示重点）；无 Key / mock 环境降级不崩（R7）。
"""
from __future__ import annotations

import re
import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

from . import llm
from .vectorstore import store as vector_store
from .vectorstore import embedding as embed_module

# 引用角标：[1]、[12]；越界/非法编号一律剔除
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_IMAGE_PLACEHOLDER = re.compile(r"\[IMAGE:[A-Za-z0-9_.-]+\]")
# 问题中的显式年份（与摄取侧 extract_doc_date 同窗口 2000-2029，防编号串误判）
_YEAR_IN_QUESTION = re.compile(r"(?<!\d)(20[0-2]\d)(?!\d)")

SEARCH_TOP_K = 5
SNIPPET_CHAR_LIMIT = 160       # sources.snippet 长度（前端悬浮展示）
BLOCK_TEXT_CHAR_LIMIT = 1500   # 单块进上下文的长度上限（防 prompt 膨胀）


def extract_year_filter(question: str) -> dict | None:
    """问题中显式年份 → 检索的 doc_year 过滤条件（防跨年报表张冠李戴）。

    收集问题里出现的全部年份："2024 年报与 2025 年报营收对比" → $in [2024, 2025]；
    无显式年份 → None（不过滤，保持现状）。纯正则零成本，只认 4 位显式年份——
    "去年/最新一期"这类相对时间词初级版不处理（需模型解析，留给后续版）。
    """
    years = sorted({int(match.group(1)) for match in _YEAR_IN_QUESTION.finditer(question or "")})
    if not years:
        return None
    return {"doc_year": {"$in": years}}

# 问候/闲聊兜底（规则先行，不请模型）：纯寒暄不触发检索
_GREETING_KEYWORDS = ("你好", "您好", "hi", "hello", "嗨", "谢谢", "感谢", "再见", "你是谁", "在吗", "在么")
_NON_DOCUMENT_HINT = "（注：以上为综合回答，知识库中未检索到可直接引用的文档出处。）"


class QAState(TypedDict, total=False):
    question: str
    needs_search: bool            # route 判定：文档问题才检索
    hits: list                    # search 产物 [{id, text, metadata, distance}]
    raw_answer: str               # answer 节点 LLM 原文（含 [n] 角标）
    answer: str                   # citations 清洗后的最终答复
    sources: list                 # 前端契约 [{document_name, page_number, snippet}]
    usage: dict                   # chat token 成本
    note: str | None              # 降级/异常说明
    degraded: bool                # 无 Key / mock 环境标记
    timings: dict[str, float]


def _timed(state: QAState, key: str, elapsed: float) -> dict:
    timings = dict(state.get("timings") or {})
    timings[key] = round(elapsed, 3)
    return timings


# ---------------- 节点 ----------------

def route_node(state: QAState) -> dict:
    """判别是否需要检索：问候/闲聊不检索（greet 直答），其余走检索。"""
    question = (state.get("question") or "").strip()
    lowered = question.lower()
    is_greeting = (
        len(question) <= 40
        and any(keyword in lowered for keyword in _GREETING_KEYWORDS)
    )
    return {"needs_search": not is_greeting}


def greet_node(state: QAState) -> dict:
    """寒暄回应：固定简短文本，不检索、不引用（零模型成本）。"""
    return {
        "answer": "你好，我是 doc-agent 文档助手。把文档摄取进知识库后，可以问我文档里的内容（会带出处回答）。",
        "sources": [],
        "usage": {},
        "degraded": False,
    }


def search_node(state: QAState) -> dict:
    """知识库向量检索（两段式，2026-09-08 起带年份感知，防跨年报表张冠李戴）。

    恒做一次不限年份的语义检索作基准；问题含显式年份时再加一次 doc_year
    过滤检索，并按下述规则裁决：
    - 过滤命中与语义 top1 同 doc_id（该主题在目标年份确有内容）→ 用过滤结果，
      年份护栏生效——问 2025 年报绝不会答成 2024 的数字；
    - 过滤命中跑题（如库中只有"振华 2025"，问"振华 2024"过滤到的是天力 2024）
      或过滤为空 → 回退语义结果 + note 明示"目标年份无此主题文档"，
      不把可答问题过滤成"未找到"（模型据块头文档日期自行注意年份）。
    """
    start = time.perf_counter()
    question = state["question"]
    try:
        semantic_hits = vector_store.search_docs([question], top_k=SEARCH_TOP_K)
        where = extract_year_filter(question)
        if not where:
            return {
                "hits": semantic_hits,
                "degraded": embed_module.info()["degraded"],
                "timings": _timed(state, "search", time.perf_counter() - start),
            }

        years = sorted(where["doc_year"]["$in"])
        year_hits = vector_store.search_docs([question], top_k=SEARCH_TOP_K, where=where)
        top_doc_id = ((semantic_hits[0] or {}).get("metadata") or {}).get("doc_id") if semantic_hits else None
        topic_matched = bool(top_doc_id) and any(
            ((hit.get("metadata") or {}).get("doc_id") == top_doc_id) for hit in year_hits
        )
        if topic_matched:
            hits, note = year_hits, None
        else:
            hits = semantic_hits
            reason = f"“{top_doc_id}”" if top_doc_id else "该主题"
            note = f"知识库无 {years} 年相关文档，已按语义不限年份检索（{reason} 的近期文档）"
        return {
            "hits": hits,
            "note": note,
            "degraded": embed_module.info()["degraded"],
            "timings": _timed(state, "search", time.perf_counter() - start),
        }
    except Exception as exc:
        return {
            "hits": [],
            "note": f"检索失败: {exc}",
            "degraded": True,
            "timings": _timed(state, "search", time.perf_counter() - start),
        }


def answer_node(state: QAState) -> dict:
    """基于检索块生成带 [n] 引用的答复；hits 空或无 Key 时降级不调模型。"""
    start = time.perf_counter()
    hits = state.get("hits") or []
    if not hits:
        return {
            "raw_answer": "知识库中未找到与问题相关的内容（可先摄取文档，或换个问法）。",
            "note": "检索无命中",
            "timings": _timed(state, "answer", time.perf_counter() - start),
        }

    prompt = _build_answer_messages(state["question"], hits)
    completion = llm.chat_completion(prompt, temperature=0.2)
    if completion is None:
        return {
            "raw_answer": "（未配置模型 Key 或模型调用失败：已检索到文档内容，但无法生成带出处的回答。）",
            "note": "chat 未可用（无 Key/调用失败）",
            "degraded": True,
            "timings": _timed(state, "answer", time.perf_counter() - start),
        }
    return {
        "raw_answer": completion["text"],
        "usage": completion["usage"],
        "timings": _timed(state, "answer", time.perf_counter() - start),
    }


def citations_node(state: QAState) -> dict:
    """幽灵引用剔除 + sources 组装：只保留对可见块的引用，越界角标删除。"""
    raw_answer = (state.get("raw_answer") or "").strip()
    hits = state.get("hits") or []
    valid_count = len(hits)

    cleaned_answer, referenced_numbers = strip_out_of_range_citations(raw_answer, valid_count)
    if not referenced_numbers and hits and cleaned_answer and "未找到" not in cleaned_answer[:20]:
        cleaned_answer = cleaned_answer + "\n\n" + _NON_DOCUMENT_HINT

    sources = [
        _build_source(hits[number - 1]) for number in referenced_numbers
    ]
    return {"answer": cleaned_answer, "sources": sources}


def strip_out_of_range_citations(raw_answer: str, valid_count: int) -> tuple[str, list[int]]:
    """幽灵引用剔除（纯函数，verify 可独立单测）：只保留 1..valid_count 的角标。

    越界/非法编号的角标被删除、句子文本保留（demo1 strip_ghost 教训：删标注
    不删内容——答案本身是对的，只是出处无法核实）；返回 (清洗文本, 有效编号序)。
    """
    cleaned = _CITATION_PATTERN.sub(
        lambda match: match.group(0) if 1 <= int(match.group(1)) <= valid_count else "",
        raw_answer,
    ).strip()
    referenced_numbers: list[int] = []
    for match in _CITATION_PATTERN.finditer(raw_answer):
        number = int(match.group(1))
        if 1 <= number <= valid_count and number not in referenced_numbers:
            referenced_numbers.append(number)
    return cleaned, referenced_numbers


# ---------------- 辅助 ----------------

def _build_answer_messages(question: str, hits: list) -> list[dict]:
    """组装 system（引用约束）+ user（编号块 + 问题）。块文本里的 [IMAGE] 占位转 (图)。

    块头带文档日期（doc_date，2026-09-08 起）：模型据日期判断回答口径——
    问题问 2025 年报而命中的是 2024 块时，模型应能识别年份错位而非照抄。
    """
    reference_lines = []
    for index, hit in enumerate(hits, start=1):
        metadata = hit.get("metadata") or {}
        block_text = _IMAGE_PLACEHOLDER.sub("(图)", hit.get("text") or "")[:BLOCK_TEXT_CHAR_LIMIT]
        page_hint = f"，第 {metadata.get('page')} 页" if metadata.get("page") else ""
        date_hint = f"，文档日期 {metadata['doc_date']}" if metadata.get("doc_date") else ""
        document_hint = metadata.get("title") or metadata.get("source_file") or "未知文档"
        reference_lines.append(f"[{index}]（来源：{document_hint}{page_hint}{date_hint}）\n{block_text}")

    system_prompt = (
        "你是 doc-agent 文档问答助手，基于下方「参考文档块」回答用户问题。规则：\n"
        "1. 只依据提供的参考块作答，严禁编造块外的信息；块内没有答案时明确说明。\n"
        "2. 引用某个块时，在该结论句末尾加角标 [n]，n 必须是该块在参考资料中的编号（1 起）。\n"
        "3. 同一条信息可合并引用多个块，例如 [1][3]。\n"
        "4. 回答使用与问题相同的语言，简明准确。\n\n"
        "参考文档块：\n" + "\n\n".join(reference_lines)
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def _build_source(hit: dict) -> dict:
    """检索块 → 前端 sources 契约字段（chat.html 已按此渲染）。"""
    metadata = hit.get("metadata") or {}
    document_name = metadata.get("title") or metadata.get("source_file") or "未知文档"
    page_number = metadata.get("page")
    snippet = _IMAGE_PLACEHOLDER.sub("(图)", hit.get("text") or "").strip()[:SNIPPET_CHAR_LIMIT]
    image_ids = [image_id for image_id in (metadata.get("image_ids") or "").split(",") if image_id]
    return {
        "block_id": hit.get("id"),  # M4 评估：引用可回查率按 block_id 校验（检索块必回）
        "document_name": document_name,
        "page_number": page_number,
        "doc_date": metadata.get("doc_date"),  # 文档日期（前端溯源展示；无日期文档为 None）
        "snippet": snippet,
        "image_ids": image_ids,  # 引用块关联的图（前端经 /api/images/{id} 渲染）
    }


# ---------------- 图与入口 ----------------

def build_qa_graph():
    """组装问答图（route → greet|search → answer → citations）。"""
    graph = StateGraph(QAState)
    graph.add_node("route", route_node)
    graph.add_node("greet", greet_node)
    graph.add_node("search", search_node)
    graph.add_node("answer", answer_node)
    graph.add_node("citations", citations_node)

    graph.set_entry_point("route")
    graph.add_conditional_edges(
        "route",
        lambda state: "greet" if state.get("needs_search") is False else "search",
        {"greet": "greet", "search": "search"},
    )
    graph.add_edge("greet", END)
    graph.add_edge("search", "answer")
    graph.add_edge("answer", "citations")
    graph.add_edge("citations", END)
    return graph.compile()


_graph_cache = None


def _default_graph():
    global _graph_cache
    if _graph_cache is None:
        _graph_cache = build_qa_graph()
    return _graph_cache


def run_question(question: str, graph=None) -> dict:
    """问答入口（server / CLI 共用）：单问题 → {answer, sources, usage, ...}。

    图内/图外异常都兜底为可读答复，绝不外抛（R7 降级不崩）。
    """
    start = time.perf_counter()
    graph = graph or _default_graph()
    try:
        final_state = graph.invoke({"question": question})
    except Exception as exc:
        return {
            "answer": f"（问答处理异常：{exc}）",
            "sources": [],
            "usage": {},
            "note": f"流水线异常: {exc}",
            "degraded": True,
        }
    return {
        "answer": final_state.get("answer") or "",
        "sources": final_state.get("sources") or [],
        "usage": final_state.get("usage") or {},
        "note": final_state.get("note"),
        "degraded": bool(final_state.get("degraded")),
        "elapsed_s": round(time.perf_counter() - start, 2),
    }
