"""M1 摄取流水线编排：LangGraph 状态机版（探测 → 解析判级 → 切片 → 落库）。

为什么用 LangGraph（D4，2026-09-07 拍板）：
    摄取是确定性数据流水线，没有模型自主决策循环——上编排不是要"智能"，
    而是借状态机外壳组织六步：节点 = 步骤、条件边 = 失败/分支路由、
    state = 逐步累积的每文档报告。与消费侧问答图同一框架两种用法，
    是本 demo 的核心叙事（需求 §1.1 / §4.2）。
    与 8 条守则的关系：内层切块器 / 质量门 / 落库全是纯函数（可独立测试、
    不感知 graph），编排只是薄外壳——M2 在此挂"质量门红页 → VLM 转录"
    的条件边，内层仍不动。

M1 节点（每文档一条流水线，批次层 for 串行调用，Send 并行留 M2+）：
    detect → parse → gate → chunk → store
    - detect：扩展名 + 内容嗅探识别格式；不支持 → FAIL 出口
    - parse ：PDF 一次打开完成逐页取文本 + 质量门判级（assess_pdf）；非 PDF 空转
    - gate  ：M1 只把判级结果整理进报告（红页记录在案，VLM 补全是 M2 锚点）
    - chunk ：复用 chunk_document（分格式定制切片，纯函数层零改动）
    - store ：查库 diff（md5）→ 命中跳过 / 变化块才 embedding + upsert（幂等）

每个节点后的条件边：节点内异常一律转为 error 字段 + status=failed，
图优雅收尾返回报告而非抛异常 —— 单文档失败不拖垮整批（demo1 教训）。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from ..vectorstore import store as vector_store
from .chunk_models import ChunkingResult
from .chunking import chunk_document
from .file_type_detection import UnsupportedFileError, detect_file_type
from .pdf_quality import assess_pdf


class IngestState(TypedDict, total=False):
    """单文档摄取流水线的运行态（节点逐步累积，图结束时即报告素材）。

    刻意不加 reducer：每节点读回当前值合并后整体返回，state 变化一目了然。
    """

    file_path: str
    doc_id: str                     # 库内 id 前缀（= 文件名 stem，幂等主键）
    document_type: str | None       # detect 结果（格式）
    detected_by: str | None         # 探测依据（扩展名 / 嗅探）
    quality_raw: dict | None        # parse 阶段质量门原始输出（PDF）
    quality: dict | None            # gate 整理后的报告友好结构（非 PDF 为 None）
    chunking: ChunkingResult | None  # 切片产物（store 消费后即弃）
    embedding: dict | None          # store 返回的 provider / degraded
    status: str                     # ok / skipped / failed
    error: str | None               # failed 时的原因（进报告）
    stored: int                     # 本次实际 upsert 块数
    skipped: int                    # md5 命中跳过块数
    timings: dict[str, float]       # 每节点耗时（秒），报告 / 延迟优化的依据


# ---------------- 节点 ----------------

def _timed(state: IngestState, key: str, elapsed: float) -> dict:
    timings = dict(state.get("timings") or {})
    timings[key] = round(elapsed, 3)
    return timings


def detect_node(state: IngestState) -> dict:
    """识别格式；不支持/无法识别 → failed（报告带补救提示，批次不中断）。"""
    start = time.perf_counter()
    try:
        detection = detect_file_type(Path(state["file_path"]))
        return {
            "document_type": detection.document_type.value,
            "detected_by": detection.detected_by,
            "timings": _timed(state, "detect", time.perf_counter() - start),
        }
    except (UnsupportedFileError, ValueError) as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "timings": _timed(state, "detect", time.perf_counter() - start),
        }


def parse_node(state: IngestState) -> dict:
    """PDF：一次打开完成逐页文本提取 + 质量门判级；非 PDF 无页概念，空转。"""
    if state.get("document_type") != "pdf":
        return {}
    start = time.perf_counter()
    try:
        quality_raw = assess_pdf(Path(state["file_path"]))
        return {
            "quality_raw": quality_raw,
            "timings": _timed(state, "parse", time.perf_counter() - start),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"PDF 解析失败: {exc}",
            "timings": _timed(state, "parse", time.perf_counter() - start),
        }


def gate_node(state: IngestState) -> dict:
    """质量门：M1 把红黄绿统计与红页原因整理进报告。

    M2 锚点：这里读 state.quality 后走条件边"红页数 > 0 → VLM 转录节点"，
    再回接 chunk；M1 无慢路径，红页文本仍照常入块（不越权修补）。
    """
    quality_raw = state.get("quality_raw")
    if not quality_raw:
        return {}
    stats = quality_raw["stats"]
    return {
        "quality": {
            "red": stats["红"],
            "yellow": stats["黄"],
            "green": stats["绿"],
            "red_pages": quality_raw["red_pages"],
            "notes": quality_raw["notes"],
        }
    }


def chunk_node(state: IngestState) -> dict:
    """按格式定制切片（复用 chunk_document，纯函数层零改动）。"""
    start = time.perf_counter()
    try:
        result = chunk_document(Path(state["file_path"]))
        return {
            "chunking": result,
            "document_type": result.document_type,
            "timings": _timed(state, "chunk", time.perf_counter() - start),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"切片失败: {exc}",
            "timings": _timed(state, "chunk", time.perf_counter() - start),
        }


def store_node(state: IngestState) -> dict:
    """落库：md5 幂等 diff → 仅变化块 embedding + upsert（详见 vectorstore.store）。"""
    chunking = state.get("chunking")
    if not chunking or not chunking.chunks:
        return {"status": "skipped", "stored": 0, "skipped": 0}

    start = time.perf_counter()
    try:
        outcome = vector_store.upsert_doc_chunks(chunking.chunks, state["doc_id"])
        # 全 skip = 文档级 skipped；有新写入 = ok
        doc_status = "skipped" if outcome["stored"] == 0 and outcome["skipped"] > 0 else "ok"
        return {
            "status": doc_status,
            "stored": outcome["stored"],
            "skipped": outcome["skipped"],
            "embedding": {"provider": outcome["provider"], "degraded": outcome["degraded"]},
            "timings": _timed(state, "store", time.perf_counter() - start),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"落库失败: {exc}",
            "timings": _timed(state, "store", time.perf_counter() - start),
        }


# ---------------- 条件路由 ----------------

def _is_failed(state: IngestState) -> bool:
    return state.get("status") == "failed"


# ---------------- 图 ----------------

def build_ingest_graph():
    """组装单文档摄取流水线图（每节点后：失败 → FAIL 出口，成功 → 下一节点）。"""
    graph = StateGraph(IngestState)
    graph.add_node("detect", detect_node)
    graph.add_node("parse", parse_node)
    graph.add_node("gate", gate_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("store", store_node)

    graph.set_entry_point("detect")
    for node_name, next_node in [
        ("detect", "parse"),
        ("parse", "gate"),
        ("gate", "chunk"),
        ("chunk", "store"),
    ]:
        graph.add_conditional_edges(
            node_name,
            lambda state: "fail" if _is_failed(state) else "ok",
            {"ok": next_node, "fail": END},
        )
    graph.add_edge("store", END)
    return graph.compile()


_graph_cache = None


def _default_graph():
    global _graph_cache
    if _graph_cache is None:
        _graph_cache = build_ingest_graph()
    return _graph_cache


def run_document(file_path, graph=None) -> dict:
    """摄取单个文档，返回逐文档报告（图内/图外异常都不外抛，批次可 for 直跑）。

    报告字段即 M1 雏形，M2 扩展（VLM 块数 / token 成本等）在此追加。
    """
    path = Path(file_path)
    start = time.perf_counter()
    graph = graph or _default_graph()

    try:
        final = graph.invoke({"file_path": str(path), "doc_id": path.stem})
    except Exception as exc:
        return {
            "source_file": path.name,
            "doc_id": path.stem,
            "status": "failed",
            "error": f"流水线异常: {exc}",
            "chunks": 0,
            "stored": 0,
            "skipped": 0,
            "elapsed_s": round(time.perf_counter() - start, 2),
        }

    chunking = final.get("chunking")
    embedding = final.get("embedding") or {}
    return {
        "source_file": path.name,
        "doc_id": path.stem,
        "status": final.get("status", "ok"),
        "format": final.get("document_type"),
        "detected_by": final.get("detected_by"),
        "quality": final.get("quality"),
        "chunks": len(chunking.chunks) if chunking else 0,
        "stored": final.get("stored", 0),
        "skipped": final.get("skipped", 0),
        "provider": embedding.get("provider"),
        "degraded": embedding.get("degraded", False),
        "error": final.get("error"),
        "timings": final.get("timings", {}),
        "elapsed_s": round(time.perf_counter() - start, 2),
    }
