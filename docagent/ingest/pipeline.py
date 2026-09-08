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
    detect → parse → gate → [红页 → vlm] → chunk → store → save_images
    - detect：扩展名 + 内容嗅探识别格式；不支持 → FAIL 出口
    - parse ：PDF 一次打开完成逐页取文本 + 质量门判级（assess_pdf）；非 PDF 空转
    - gate  ：把红黄绿统计整理进报告（红页号记录在案，驱动 vlm 条件路由）
    - vlm（M2，2026-09-08）：红页整页渲染 → qwen-vl 转录，产物 {页号: 文本}
      进 state，chunk 消费时替换该页原文（同块位 id 不变、md5 变 → upsert 覆盖
      存量乱码块，自愈无需删除）；无 Key/失败 → 降级 note，红页保持原文入库
    - chunk ：复用 chunk_document（分格式定制切片 + 图片占位提取 + 红页转录注入）
    - store ：查库 diff（md5）→ 命中跳过 / 变化块才 embedding + upsert（幂等）
    - save_images（2026-09-08）：文档入库成功后，把切块提取的图片落盘+登记
      sqlite（幂等）；图片是附属展示资源——落盘失败只记 note，不判文档 failed

每个节点后的条件边：节点内异常一律转为 error 字段 + status=failed，
图优雅收尾返回报告而非抛异常 —— 单文档失败不拖垮整批（demo1 教训）。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .. import config
from ..images import repo as image_repo
from ..vectorstore import store as vector_store
from . import vlm_transcribe
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
    images_saved: int               # 本次实际落盘图片数（save_images）
    images_skipped: int             # 已存在跳过数（幂等重跑）
    image_note: str | None          # 图片落盘失败等附属告警（不判 failed）
    vlm_transcripts: dict | None    # vlm 产物：{页号: 转录文本}（chunk 消费）
    vlm_pages: int                  # 成功转录页数
    vlm_failed: list | None         # 转录失败页（保留原文）
    vlm_note: str | None            # 无 Key / 超上限 / 失败 说明
    vlm_usage: dict | None          # token 成本 {prompt_tokens, completion_tokens}
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
    """质量门：把红黄绿统计与红页原因整理进报告（红页号驱动 vlm 条件路由）。"""
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


def vlm_node(state: IngestState) -> dict:
    """慢路径兜底（M2）：质量门红页 → 整页渲染 → qwen-vl 转录。

    无 Key / 转录失败 → 降级：红页保持原文入块（M1 行为），只记 note 不判 failed。
    转录页数受 VLM_MAX_PAGES_PER_DOC 护栏（成本保护），超出部分报告标注。
    """
    quality_raw = state.get("quality_raw") or {}
    red_pages = list(quality_raw.get("red_pages") or [])
    if not red_pages:
        return {}

    start = time.perf_counter()
    # 降级一致性：未配 Key 或显式 mock provider 环境（verify/无预算）都不调 VLM
    if not config.DASHSCOPE_API_KEY or config.EMBEDDING_PROVIDER == "mock":
        reason = "未配 DASHSCOPE_API_KEY" if not config.DASHSCOPE_API_KEY else "mock provider 环境"
        return {
            "vlm_pages": 0,
            "vlm_note": f"{reason}：{len(red_pages)} 个红页未转录（保持原文入库）",
            "timings": _timed(state, "vlm", time.perf_counter() - start),
        }

    try:
        transcripts, usage_total, failed_pages = vlm_transcribe.transcribe_pages(
            Path(state["file_path"]),
            red_pages,
            max_pages=config.VLM_MAX_PAGES_PER_DOC,
        )
        note_parts: list[str] = []
        if failed_pages:
            note_parts.append(f"{len(failed_pages)} 页转录失败保留原文: P{failed_pages[:5]}")
        over_limit = len(red_pages) - config.VLM_MAX_PAGES_PER_DOC
        if over_limit > 0:
            note_parts.append(f"超过单文档转录上限，{over_limit} 页未转录")
        return {
            "vlm_transcripts": transcripts,
            "vlm_pages": len(transcripts),
            "vlm_failed": failed_pages,
            "vlm_usage": usage_total,
            "vlm_note": "；".join(note_parts) if note_parts else None,
            "timings": _timed(state, "vlm", time.perf_counter() - start),
        }
    except Exception as exc:
        return {
            "vlm_pages": 0,
            "vlm_note": f"VLM 转录异常: {exc}",
            "timings": _timed(state, "vlm", time.perf_counter() - start),
        }


def chunk_node(state: IngestState) -> dict:
    """按格式定制切片（复用 chunk_document；红页转录文本经 vlm_transcripts 注入）。"""
    start = time.perf_counter()
    try:
        result = chunk_document(
            Path(state["file_path"]),
            vlm_transcripts=state.get("vlm_transcripts"),
        )
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


# ---------------- 条件路由与图片落盘 ----------------

def _is_failed(state: IngestState) -> bool:
    return state.get("status") == "failed"


def save_images_node(state: IngestState) -> dict:
    """把切块提取的图片落盘 + sqlite 登记（幂等）。

    图片是块的附属展示资源：落盘失败只记 note 进报告，不判文档 failed——
    文档主链路（向量入库）不受图资源问题拖累。
    """
    chunking = state.get("chunking")
    if not chunking or not chunking.images:
        return {"images_saved": 0, "images_skipped": 0}

    start = time.perf_counter()
    try:
        outcome = image_repo.save_images(state["doc_id"], chunking.images)
        return {
            "images_saved": outcome["saved"],
            "images_skipped": outcome["skipped"],
            "timings": _timed(state, "save_images", time.perf_counter() - start),
        }
    except Exception as exc:
        return {
            "image_note": f"图片落盘失败: {exc}",
            "timings": _timed(state, "save_images", time.perf_counter() - start),
        }


def _route_after_gate(state: IngestState) -> str:
    """gate 后三分支：failed → END；PDF 且含红页 → vlm 慢路径；否则 → chunk。"""
    if _is_failed(state):
        return "fail"
    quality_raw = state.get("quality_raw") or {}
    is_pdf_with_red_pages = (
        state.get("document_type") == "pdf" and bool(quality_raw.get("red_pages"))
    )
    return "vlm" if is_pdf_with_red_pages else "chunk"


# ---------------- 图 ----------------

def build_ingest_graph():
    """组装单文档摄取流水线图（每节点后：失败 → FAIL 出口，成功 → 下一节点）。"""
    graph = StateGraph(IngestState)
    graph.add_node("detect", detect_node)
    graph.add_node("parse", parse_node)
    graph.add_node("gate", gate_node)
    graph.add_node("vlm", vlm_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("store", store_node)
    graph.add_node("save_images", save_images_node)

    graph.set_entry_point("detect")
    for node_name, next_node in [
        ("detect", "parse"),
        ("parse", "gate"),
    ]:
        graph.add_conditional_edges(
            node_name,
            lambda state: "fail" if _is_failed(state) else "ok",
            {"ok": next_node, "fail": END},
        )
    graph.add_conditional_edges("gate", _route_after_gate, {"vlm": "vlm", "chunk": "chunk", "fail": END})
    graph.add_edge("vlm", "chunk")  # vlm 内部已降级兜底，不外抛，无需失败边
    graph.add_conditional_edges(
        "chunk",
        lambda state: "fail" if _is_failed(state) else "ok",
        {"ok": "store", "fail": END},
    )
    # store 成功（ok/skipped）才落图；store 失败直接收尾（图留待下次重跑补）
    graph.add_conditional_edges(
        "store",
        lambda state: "fail" if _is_failed(state) else "ok",
        {"ok": "save_images", "fail": END},
    )
    graph.add_edge("save_images", END)
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
        "images": len(chunking.images) if chunking else 0,  # 切块提取到的图数
        "images_saved": final.get("images_saved", 0),
        "images_skipped": final.get("images_skipped", 0),
        "image_note": final.get("image_note"),
        "vlm_pages": final.get("vlm_pages", 0),
        "vlm_failed": final.get("vlm_failed") or [],
        "vlm_note": final.get("vlm_note"),
        "vlm_usage": final.get("vlm_usage") or {},
        "provider": embedding.get("provider"),
        "degraded": embedding.get("degraded", False),
        "error": final.get("error"),
        "timings": final.get("timings", {}),
        "elapsed_s": round(time.perf_counter() - start, 2),
    }
