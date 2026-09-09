"""doc-agent API 服务入口 —— 前端页面托管 + ingest 真实管线 + 问答真实链路

当前状态：M3（2026-09-08）。
- /chat、/upload：两个前端页面（问答 / 文档上传）
- /health：存活检查
- POST /api/v1/ingest：**真实摄取管线**（保存到 inbox → run_document 编排
  探测/解析/质量门/切片/落库 → 逐文档报告）；与 CLI `ingest` 同构
- POST /api/v1/chat：**真实问答链路**（run_question：检索 + 生成 + 引用溯源）；
  响应 final_answer 含 [n] 角标 + sources（document_name/page_number/snippet）
- GET /api/images/{image_id}：图片资源（引用块 [IMAGE] 占位渲染）

ingest 契约（与 upload.html 对齐）：业务失败也返回 HTTP 200 + status=failed，
网络错误才非 200 —— 页面据此区分"文档没解析成功"与"服务连不上"。
"""
from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import config
from ..images import repo as image_repo
from ..ingest.pipeline import run_document
from ..qa import run_question

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="doc-agent", version="0.1.0")

# ---- 异步摄取：上传与处理解耦（2026-09-09）----
# 痛点：原 /api/v1/ingest 在请求线程内同步跑完整管线（保存→解析→切片→入库），
# 大文件（如 220 页年报）会长时间占住请求，前端只能干等「上传中」，且串行上传时
# 一个大文件会阻塞其后所有文件。改为：上传仅落盘 inbox 并即时返回 job_id + ETA，
# 真正的摄取在后台 worker 线程池并发执行，前端轮询状态。
_INGEST_LOCK = threading.Lock()
_INGEST_JOBS: dict[str, dict] = {}
_INGEST_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(os.getenv("DOCAGENT_INGEST_WORKERS", "2")),
    thread_name_prefix="ingest",
)
# ETA 估算：每页约耗时（秒），按实测校准（振华 220 页年报数分钟量级）
_SEC_PER_PAGE = float(os.getenv("DOCAGENT_INGEST_SEC_PER_PAGE", "2.0"))

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---- 页面路由 ----
@app.get("/", include_in_schema=False)
def home_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "chat.html")


@app.get("/chat", include_in_schema=False)
def chat_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "chat.html")


@app.get("/upload", include_in_schema=False)
def upload_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "upload.html")


# ---- 存活检查 ----
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ---- 文档摄取（M1 真实管线，2026-09-09 起为异步）----
@app.post("/api/v1/ingest")
def ingest_upload(file: UploadFile) -> JSONResponse:
    """上传单文档 → inbox 落盘 → **立即返回** job_id + ETA；真正摄取在后台并发执行。

    响应（契约 §upload.html）：
        job_id: 任务号，前端据此轮询 GET /api/v1/ingest/status
        status: "queued"（已接收，排队/处理中）
        pages / eta_seconds：用于即时给出「预计多久后可查询」
        message：给用户的 ETA 提示文案
    业务失败（解析/入库出错）不在此返回，而是落在轮询结果里（status=failed）。
    仅落盘失败才在此直接返回 status=failed。
    """
    filename = Path(file.filename or "unknown").name  # 只取文件名，防路径穿越
    config.INBOX_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = config.INBOX_DIR / filename

    try:
        with saved_path.open("wb") as target:
            shutil.copyfileobj(file.file, target)  # 同文件重传 = 同名覆盖，内容幂等
    except Exception as exc:
        return JSONResponse({"filename": filename, "status": "failed", "error": f"保存失败: {exc}"})

    pages, eta = _estimate_pages_and_eta(saved_path)
    job_id = "ing_" + uuid.uuid4().hex[:12]
    with _INGEST_LOCK:
        # 轻量回收：注册表过大时丢掉最旧的已完成任务，避免无限增长
        if len(_INGEST_JOBS) > 200:
            oldest = sorted(
                (j for j in _INGEST_JOBS.values() if j["status"] in ("done", "failed")),
                key=lambda j: j["finished_at"] or 0,
            )
            for j in oldest[: len(_INGEST_JOBS) - 200]:
                _INGEST_JOBS.pop(j["job_id"], None)
        _INGEST_JOBS[job_id] = {
            "job_id": job_id,
            "filename": filename,
            "pages": pages,
            "eta_seconds": eta,
            "status": "queued",
            "enqueued_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }
    _INGEST_EXECUTOR.submit(_process_job, job_id, str(saved_path))

    return JSONResponse(
        {
            "job_id": job_id,
            "filename": filename,
            "status": "queued",
            "pages": pages,
            "eta_seconds": eta,
            "message": f"已接收，后台处理中，预计约 {_human_eta(eta)} 后可查询",
        }
    )


@app.get("/api/v1/ingest/status")
def ingest_status(job_id: str) -> JSONResponse:
    """轮询摄取任务状态。前端据此把「处理中」翻成「已完成/失败」并展示结果。"""
    with _INGEST_LOCK:
        job = _INGEST_JOBS.get(job_id)
        if not job:
            return JSONResponse({"job_id": job_id, "status": "not_found"}, status_code=404)
        snapshot = dict(job)
    # 进度提示：基于 ETA 给出「已等待 / 预计剩余」
    now = time.time()
    enqueued = snapshot.get("enqueued_at") or now
    waited = int(now - enqueued)
    remaining = max(0, (snapshot.get("eta_seconds") or 0) - waited)
    snapshot["waited_s"] = waited
    snapshot["remaining_s"] = remaining
    return JSONResponse(snapshot)


@app.get("/api/v1/ingest/jobs")
def ingest_jobs() -> JSONResponse:
    """返回全部摄取任务的精简快照（按提交时间倒序），供前端「最近上传记录」面板展示与回查。

    用户关掉页面后任务仍在服务端后台执行；再次打开本页即可看到历史任务及其进度/结果，
    无需一直挂在页面。任务注册表为内存态，服务重启会清空（已知边界，不持久化）。
    """
    with _INGEST_LOCK:
        ordered = sorted(
            _INGEST_JOBS.values(),
            key=lambda j: j.get("enqueued_at") or 0,
            reverse=True,
        )
        snapshots = [
            {
                "job_id": j["job_id"],
                "filename": j["filename"],
                "pages": j.get("pages"),
                "eta_seconds": j.get("eta_seconds"),
                "status": j["status"],
                "enqueued_at": j.get("enqueued_at"),
                "finished_at": j.get("finished_at"),
                "result": j.get("result"),
                "error": j.get("error"),
            }
            for j in ordered
        ]
    return JSONResponse({"jobs": snapshots, "count": len(snapshots)})


def _process_job(job_id: str, saved_path: str) -> None:
    """后台 worker：跑完整摄取管线，把结果写回 job 注册表。失败永不外抛。"""
    with _INGEST_LOCK:
        job = _INGEST_JOBS.get(job_id)
        if job:
            job["status"] = "processing"
            job["started_at"] = time.time()
    try:
        report = run_document(saved_path)  # 图内已隔离失败，永不外抛
        result = {
            "status": "ok" if report.get("status") != "failed" else "failed",
            "pages_parsed": None,
            "chunks_created": report.get("chunks"),
            "stored": report.get("stored"),
            "skipped": report.get("skipped"),
            "format": report.get("format"),
            "provider": report.get("provider"),
            "degraded": report.get("degraded"),
            "error": report.get("error"),
            "elapsed_s": report.get("elapsed_s"),
        }
        quality = report.get("quality")
        if quality:
            result["pages_parsed"] = quality["red"] + quality["yellow"] + quality["green"]
        with _INGEST_LOCK:
            job = _INGEST_JOBS.get(job_id)
            if job:
                job["result"] = result
                job["status"] = result["status"]
                job["finished_at"] = time.time()
    except Exception as exc:
        with _INGEST_LOCK:
            job = _INGEST_JOBS.get(job_id)
            if job:
                job["status"] = "failed"
                job["error"] = str(exc)
                job["finished_at"] = time.time()


def _estimate_pages_and_eta(file_path: Path) -> tuple[int | None, int]:
    """快速估算页数 + 入库耗时（秒）。仅做即时 ETA，不阻塞上传响应。"""
    pages = None
    try:
        import pymupdf

        with pymupdf.open(str(file_path)) as doc:
            pages = doc.page_count
    except Exception:
        pages = None
    if pages:
        eta = int(pages * _SEC_PER_PAGE)
    else:
        # 非 PDF 或无页数：按文件大小粗估（每 MB ≈ 20s）
        mb = (file_path.stat().st_size or 0) / (1024 * 1024)
        eta = max(20, int(mb * 20))
    return pages, max(15, eta)


def _human_eta(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds} 秒"
    return f"{round(seconds / 60)} 分钟"


# ---- 问答（M3 真实链路：检索 + 生成 + 引用溯源）----
@app.post("/api/v1/chat")
def chat_answer(payload: dict) -> dict:
    started_at = time.time()
    user_message = (payload.get("message") or "").strip()
    thread_id = payload.get("thread_id") or _new_thread_id()
    if not user_message:
        return {
            "thread_id": thread_id,
            "final_answer": "（空问题）请告诉我你想查什么。",
            "sources": [],
            "duration_s": round(time.time() - started_at, 2),
        }

    outcome = run_question(user_message)
    return {
        "thread_id": thread_id,
        "final_answer": outcome["answer"],
        "sources": outcome["sources"],
        "degraded": outcome["degraded"],
        "usage": outcome["usage"],
        "note": outcome.get("note"),
        "duration_s": round(time.time() - started_at, 2),
    }


# ---- 图片资源（M3 展示层：引用块内的 [IMAGE:xxx] 由前端按此地址渲染）----
@app.get("/api/images/{image_id}", response_model=None)
def serve_image(image_id: str):
    image_path = image_repo.resolve_file_path(image_id)
    if image_path is None:
        return JSONResponse({"error": f"图片不存在: {image_id}"}, status_code=404)
    return FileResponse(str(image_path))


def _new_thread_id() -> str:
    """服务端会话号：与前端 localStorage 的 thread_id 同一格式约定。"""
    return "thread_" + uuid.uuid4().hex[:12]
