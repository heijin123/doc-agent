"""doc-agent API 服务入口 —— 前端页面托管 + ingest 真实管线 + chat 占位

当前状态：M1 收尾（2026-09-07）。
- /chat、/upload：两个前端页面（问答 / 文档上传）
- /health：存活检查
- POST /api/v1/ingest：**真实摄取管线**（保存到 inbox → run_document 编排
  探测/解析/质量门/切片/落库 → 逐文档报告）；与 CLI `ingest` 同构
- POST /api/v1/chat：演示占位（M3 接入问答编排 + 检索后替换实现）

ingest 契约（与 upload.html 对齐）：业务失败也返回 HTTP 200 + status=failed，
网络错误才非 200 —— 页面据此区分"文档没解析成功"与"服务连不上"。
"""
from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import config
from ..ingest.pipeline import run_document

CHAT_DEMO_MODE = True  # M3 接入真实问答编排后置 False

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="doc-agent", version="0.1.0")

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
    return {"status": "ok", "chat_demo_mode": CHAT_DEMO_MODE}


# ---- 文档摄取（M1 真实管线）----
@app.post("/api/v1/ingest")
def ingest_upload(file: UploadFile) -> JSONResponse:
    """上传单文档 → inbox 落盘 → 摄取流水线（探测/质量门/切片/幂等落库）。

    响应字段（契约 §upload.html）：
        status: "ok" | "failed"   —— skipped（幂等命中）也算 ok，附 stored=0/skipped=N
        pages_parsed / chunks_created：PDF 报页数（质量门红黄绿之和），非 PDF 无页概念报 null
    """
    started_at = time.time()
    filename = Path(file.filename or "unknown").name  # 只取文件名，防路径穿越
    config.INBOX_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = config.INBOX_DIR / filename

    try:
        with saved_path.open("wb") as target:
            shutil.copyfileobj(file.file, target)  # 同文件重传 = 同名覆盖，内容幂等
    except Exception as exc:
        return JSONResponse({"filename": filename, "status": "failed", "error": f"保存失败: {exc}"})

    report = run_document(saved_path)  # 图内已隔离失败，永不外抛

    quality = report.get("quality")
    pages_parsed = None
    if quality:
        pages_parsed = quality["red"] + quality["yellow"] + quality["green"]

    return JSONResponse(
        {
            "filename": filename,
            "status": "ok" if report["status"] != "failed" else "failed",
            "pages_parsed": pages_parsed,
            "chunks_created": report["chunks"],
            "stored": report["stored"],
            "skipped": report["skipped"],
            "format": report.get("format"),
            "provider": report.get("provider"),
            "degraded": report.get("degraded"),
            "error": report.get("error"),
            "elapsed_s": round(time.time() - started_at, 2),
        }
    )


# ---- 问答：演示占位（M3 接入真实编排 + 检索链路后替换）----
@app.post("/api/v1/chat")
def chat_placeholder(payload: dict) -> dict:
    started_at = time.time()
    user_message = (payload.get("message") or "").strip()
    thread_id = payload.get("thread_id") or _new_thread_id()

    if CHAT_DEMO_MODE:
        final_answer = (
            f"（演示占位答复）我已收到你的问题：「{user_message[:60]}」。\n"
            "问答链路将在 M3 接入：知识库检索 → 引用溯源 → 编排答复。当前页面骨架可正常交互。"
        )
        sources: list[dict] = []
    else:
        # 真实实现入口：编排 graph 调用 + 检索结果组装 sources
        final_answer = ""
        sources = []

    return {
        "thread_id": thread_id,
        "final_answer": final_answer,
        "sources": sources,
        "duration_s": round(time.time() - started_at, 2),
    }


def _new_thread_id() -> str:
    """服务端会话号：与前端 localStorage 的 thread_id 同一格式约定。"""
    return "thread_" + uuid.uuid4().hex[:12]
