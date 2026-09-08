"""VLM 转录（M2 慢路径兜底）：质量门红页 → 整页渲染 → qwen-vl 转录页面文字。

为什么整页渲染转录（2026-09-08 设计）：
- 红页两类成因都支持整页：坏字体页（文字层乱码但视觉可读）转录整页即还原；
  扫描/纯图页整页即图。页级粒度与切片"一页一块"天然对齐。
- demo1 的 figure 区域裁剪（vlm_describe_figure）是 fig1 专用方案，
  通用场景需要版面分析找图边界，成本高且只覆盖一种成因——不做。

调用约定（编排 vlm 节点消费）：
    transcribe_pages(pdf_path, page_numbers, max_pages) ->
        (transcripts: dict[int, str], usage: dict, failed_pages: list[int])
- 一次打开 PDF 渲染多页，逐页调模型（避免每页重开文档）
- 无 DASHSCOPE_API_KEY / 调用失败 → 返回空结果（该页不进 transcripts，
  编排降级为 M1 行为：红页原文照常入块），绝不抛异常中断批次
- max_pages 截断：超出部分不进本轮转录（成本护栏），由编排报告"未转录 N 页"

usage 汇总 {prompt_tokens, completion_tokens}——VLM 图像按图计 token，
进摄取报告的成本字段（§6 非功能：LLM 只在 VLM 转录处用，成本可见）。
"""
from __future__ import annotations

import base64
import time
from pathlib import Path

import httpx
import pymupdf
from openai import (
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    OpenAI,
    RateLimitError,
)

from .. import config

# 转录退避重试（与 embedding 同款）：每模型最多 2 次尝试，退避 1s/2s
VLM_MAX_ATTEMPTS = 2
VLM_RETRY_BACKOFF_SECONDS = 1.0

# 客户端超时（2026-09-08 实证教训）：不给 timeout 时大图请求可能永久挂起——
# read 180s 覆盖 qwen-vl 慢生成（实测单页转录 ~43s），connect 15s 快速失败
CLIENT_TIMEOUT = httpx.Timeout(connect=15.0, read=180.0, write=120.0, pool=10.0)

TRANSCRIBE_PROMPT = (
    "你是文档转录器。请完整转录这张页面图像中的全部文字内容（包括图内文字、表格文字），"
    "按从上到下、从左到右的阅读顺序组织，保留原文语言，输出 markdown 结构（标题用 # 标记）。"
    "只输出转录内容本身，不要解释、不要总结、不要省略任何文字。"
)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """OpenAI 兼容客户端（与 embedding 共用 DashScope 端点与 Key）。

    显式超时必须有：见 CLIENT_TIMEOUT 注释（大图请求无超时会永久挂起）。
    """
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL,
            timeout=CLIENT_TIMEOUT,
        )
    return _client


def transcribe_pages(
    pdf_path: Path,
    page_numbers: list[int],
    max_pages: int | None = None,
) -> tuple[dict[int, str], dict, list[int]]:
    """转录指定页：返回 (页号→文本, usage 汇总, 失败页列表)。无 Key/失败不抛。"""
    if not config.DASHSCOPE_API_KEY or not page_numbers:
        return {}, {"prompt_tokens": 0, "completion_tokens": 0}, []

    limited_pages = page_numbers[:max_pages] if max_pages else page_numbers
    transcripts: dict[int, str] = {}
    failed_pages: list[int] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}

    with pymupdf.open(str(pdf_path)) as document:
        for page_number in limited_pages:
            page = document.load_page(page_number - 1)  # 页号 1 起始 → pymupdf 0 起始
            try:
                text, usage = _transcribe_one_page(page)
            except Exception:
                failed_pages.append(page_number)  # 单页失败不阻断后续页
                continue
            if text and text.strip():
                transcripts[page_number] = text
            else:
                failed_pages.append(page_number)
            if usage:
                usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
                usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
    return transcripts, usage_total, failed_pages


def _transcribe_one_page(page) -> tuple[str | None, dict | None]:
    """渲染单页 PNG → qwen-vl 转录；主模型失败降级备用模型；均失败抛异常。"""
    image_data_uri = _page_to_data_uri(page)

    last_error: Exception | None = None
    for model in (config.QWEN_VL_MODEL, config.VLM_FALLBACK_MODEL):
        try:
            return _chat_with_retry(model, image_data_uri)
        except Exception as exc:  # 单个模型整体失败 → 换备用模型
            last_error = exc
            continue
    raise RuntimeError(f"VLM 转录失败: {last_error}")


def _chat_with_retry(model: str, image_data_uri: str) -> tuple[str, dict]:
    """单模型转录调用，指数退避重试（连接错误/超时/429/5xx；4xx 参数问题直接抛）。"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_uri}},
                {"type": "text", "text": TRANSCRIBE_PROMPT},
            ],
        }
    ]
    last_error: Exception | None = None
    for attempt in range(VLM_MAX_ATTEMPTS):
        try:
            completion = _get_client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=8192,
            )
            content = completion.choices[0].message.content or ""
            usage = completion.usage
            usage_dict = {}
            if usage is not None:
                usage_dict = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                }
            return content, usage_dict
        except (APIConnectionError, APITimeoutError, RateLimitError, APIStatusError) as exc:
            last_error = exc
            if isinstance(exc, APIStatusError) and not isinstance(exc, RateLimitError):
                if exc.status_code < 500:
                    raise  # 4xx（除 429）：参数/配额问题，重试无意义
            if attempt < VLM_MAX_ATTEMPTS - 1:
                time.sleep(VLM_RETRY_BACKOFF_SECONDS * (2**attempt))
    raise last_error  # type: ignore[misc]


def _page_to_data_uri(page) -> str:
    """整页渲染 PNG → base64 data URI（zoom 提高分辨率保证小字可读）。"""
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(config.VLM_PAGE_ZOOM, config.VLM_PAGE_ZOOM))
    png_bytes = pixmap.tobytes("png")
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"
