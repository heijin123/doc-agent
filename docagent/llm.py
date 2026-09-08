"""Chat 完成封装（M3 问答层用）：OpenAI 兼容 + usage 返回 + 显式超时 + 指数退避。

与 vlm_transcribe / embedding 同一 DashScope 端点与 Key，同一套客户端纪律：
- httpx 显式超时（2026-09-08 实证教训：不给 timeout 偶发无响应会永久挂起）
- 网络抖动/429/5xx 指数退避重试；4xx 参数问题直接抛不浪费重试
- 无 DASHSCOPE_API_KEY → 返回 None（调用方降级，不抛中断链路）

对外：
    chat_completion(messages, temperature=0.3, max_tokens=2000)
        -> {"text": str, "usage": {"prompt_tokens", "completion_tokens"}} | None
"""
from __future__ import annotations

import time

import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    OpenAI,
    RateLimitError,
)

from . import config

CHAT_MAX_ATTEMPTS = 3
CHAT_RETRY_BACKOFF_SECONDS = 1.0
CLIENT_TIMEOUT = httpx.Timeout(connect=15.0, read=120.0, write=60.0, pool=10.0)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL,
            timeout=CLIENT_TIMEOUT,
        )
    return _client


def chat_completion(
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 2000,
) -> dict | None:
    """单次 chat 补全：成功返回 {text, usage}；无 Key / 重试耗尽返回 None。"""
    if not config.DASHSCOPE_API_KEY:
        return None

    last_error: Exception | None = None
    for attempt in range(CHAT_MAX_ATTEMPTS):
        try:
            completion = _get_client().chat.completions.create(
                model=config.QWEN_CHAT_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            usage = completion.usage
            usage_dict = {}
            if usage is not None:
                usage_dict = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                }
            return {"text": completion.choices[0].message.content or "", "usage": usage_dict}
        except (APIConnectionError, APITimeoutError, RateLimitError, APIStatusError) as exc:
            last_error = exc
            if isinstance(exc, APIStatusError) and not isinstance(exc, RateLimitError):
                if exc.status_code < 500:
                    return None  # 4xx（除 429）：参数/配额问题，重试无意义
            if attempt < CHAT_MAX_ATTEMPTS - 1:
                time.sleep(CHAT_RETRY_BACKOFF_SECONDS * (2**attempt))
    return None  # 重试耗尽（调用方降级提示）
