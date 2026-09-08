"""Embedding 提供方：dashscope 真实向量 / mock 本地哈希向量（无 Key 降级）。

对外两个函数（摄取 store 与后续检索共用）：
    info()                    -> {"provider": "dashscope" | "mock", "degraded": bool}
    embed_texts(texts, batch_size=10) -> list[list[float]]

dashscope 走 OpenAI 兼容接口，分批 ≤10 条（demo1 实证单次超 20 报 400）；
未配 DASHSCOPE_API_KEY 时自动降级 mock（bigram 计数哈希到固定维度）——
摄取全链路仍可跑通，报告标注 provider=mock + degraded=True（R7 降级不崩）。

真实链路注意（2026-09-07 实证）：大批量连续 embedding（如 800+ 块年报）
会触发服务端断连/限流——每批做 3 次指数退避重试（网络抖动/429/5xx 均覆盖）。
demo1 教训重演：mock 全绿 ≠ 真实可跑，网络层问题只有真 Key 长跑才暴露。

注意：dashscope(1024 维) 与 mock(256 维) 维度不同，同一 Chroma 库内混用会
报维度错误 → 切换 provider 需清空 data/chroma_db 重灌（config 已注明）。
"""
from __future__ import annotations

import hashlib
import threading
import time

import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    OpenAI,
    RateLimitError,
)

from .. import config

_client: OpenAI | None = None
_client_lock = threading.Lock()
_resolved: dict | None = None

# embedding 重试：每批最多 3 次尝试，退避 1s/2s/4s（覆盖连接抖动与限流 429）
EMBED_MAX_ATTEMPTS = 3
EMBED_RETRY_BACKOFF_SECONDS = 1.0

# 客户端超时（2026-09-08 从 VLM 侧同款教训补）：不给 timeout 偶发无响应会永久挂起
CLIENT_TIMEOUT = httpx.Timeout(connect=15.0, read=120.0, write=60.0, pool=10.0)


def _resolve() -> dict:
    """决定 provider 与 degraded 标记（模块级缓存，进程内只判一次）。"""
    global _resolved
    if _resolved is None:
        with _client_lock:
            if _resolved is None:
                name = config.EMBEDDING_PROVIDER
                if name == "dashscope" and not config.DASHSCOPE_API_KEY:
                    name = "mock"  # 配了真 provider 但没 Key → 兜底降级
                _resolved = {
                    "provider": name,
                    "degraded": name == "mock",  # mock 即非语义向量，一律标降级
                }
    return _resolved


def info() -> dict:
    """当前提供方信息（摄取报告逐文档标注用）。"""
    return dict(_resolve())


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = OpenAI(
                    api_key=config.DASHSCOPE_API_KEY,
                    base_url=config.DASHSCOPE_BASE_URL,
                    timeout=CLIENT_TIMEOUT,  # 见常量注释：防永久挂起
                )
    return _client


def _embed_batch_with_retry(batch: list[str]) -> list[list[float]]:
    """单批 embedding，失败指数退避重试（Connection error / 429 / 5xx）。"""
    last_error: Exception | None = None
    for attempt in range(EMBED_MAX_ATTEMPTS):
        try:
            completion = _get_client().embeddings.create(
                model=config.QWEN_EMBEDDING_MODEL,
                input=batch,
                dimensions=config.EMBED_DIMENSIONS,
                encoding_format="float",
            )
            return [item.embedding for item in completion.data]
        except (APIConnectionError, APITimeoutError, RateLimitError, APIStatusError) as exc:
            last_error = exc
            # APIStatusError 仅 5xx 值得重试；4xx（除 429 限流）是参数问题，重试无意义
            if isinstance(exc, APIStatusError) and not isinstance(exc, RateLimitError):
                if exc.status_code < 500:
                    raise
            if attempt < EMBED_MAX_ATTEMPTS - 1:
                time.sleep(EMBED_RETRY_BACKOFF_SECONDS * (2**attempt))
    raise last_error  # type: ignore[misc]


def embed_texts(texts: list[str], batch_size: int = 10) -> list[list[float]]:
    """文本列表 → 向量列表（顺序与输入一致）。"""
    if _resolve()["provider"] == "mock":
        return [_mock_vector(text) for text in texts]

    results: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        results.extend(_embed_batch_with_retry(texts[start : start + batch_size]))
    return results


def _mock_vector(text: str) -> list[float]:
    """字符 bigram 计数哈希 → 固定维度稀疏向量（确定性、零成本，非语义向量）。

    仅用于无 Key 时打通摄取/检索链路与回归验证，检索结果无实际语义。
    """
    dimensions = config.MOCK_EMBED_DIMENSIONS
    vector = [0.0] * dimensions
    for index in range(len(text) - 1):
        bigram = text[index : index + 2]
        digest = hashlib.md5(bigram.encode("utf-8")).hexdigest()
        vector[int(digest[:8], 16) % dimensions] += 1.0
    return vector
