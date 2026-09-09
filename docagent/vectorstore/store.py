"""文档块向量库：Chroma 持久化 + md5 增量幂等 upsert（沿用 demo1 已验证模型）。

对外接口：
    get_collection()                       -> chromadb.Collection
    upsert_doc_chunks(chunks, doc_id)      -> dict  # 单文档幂等入库
    search_docs(query_texts, top_k)        -> list  # 显式向量检索（M3 问答检索层）

id 业务主键 = {doc_id}:{block_seq}（溯源 schema §3.1 定稿，block_seq 由切片层补全）。
md5 只对块文本计算：库里同 id 且 md5 相同 → skip（不 touch、不重嵌、不重计费）；
id 不存在或 md5 不同 → 重新 embedding + upsert（同 id 覆盖，增量补录语义）。

禁止反向 diff 删除：本层只提供 upsert，永不推断删除（文档下架/删档是业务层
职责——demo1 铁律，防增量更新模式误删未更新文档）。

Chroma 坑（2026-09-08 实证）：collection.query(query_texts=...) 若不给
query_embeddings，chroma 会用内置默认 embedding 函数（all-MiniLM-L6-v2），
首次使用触发 79MB ONNX 模型下载且与库内向量维度不一致——检索必须显式
用 embed_texts 生成 query 向量（search_docs 已封装）。

metadata 写入 chroma 的值需为标量（str/int/float/bool）；块溯源键全部合规，
本层追加 doc_id 与 md5 后直接落库。
"""
from __future__ import annotations

import hashlib
import threading
from datetime import datetime

import chromadb

from .. import config
from ..ingest.chunk_models import DocumentChunk
from . import embedding as embed_module

# ---- 并发安全（2026-09-09 修：多 worker 并发摄取会同时写同一 Chroma sqlite）----
# 1) Chroma 要求同一路径只建一个 PersistentClient；原 get_collection 每次 new 一个，
#    多 worker 并发 → "database is locked"/"SQLite objects created in a thread"。改为进程内单例。
# 2) 嵌入 + 落库（sqlite 写）用一把进程级锁串行化：解析/切块仍并行，只有碰 Chroma
#    与打 DashScope 的共享段串行，杜绝写锁争用（大文件长跑必炸）。
_client = None
_client_lock = threading.Lock()
_store_lock = threading.Lock()


def get_collection() -> chromadb.Collection:
    """返回进程内唯一持久化 Collection（Chroma 要求同一路径只建一个 client）。"""
    global _client
    with _client_lock:
        if _client is None:
            db = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
            _client = db.get_or_create_collection(name=config.COLLECTION_NAME)
    return _client


def _text_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def delete_doc_blocks(doc_id: str) -> int:
    """显式删除某文档的全部块（业务层重建调用，如 VLM 转录升级后的乱码块清理）。

    与"永不推断删除"铁律的关系：本函数只做显式删除，不做 diff 推断——
    调用方（CLI/编排）明确知道该文档需要重建时才调用；日常增量摄取绝不触碰。
    """
    collection = get_collection()
    existing = collection.get(where={"doc_id": doc_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    return len(existing["ids"])


def upsert_doc_chunks(chunks: list[DocumentChunk], doc_id: str) -> dict:
    """单个文档的块整体幂等入库：diff → 仅对变化块 embedding + upsert。

    返回（进摄取报告）：
        stored / skipped          本次实际写入 / md5 命中跳过的块数
        provider / degraded       由 embedding 提供方决定（报告标注降级）
    """
    provider = embed_module.info()
    if not chunks:
        return {"stored": 0, "skipped": 0, **provider}

    collection = get_collection()
    local_items = []
    # 本批次的入库时刻：新写/变化块统一打点；md5 命中 skip 的块保留首次入库值
    ingested_at = datetime.now().astimezone().isoformat(timespec="seconds")
    for block_index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata)  # 拷贝，不污染切块层产物
        metadata["doc_id"] = doc_id
        metadata["md5"] = _text_md5(chunk.text)
        metadata["ingested_at"] = ingested_at
        local_items.append(
            {
                "id": f"{doc_id}:{metadata.get('block_seq', block_index)}",
                "metadata": metadata,
                "text": chunk.text,
            }
        )

    # 串行化：diff 读 + 嵌入 + 落库 同持 _store_lock，保证同一时刻仅一个 worker 碰
    # Chroma sqlite（与打 DashScope）。解析/切块在锁外并行，仅此共享段串行。
    skipped = 0
    with _store_lock:
        db_result = collection.get(ids=[item["id"] for item in local_items])
        db_md5_by_id = {
            db_id: (meta or {}).get("md5")
            for db_id, meta in zip(db_result["ids"], db_result["metadatas"])
        }
        to_upsert = [
            item for item in local_items
            if db_md5_by_id.get(item["id"]) != item["metadata"]["md5"]
        ]
        skipped = len(local_items) - len(to_upsert)
        if not to_upsert:
            return {"stored": 0, "skipped": skipped, **provider}
        embeddings = embed_module.embed_texts([item["text"] for item in to_upsert])
        collection.upsert(
            ids=[item["id"] for item in to_upsert],
            documents=[item["text"] for item in to_upsert],
            embeddings=embeddings,
            metadatas=[item["metadata"] for item in to_upsert],
        )
    return {"stored": len(to_upsert), "skipped": skipped, **provider}


def search_docs(query_texts: list[str], top_k: int = 5, where: dict | None = None) -> list[dict]:
    """向量检索：显式 query embedding（绕开 chroma 默认模型下载/维度不一致）。

    where（2026-09-08 起，问答层年份过滤用）：Chroma metadata 过滤条件，
    例如 {"doc_year": {"$in": [2024, 2025]}} 只在这两年份的块里检索——
    防跨年报表同名指标张冠李戴；无日期文档（doc_year 缺失）自动被排除。

    返回按距离升序的块列表：[{id, text, metadata, distance}]——
    M3 问答检索层据此组装上下文与引用溯源（metadata.page/section/doc_id）。
    """
    collection = get_collection()
    query_embeddings = embed_module.embed_texts(query_texts)
    query_kwargs: dict = {"query_embeddings": query_embeddings, "n_results": top_k}
    if where:
        query_kwargs["where"] = where
    hits = collection.query(
        **query_kwargs,
        include=["documents", "metadatas", "distances"],
    )
    results: list[dict] = []
    for index, block_id in enumerate(hits["ids"][0]):
        results.append(
            {
                "id": block_id,
                "text": hits["documents"][0][index],
                "metadata": hits["metadatas"][0][index],
                "distance": hits["distances"][0][index],
            }
        )
    return results
