"""文档块向量库：Chroma 持久化 + md5 增量幂等 upsert（沿用 demo1 已验证模型）。

对外接口：
    get_collection()                       -> chromadb.Collection
    upsert_doc_chunks(chunks, doc_id)      -> dict  # 单文档幂等入库

id 业务主键 = {doc_id}:{block_seq}（溯源 schema §3.1 定稿，block_seq 由切片层补全）。
md5 只对块文本计算：库里同 id 且 md5 相同 → skip（不 touch、不重嵌、不重计费）；
id 不存在或 md5 不同 → 重新 embedding + upsert（同 id 覆盖，增量补录语义）。

禁止反向 diff 删除：本层只提供 upsert，永不推断删除（文档下架/删档是业务层
职责——demo1 铁律，防增量更新模式误删未更新文档）。

metadata 写入 chroma 的值需为标量（str/int/float/bool）；块溯源键全部合规，
本层追加 doc_id 与 md5 后直接落库。
"""
from __future__ import annotations

import hashlib

import chromadb

from .. import config
from ..ingest.chunk_models import DocumentChunk
from . import embedding as embed_module


def get_collection() -> chromadb.Collection:
    """返回持久化 Collection（不存在则创建）。"""
    db = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return db.get_or_create_collection(name=config.COLLECTION_NAME)


def _text_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


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
    for block_index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata)  # 拷贝，不污染切块层产物
        metadata["doc_id"] = doc_id
        metadata["md5"] = _text_md5(chunk.text)
        local_items.append(
            {
                "id": f"{doc_id}:{metadata.get('block_seq', block_index)}",
                "metadata": metadata,
                "text": chunk.text,
            }
        )

    # diff：查库内已有同 id 记录的 md5
    db_result = collection.get(ids=[item["id"] for item in local_items])
    db_md5_by_id = {
        db_id: (meta or {}).get("md5")
        for db_id, meta in zip(db_result["ids"], db_result["metadatas"])
    }

    to_upsert = []
    skipped = 0
    for item in local_items:
        if db_md5_by_id.get(item["id"]) == item["metadata"]["md5"]:
            skipped += 1  # 内容未变：不 touch，保留旧向量与旧 metadata
        else:
            to_upsert.append(item)

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
