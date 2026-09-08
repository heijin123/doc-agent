"""一次性迁移：为存量块补日期 metadata（doc_date / doc_year / ingested_at）。

背景（2026-09-08）：md5 幂等只对块文本计算——metadata 增加新字段不会触发
增量重嵌（同 id + 同 md5 → skip，旧 metadata 原样保留）。因此全库需要一次
显式升级。本脚本只更新 metadata、不重算 embedding、不重跑 VLM/切块，
秒级完成且零 token 成本。

写入逻辑（与摄取侧 chunking._complete_chunk_metadata / store 注入保持一致）：
    doc_date    = 从 metadata.title（= 文件名 stem）解析（extract_doc_date）
    doc_year    = doc_date 有值时写 int 年份
    ingested_at = 升级执行时刻（旧块真实首入时间不可考，此为近似）

幂等：可重复执行（重复 update 同值）；升级后正常增量摄取不受影响
（新摄取自带日期键，md5 命中跳过时保留本脚本写入的值）。

用法：.venv/Scripts/python.exe scripts/upgrade_add_date_metadata.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docagent.ingest.chunk_models import extract_doc_date  # noqa: E402
from docagent.vectorstore import store as vector_store  # noqa: E402


def main() -> int:
    collection = vector_store.get_collection()
    ingested_at = datetime.now().astimezone().isoformat(timespec="seconds")

    # 全量取回（块数 1400 级，一次 get 可承受；分页留待库更大时再改）
    data = collection.get(include=["metadatas"])
    ids, metadatas = data["ids"], data["metadatas"]
    if not ids:
        print("库为空，无需升级")
        return 0

    to_update_ids: list[str] = []
    to_update_metadatas: list[dict] = []
    for block_id, raw_metadata in zip(ids, metadatas):
        metadata = dict(raw_metadata or {})
        doc_date = extract_doc_date(metadata.get("title") or "")
        if doc_date:
            metadata["doc_date"] = doc_date
            metadata["doc_year"] = int(doc_date[:4])
        else:
            metadata.pop("doc_date", None)
            metadata.pop("doc_year", None)
        metadata["ingested_at"] = ingested_at
        to_update_ids.append(block_id)
        to_update_metadatas.append(metadata)

    # Chroma update 只更 metadata，保留原 documents/embeddings（零重嵌）
    batch_size = 500
    for offset in range(0, len(to_update_ids), batch_size):
        end = offset + batch_size
        collection.update(
            ids=to_update_ids[offset:end],
            metadatas=to_update_metadatas[offset:end],
        )
        print(f"已更新 {end}/{len(to_update_ids)}")

    with_dates = sum(1 for metadata in to_update_metadatas if metadata.get("doc_year"))
    print(f"完成：{len(to_update_ids)} 块补 ingested_at；其中 {with_dates} 块解析到文档年份")
    return 0


if __name__ == "__main__":
    sys.exit(main())
