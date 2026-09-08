"""图片资源仓库：图文件落磁盘 + sqlite 登记索引（幂等，图片不入向量库）。

背景（D6 文本单通道 + 2026-09-08 图片引用设计）：
    图片不参与向量检索。切块器提取图片 → 文本流原位写 [IMAGE:{image_id}]
    占位符 → 本仓库把图文件落到 data/images/{doc_id}/ 并登记 sqlite；
    块 metadata.image_ids 串起引用；召回后由展示层按 image_id 取图填充（M3）。

幂等语义（与向量库同款"增量不删除"哲学）：
    - image_id = {doc_stem}-{内容md5前10}（chunk_models.make_image_id）
    - 同文档重跑 → 同 image_id → 文件已存在则不重写；sqlite INSERT OR IGNORE
    - 永不推断删除：文档改版后不再被引用的孤儿图由业务层清理（与 vectorstore 同铁律）

sqlite 仅作索引（id → 文件位置），python 内置 sqlite3，单进程摄取足够。
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from .. import config
from ..ingest.chunk_models import ExtractedImage

_SCHEMA = """
CREATE TABLE IF NOT EXISTS image_files (
    id          TEXT PRIMARY KEY,   -- image_id（doc_stem-内容md5前10）
    doc_id      TEXT NOT NULL,
    rel_path    TEXT NOT NULL,      -- {doc_id}/{image_id}.{ext}（相对 IMAGES_DIR）
    ext         TEXT NOT NULL,
    bytes_size  INTEGER NOT NULL,
    created_at  TEXT NOT NULL
)
"""


def _connect() -> sqlite3.Connection:
    """打开（必要时创建）图片索引库，确保表存在。"""
    config.IMAGES_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(config.IMAGES_DB_PATH))
    connection.execute(_SCHEMA)
    return connection


def save_images(doc_id: str, images: list[ExtractedImage]) -> dict:
    """落盘并登记一批图片；返回 {"saved": n, "skipped": n}（进摄取报告）。

    同一 image_id 重跑：文件存在（同 id 即同内容）跳过重写，登记幂等。
    """
    if not images:
        return {"saved": 0, "skipped": 0}

    target_dir = config.IMAGES_DIR / doc_id
    target_dir.mkdir(parents=True, exist_ok=True)
    connection = _connect()
    saved_count = 0
    skipped_count = 0
    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        for image in images:
            rel_path = f"{doc_id}/{image.image_id}.{image.ext}"  # 相对 IMAGES_DIR
            target_file = config.IMAGES_DIR / rel_path
            if target_file.exists() and target_file.stat().st_size == len(image.data):
                skipped_count += 1  # 同 id 即同内容（id 含内容 md5），不重写
            else:
                target_file.write_bytes(image.data)
                saved_count += 1
            connection.execute(
                "INSERT OR IGNORE INTO image_files (id, doc_id, rel_path, ext, bytes_size, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (image.image_id, doc_id, rel_path, image.ext, len(image.data), created_at),
            )
        connection.commit()
    finally:
        connection.close()
    return {"saved": saved_count, "skipped": skipped_count}


def lookup(image_id: str) -> dict | None:
    """按 image_id 取图片登记信息（M3 展示层取图填充占位符用）。"""
    connection = _connect()
    try:
        row = connection.execute(
            "SELECT id, doc_id, rel_path, ext FROM image_files WHERE id = ?", (image_id,)
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    return {"image_id": row[0], "doc_id": row[1], "rel_path": row[2], "ext": row[3]}


def count_images() -> int:
    """已登记图片总数（验证/报告用）。"""
    connection = _connect()
    try:
        (count,) = connection.execute("SELECT COUNT(*) FROM image_files").fetchone()
    finally:
        connection.close()
    return count


def resolve_file_path(image_id: str) -> Path | None:
    """image_id → 磁盘文件绝对路径（文件已删则 None）。"""
    record = lookup(image_id)
    if record is None:
        return None
    file_path = config.IMAGES_DIR / record["rel_path"]
    return file_path if file_path.exists() else None
