"""Excel(.xlsx) 切片策略：一行 = 一条业务记录，整行作为一个知识块。

原理（demo1 期已实证）：表格不同于散文，行与列是作者切好的业务边界——
"一个商品/一笔订单/一条员工记录"就是检索时的最小问题单元。
若按字符窗口切表格，一行记录会被拦腰切断，检索只能命中半个记录。

实现：
- 每个 sheet 独立处理，sheet 名进 section（溯源"数据在哪个表"）
- 第一行当表头，其后每行拼成 "字段名：值" 文本（值缺失的字段跳过）
- 合并单元格：pandas 只保留左上角值，其余位置为空 → 空值字段天然被跳过，
  不留空壳；跨行合并的组信息由同 sheet 其他行补全（v1 取舍，够演示用）

v1 取舍：不做跨 sheet 合并、不做公式求值（openpyxl 读公式单元格得公式串，
演示语料为纯值表；真实业务接数据导出件即可）。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .chunk_models import MAX_CHUNK_CHARS, DocumentChunk, split_oversized_text


def chunk_excel_file(file_path: Path, source_file: str, document_title: str) -> list[DocumentChunk]:
    """读取 .xlsx，每个 sheet 按"一行一记录"切成知识块。"""
    chunks: list[DocumentChunk] = []
    # dtype=str：号码/单号等纯数字列不丢前导零；sheet_name=None 读全部 sheet
    sheets = pd.read_excel(file_path, sheet_name=None, dtype=str)

    for sheet_name, dataframe in sheets.items():
        chunks.extend(_chunk_one_sheet(dataframe, sheet_name, source_file, document_title))
    return chunks


def _chunk_one_sheet(
    dataframe: pd.DataFrame,
    sheet_name: str,
    source_file: str,
    document_title: str,
) -> list[DocumentChunk]:
    dataframe = dataframe.dropna(how="all")  # 全空行（跨行合并的延续行）无信息，删
    if dataframe.empty:
        return []

    column_names = [str(column) for column in dataframe.columns]
    chunks: list[DocumentChunk] = []
    for row_number, row_values in dataframe.iterrows():
        text_parts = _format_row_as_text(column_names, row_values)
        if not text_parts:
            continue
        record_text = "；".join(text_parts)
        if len(record_text) > MAX_CHUNK_CHARS:
            # 单行记录异常长（如备注大段文本）：按句子二次切，仍属该 sheet
            for sub_text in split_oversized_text(record_text):
                chunks.append(_build_chunk(sub_text, sheet_name, row_number, source_file, document_title))
        else:
            chunks.append(_build_chunk(record_text, sheet_name, row_number, source_file, document_title))
    return chunks


def _format_row_as_text(column_names: list[str], row_values: pd.Series) -> list[str]:
    """一行记录 → ["商品名称：蓝牙耳机", "价格：199 元", ...]；空单元格跳过。"""
    text_parts: list[str] = []
    for column_name, value in zip(column_names, row_values):
        if pd.isna(value):
            continue  # 合并单元格/空值：不留 "字段名：nan" 空壳
        value_text = str(value).strip()
        if not value_text:
            continue
        text_parts.append(f"{column_name}：{value_text}")
    return text_parts


def _build_chunk(
    record_text: str,
    sheet_name: str,
    row_number: int,
    source_file: str,
    document_title: str,
) -> DocumentChunk:
    metadata = {
        "doc_id": None,
        "title": document_title,
        "page": row_number + 1,  # 借用 page 字段存"表格行号"，溯源可指到具体行（1 起始）
        "section": f"工作表 {sheet_name}",
        "block_type": "table_record",
        "source_file": source_file,
        "format": "excel",
    }
    return DocumentChunk(text=record_text, metadata=metadata)
