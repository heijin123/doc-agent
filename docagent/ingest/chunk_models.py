"""切块的统一数据结构、块大小参数与通用二次切工具——所有格式切块器共用。

溯源字段与需求文档 §3.1 对齐：doc_id / title / page / section / block_type / block_seq。
- doc_id     ：入库时由摄取编排生成（本 demo 每文档一个 doc_id），切块阶段留 None
- block_seq  ：文档内块序号，分发层统一切块完成后补（保证连续 0..n-1）
- 其余溯源键  ：各切块器按格式填充（没有对应信息的键不出现，如 pdf 才有 page）

二次切工具放这里（而非某个格式模块）：pdf/excel/json/ppt/docx/md 的
"单块超长"都要二次切，放公共处避免各格式反向依赖文本类模块。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---- 块大小参数 ----
# 目标块大小按"embedding 有效长度内尽量装语义完整单元"取字符数：
# 中文约 0.6~1 token/字，900 字符 ≈ 600~900 token，留足余量（学习计划"512 token 起步"）
TARGET_CHUNK_CHARS = 900
MAX_CHUNK_CHARS = 1200      # 单块硬上限，超过必须二次切
OVERLAP_CHARS = 80          # 二次切时的相邻重叠，保住跨切点的语义连贯

# 句子边界（中英标点）——二次切按句切分，尽量不把一句话劈两半
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])")

# 溯源键集合：验证脚本断言"每个块都只含这些已知键"（防切块器自造散键）
TRACEABILITY_KEYS = (
    "doc_id",
    "title",
    "page",
    "section",
    "block_type",
    "block_seq",
    "source_file",
    "format",
)


@dataclass
class DocumentChunk:
    """一个可检索的知识块：文本 + 溯源元数据。"""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkingResult:
    """单文档切块结果——摄取报告按此逐文档记录。"""

    source_file: str
    document_type: str
    detected_by: str
    chunks: list[DocumentChunk]


# ==================== 通用二次切工具 ====================

def split_oversized_text(text: str) -> list[str]:
    """超长文本二次切：先按段落聚到 TARGET 大小，单段仍超限再按句子硬切。

    返回值均已去除首尾空白；切点语义丢失由相邻块重叠（OVERLAP_CHARS）补偿。
    各格式切块器共用：单页/单行/单元素文本超过 MAX_CHUNK_CHARS 时调用。
    """
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]

    # 单段都没超限：按段落聚合到目标大小
    if all(len(paragraph) <= MAX_CHUNK_CHARS for paragraph in paragraphs):
        return _merge_paragraphs_to_target(paragraphs)

    # 存在单段超限：段内按句子切，句子仍超长则固定窗口兜底
    result: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= MAX_CHUNK_CHARS:
            result.append(paragraph)
        else:
            result.extend(_split_oversized_paragraph(paragraph))
    return _merge_paragraphs_to_target(result)


def _merge_paragraphs_to_target(paragraphs: list[str]) -> list[str]:
    """把段落流聚合为块：以 MAX_CHUNK_CHARS 为硬上界贪心，段落尽量不拆。

    为什么用 MAX 而不是 TARGET 做合并阈值：若相邻两段（各约 800 字符，均合法）
    在 TARGET(900) 处切，会得到 800+? 的碎片块；段落是语义单元，能整段收进
    一块就不拆，块大小允许落在 TARGET~MAX 区间（句子级切分才用 TARGET 作目标）。
    """
    merged: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph) + 2  # +2 估段落分隔符
        if current_length + paragraph_length > MAX_CHUNK_CHARS and current_parts:
            merged.append("\n\n".join(current_parts))
            current_parts = []
            current_length = 0
        current_parts.append(paragraph)
        current_length += paragraph_length

    if current_parts:
        merged.append("\n\n".join(current_parts))
    return merged


def _split_oversized_paragraph(paragraph: str) -> list[str]:
    """单个超长段落：按句子边界切成片段，保留 OVERLAP 重叠防语义断点。"""
    sentences = [sentence for sentence in _SENTENCE_BOUNDARY.split(paragraph) if sentence.strip()]
    pieces: list[str] = []
    current_piece = ""

    for sentence in sentences:
        if len(current_piece) + len(sentence) > TARGET_CHUNK_CHARS and current_piece:
            pieces.append(current_piece)
            # 重叠 = 上片段末尾若干字符，保持跨切点的指代连续
            current_piece = current_piece[-OVERLAP_CHARS:] if len(current_piece) > OVERLAP_CHARS else ""
        current_piece += sentence

    if current_piece:
        pieces.append(current_piece)

    # 句子本身超长（无标点的超长行）仍可能 > MAX，最后窗口兜底硬切
    final_pieces: list[str] = []
    for piece in pieces:
        if len(piece) > MAX_CHUNK_CHARS:
            final_pieces.extend(_hard_split_windows(piece))
        else:
            final_pieces.append(piece)
    return final_pieces


def _hard_split_windows(text: str) -> list[str]:
    """无标点可依时的最后手段：固定窗口硬切 + 重叠。"""
    step = TARGET_CHUNK_CHARS - OVERLAP_CHARS
    windows = []
    start = 0
    while start < len(text):
        windows.append(text[start : start + TARGET_CHUNK_CHARS])
        start += step
    return windows
