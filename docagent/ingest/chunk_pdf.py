"""PDF 切片策略：一页 = 一个知识块（页内超长再按段落/句子二次切）。

PDF 没有可靠的标题层级（文字层只是坐标+字体），页边界是唯一稳定结构——
现实文档一页通常恰是"一个主题"（产品规格一页、发票一页、论文一页一个要点），
且检索时"答案在第几页"是溯源最常用的定位，page 进 metadata。

图片引用（2026-09-08 起）：
- 页面经 get_text("dict") 取 text/image 两类版面块，按阅读顺序（y 分桶 + x）
  合并为"文本流"——图片在原位写成 [IMAGE:{image_id}] 占位符（随块入库，
  展示层召回后替换为 <img>）；图字节收集进 ChunkingResult.images 待仓库落盘。
- 纯图/扫描页（无文本块）仍跳过——图内信息留给 M2 VLM 转录兜底，本层不越权。
- 双栏/乱码等复杂版面仍由质量门标黄/红后走慢路径，不做启发式猜测。

表格感知（2026-09-09 起，两轮演进）：
- 第一轮：find_tables() 检测带框线表格 → 表格区按 (行,列) 序渲染、剔除表格区内
  dict 文本防双写（根治纯 y 排序把 √ 勾插进折行中间的内容撕裂）。
- 第二轮（跨 2-3+ 页超大表）：块构造从"整页文本一次切"升级为 **part 级原子聚合**——
  * 跨页表先聚成"链"（连续页段：上页贴底表 + 下页贴顶表）：
    - 短链（链上各页表文本累计 ≤ MAX）→ 并入一页成块（一张表一个上下文）；
    - 长链（几十页财务表）→ 不合并，各页独立成块，但**非首页页顶部注入表头锚**
      （链首页表的首行列名）——否则中段页块是"无列名的数字孤儿"，问"某科目原值"
      时向量不知道哪列是原值哪列是折旧。
  * 页内文本二次切按 **part 边界**：表格是原子单元，不再与正文段落粘在一起被
    按句子切穿；表格自身超 MAX 时**按行切**（行是表的最小语义单元），且每个
    子块顶部重复该表首行（列名），子块自包含。

文本渲染注意：span 是同一行内按样式切分的片段，直接拼接会丢词间空格
（"foo bar" 在 span 边界 → "foobar"）——按 span 间隙 ≥ 0.35×字号 补空格。

溯源：page 存人类页码（1 起始）；页内二次切的子块仍带同页号；跨页合并块以
起始页为准；纯表格块 block_type="table"，混排块沿用 "page"。
"""
from __future__ import annotations

from pathlib import Path

import pymupdf

from .chunk_models import (
    MAX_CHUNK_CHARS,
    DocumentChunk,
    ExtractedImage,
    make_image_id,
    split_oversized_text,
)

_SPAN_GAP_RATIO = 0.35  # 相邻 span 间隙 ≥ 0.35×字号 → 视为词间空格
_LINE_BUCKET_POINTS = 8  # y 坐标分桶容差（同一视觉行的文本/图算同排）
_TABLE_BOTTOM_RATIO = 0.78  # 表格底缘 ≥ 页高×此比例 → 可能被页边界截断（跨页）
_TABLE_TOP_RATIO = 0.12     # 表格顶缘 ≤ 页高×此比例 → 可能是续页表（无正文隔断）


def chunk_pdf_file(
    file_path: Path,
    source_file: str,
    document_title: str,
    vlm_transcripts: dict[int, str] | None = None,
) -> tuple[list[DocumentChunk], list[ExtractedImage]]:
    """读取 PDF，按页切成知识块；同时提取页内图片（占位符已写入块文本）。

    vlm_transcripts（M2 慢路径注入）：{页号: 转录文本}——红页原文文本不可信，
    不进库；转录文本作为 figure_transcript 块**追加到文档块流尾部**（老块 id
    全部不动 → 无 block_seq 平移、无孤儿；幂等靠转录文本 md5）。默认
    None = 纯快路径（含红页原文，M1 行为，供无 Key 降级）。
    """
    source_stem = Path(source_file).stem
    chunks: list[DocumentChunk] = []
    all_images: list[ExtractedImage] = []
    seen_image_ids: set[str] = set()
    transcripts = vlm_transcripts or {}
    tail_transcript_chunks: list[tuple[str, int]] = []  # (转录文本, 页号) 待尾部追加

    # 第一遍：逐页抽版面流（正文片段 + 表格片段 + 图片），红页只记转录不产正文
    page_parts: list[list[str]] = []               # 每页渲染片段列表
    page_table_indices: list[set[int]] = []        # 每页"表格片段"的 parts 下标集合
    page_fresh_images: list[list[ExtractedImage]] = []
    page_bottom_table: list[int | None] = []       # 贴底表在 parts 的下标（None=无）
    page_top_table: list[int | None] = []          # 贴顶表在 parts 的下标（None=无）

    with pymupdf.open(str(file_path)) as document:
        for page_index, page in enumerate(document):
            page_number = page_index + 1  # 人类页码 1 起始
            transcript_text = transcripts.get(page_number)
            if transcript_text is not None:
                # 红页：原文不产块；转录文本统一收集到尾部（避免本页块数变化平移后续 id）
                tail_transcript_chunks.append((transcript_text, page_number))
                page_parts.append([])
                page_table_indices.append(set())
                page_fresh_images.append([])  # 与 page_parts 下标对齐（红页无图产出）
                page_bottom_table.append(None)
                page_top_table.append(None)
                continue
            parts, fresh_images, flags = _extract_page_stream(
                page, document, source_stem, seen_image_ids
            )
            page_parts.append(parts)
            page_table_indices.append(flags["table_part_indices"])
            page_fresh_images.append(fresh_images)
            page_bottom_table.append(flags["bottom_table_part_index"])
            page_top_table.append(flags["top_table_part_index"])

    # 第二遍：跨页表链处理（识别 → 短链合并 / 长链表头锚注入）
    _handle_table_chains(
        page_parts,
        page_table_indices,
        page_bottom_table,
        page_top_table,
    )

    # 第三遍：逐页按 part 原子成块（表格原子整块入块；超限表格按行切）
    for page_index, parts in enumerate(page_parts):
        page_number = page_index + 1
        all_images.extend(page_fresh_images[page_index])
        table_indices = page_table_indices[page_index]
        atom_groups = _page_atom_groups(parts, table_indices)
        for atom_text, atom_kind in atom_groups:
            block_type = "table" if atom_kind == "table" else "page"
            chunks.append(_build_chunk(atom_text, page_number, block_type, source_file, document_title))

    # 转录块尾部追加（block_seq 由分发层统一连续编号，老块序号不受影响）
    for transcript_text, page_number in tail_transcript_chunks:
        if len(transcript_text) > MAX_CHUNK_CHARS:
            for sub_text in split_oversized_text(transcript_text):
                chunks.append(_build_chunk(sub_text, page_number, "figure_transcript", source_file, document_title))
        else:
            chunks.append(_build_chunk(transcript_text, page_number, "figure_transcript", source_file, document_title))
    return chunks, all_images


# ---------------- 跨页表链 ----------------

def _handle_table_chains(
    page_parts: list[list[str]],
    page_table_indices: list[set[int]],
    page_bottom_table: list[int | None],
    page_top_table: list[int | None],
) -> None:
    """处理相邻页间的跨页表：短表合并成一块（一张表一个上下文）；长表分页
    独立成块 + 续页注入表头锚（防中段页成"无列名数字孤儿"）。就地修改入参。

    逐对接力式（不预构建多跳链）：跨 3+ 页大表的中间页是整页表（贴顶==贴底
    同一片段），其文本前 2 行即表头锚 → 后一跳锚源自动正确；而"页内顶部与
    底部是两张不同表"的版面不会把不相关表误串成一条链。识别只用"相邻页
    贴底/贴顶"启发式（与 PDF 无可靠结构的现实一致；列宽/表头比对属 v1.1）。
    """
    page_count = len(page_parts)
    for page_index in range(page_count - 1):
        _resolve_table_pair(
            page_index,
            page_parts, page_table_indices, page_bottom_table, page_top_table,
        )


def _resolve_table_pair(
    page_index: int,
    page_parts: list[list[str]],
    page_table_indices: list[set[int]],
    page_bottom_table: list[int | None],
    page_top_table: list[int | None],
) -> None:
    """对相邻页对 (page_index, page_index+1) 做跨页表决策（一次一跳，逐对接力）。

    - 上页贴底表 + 下页贴顶表 → 两片段文本合计 ≤ MAX → **合并**成一块
      （一张表一个上下文；短表行为）；
    - 合计 > MAX（长表/超大表）→ **不合并**，下页贴顶表顶部注入表头锚
      （上页表首行列名，取前 2 行）——下页块因此自包含列名，不成为
      "无表头的数字孤儿"。锚注入幂等（已带同锚则跳过）。

    不做多跳预构建：跨 3+ 页大表的中间页必是整页表（贴顶==贴底同一片段），
    逐对接力时该片段文本前 2 行 = 原始表头锚 → 后续跳锚源自动正确；而
    "页内顶部与底部是两张不同表"（如 p20 顶键值尾行 + 底研发表）不会误串链
    ——每对独立按当前片段判定，pop 后下标即时修正。
    """
    page = page_index
    next_page = page_index + 1
    bottom_index = page_bottom_table[page]
    top_index = page_top_table[next_page]
    if bottom_index is None or top_index is None:
        return
    current_parts = page_parts[page]
    next_parts = page_parts[next_page]
    if bottom_index >= len(current_parts) or top_index >= len(next_parts):
        return

    anchor = "\n".join(current_parts[bottom_index].splitlines()[:2])
    merged_length = len(current_parts[bottom_index]) + len(next_parts[top_index])
    if merged_length <= MAX_CHUNK_CHARS:
        # 短表 → 合并进上页块（保持一张表一个上下文）
        current_parts[bottom_index] = (
            current_parts[bottom_index].rstrip() + "\n" + next_parts[top_index].strip()
        )
        next_parts.pop(top_index)
        page_table_indices[next_page].discard(top_index)
        _shift_indices_after_pop(
            page_bottom_table, page_top_table, page_table_indices, next_page, top_index
        )
    else:
        # 长表 → 分页独立成块 + 表头锚注入（续页块自包含列名）
        table_text = next_parts[top_index]
        if table_text and not table_text.startswith(anchor):
            next_parts[top_index] = anchor + "\n" + table_text


def _shift_indices_after_pop(
    page_bottom_table: list[int | None],
    page_top_table: list[int | None],
    page_table_indices: list[set[int]],
    page_index: int,
    popped_index: int,
) -> None:
    """pop 掉 page_index 页的一个片段后，该页全部下标记录同步：
    记录 == popped（片段本身被删）→ 置 None / 移除；记录 > popped → 减 1。
    覆盖 bottom/top（跨页判定用）与 table_indices（成块标 type 用）——
    漏修 table_indices 会把 pop 后错位的标题当表格、真表格当正文。
    """
    for record in (page_bottom_table, page_top_table):
        value = record[page_index]
        if value is None:
            continue
        if value == popped_index:
            record[page_index] = None
        elif value > popped_index:
            record[page_index] = value - 1
    indices = page_table_indices[page_index]
    page_table_indices[page_index] = {
        (index - 1 if index > popped_index else index)
        for index in indices
        if index != popped_index
    }


# ---------------- 页内成块 ----------------

def _is_page_number_only(text: str) -> bool:
    """页脚页码判断：整段是纯数字（1-4 位，允许空白），如 "2" / "100" / "19 "。
    页眉/页脚页码是独立排版块且无检索价值，剔除避免产生 1-2 字垃圾块。
    """
    return bool(text.strip().isdigit() and len(text.strip()) <= 4)


def _page_atom_groups(parts: list[str], table_indices: set[int]) -> list[tuple[str, str]]:
    """把一页的渲染片段聚成"原子组"（块文本, 块类型）。

    原子 = 单个片段；表格片段是**不可拆**原子（跨页短链合并后仍是一个片段），
    正文/图片片段可相邻聚合到接近 MAX（片段间空行分隔，段落语义完整）。
    表格片段自身超 MAX → 按行切多段（每段顶部带该表首行列名，自包含）。
    """
    atoms: list[tuple[str, str]] = []  # (text, kind) kind ∈ {text, table}
    for index, part in enumerate(parts):
        if index in table_indices:
            if len(part) <= MAX_CHUNK_CHARS:
                atoms.append((part, "table"))
            else:
                atoms.extend((segment, "table") for segment in _split_table_part(part))
        else:
            # 正文/图片片段：超长才二次切（段落/句子，原逻辑）；否则整段为原子。
            # 页眉/页脚常是独立排版块——纯数字页码（"2"/"100"）无检索价值，剔除
            if _is_page_number_only(part):
                continue
            if len(part) <= MAX_CHUNK_CHARS:
                atoms.append((part, "text"))
            else:
                atoms.extend((segment, "text") for segment in split_oversized_text(part))

    # 相邻 text 原子聚合到 MAX（表格原子不参与聚合——它是完整上下文边界）
    groups: list[tuple[str, str]] = []
    buffer: list[str] = []
    buffer_length = 0
    for atom_text, atom_kind in atoms:
        if atom_kind == "table":
            if buffer:
                groups.append(("\n\n".join(buffer), "text"))
                buffer, buffer_length = [], 0
            groups.append((atom_text, "table"))
            continue
        if buffer and buffer_length + len(atom_text) + 2 > MAX_CHUNK_CHARS:
            groups.append(("\n\n".join(buffer), "text"))
            buffer, buffer_length = [], 0
        buffer.append(atom_text)
        buffer_length += len(atom_text) + 2
    if buffer:
        groups.append(("\n\n".join(buffer), "text"))
    return groups


def _split_table_part(table_text: str) -> list[str]:
    """超长表格按行切段：行是表的最小语义单元，绝不拦腰切单元格。

    每段 = 该表首行（列名）+ 若干数据行——后段因此自包含列名上下文；
    行切后仍超长的单行（罕见巨长单元格）按窗口硬切兜底。
    """
    rows = table_text.splitlines()
    if len(rows) <= 1:
        return split_oversized_text(table_text) if len(table_text) > MAX_CHUNK_CHARS else [table_text]
    header = rows[0]
    budget = MAX_CHUNK_CHARS - len(header) - 4  # 每段可承载的数据行字符预算
    pieces: list[str] = []
    current: list[str] = [header]
    current_length = len(header)
    for row in rows[1:]:
        if current_length + len(row) + 1 > budget and len(current) > 1:
            pieces.append("\n".join(current))
            current = [header]
            current_length = len(header)
        current.append(row)
        current_length += len(row) + 1
    if current:
        pieces.append("\n".join(current))
    return pieces


# ---------------- 版面流提取 ----------------

def _extract_page_stream(page, document, source_stem: str, seen_image_ids: set[str]):
    """把一页转成"正文片段 + 表格片段 + 图片占位"的阅读流。

    返回 (parts, fresh_images, flags)：
      parts        渲染片段列表——每个表格为独立片段（行列序文本）；
      fresh_images 本页新增图片；
      flags        table_part_indices（表格片段下标集合）/ bottom_table_part_index
                   （贴底表下标）/ top_table_part_index（贴顶表下标）。
    """
    page_height = page.rect.height
    page_blocks = page.get_text("dict")["blocks"]
    text_blocks = [block for block in page_blocks if block["type"] == 0]
    image_infos = page.get_image_info(xrefs=True)  # 每项含 bbox/xref/width/height

    # 表格检测：带框线表格按 (行,列) 渲染；其区域内 dict 文本剔除（防双输出）
    tables = []
    try:
        tables = page.find_tables().tables
    except Exception:
        tables = []  # 表格检测失败退回纯文本流（不阻断摄取）
    table_rects = [pymupdf.Rect(table.bbox) for table in tables]

    def center_inside_table(bounding_box: tuple) -> bool:
        center = pymupdf.Point(
            (bounding_box[0] + bounding_box[2]) / 2,
            (bounding_box[1] + bounding_box[3]) / 2,
        )
        return any(table_rect.contains(center) for table_rect in table_rects)

    # 阅读顺序排序：正文文本块 + 表格 + 图片，按 (y 桶, x) 混合排列
    items: list[tuple[str, object]] = []
    for block in text_blocks:
        if center_inside_table(block["bbox"]):
            continue  # 表格区内文本由 find_tables 行列序输出，剔除防双写
        items.append(("text", block))
    for table in tables:
        items.append(("table", table))
    for info in image_infos:
        items.append(("image", info))
    items.sort(key=lambda kind_and_payload: _reading_order_key(kind_and_payload[1]))

    parts: list[str] = []
    fresh_images: list[ExtractedImage] = []
    table_part_indices: set[int] = set()
    part_index_of: dict[int, int] = {}  # items 下标 → parts 下标（供跨页续接定位）
    bottom_table_index = None  # 贴底表格（items 下标，可能被页边界截断）
    top_table_index = None     # 贴顶表格（items 下标，可能是续页表）
    for index, (kind, payload) in enumerate(items):
        if kind == "table":
            bounding_box = _payload_bbox(payload)
            if bounding_box[3] >= page_height * _TABLE_BOTTOM_RATIO:
                bottom_table_index = index
            if bounding_box[1] <= page_height * _TABLE_TOP_RATIO:
                top_table_index = index
        if kind == "text":
            rendered = _render_text_block(payload)
            if not rendered:
                continue
        elif kind == "table":
            rendered = _render_table(payload)
            if not rendered:
                continue
        else:  # image
            xref = payload.get("xref") or 0
            if not xref:
                continue  # 无可提取 xref 的装饰性图，跳过（与旧行为一致）
            extracted = document.extract_image(xref)  # 可能抛异常（损坏流）→ 外层兜
            image_bytes = extracted["image"]
            image_ext = extracted.get("ext") or "png"
            image_id = make_image_id(source_stem, image_bytes)
            rendered = f"[IMAGE:{image_id}]"
            if image_id not in seen_image_ids:
                seen_image_ids.add(image_id)
                fresh_images.append(ExtractedImage(image_id=image_id, data=image_bytes, ext=image_ext))
        part_index_of[index] = len(parts)
        parts.append(rendered)
        if kind == "table":
            table_part_indices.add(len(parts) - 1)

    # 跨页判定：扫描全部表格（页脚页码等小文本可能排在表格后，不能只看首/尾片段）
    flags = {
        "table_part_indices": table_part_indices,
        "bottom_table_part_index": part_index_of.get(bottom_table_index) if bottom_table_index is not None else None,
        "top_table_part_index": part_index_of.get(top_table_index) if top_table_index is not None else None,
    }
    return parts, fresh_images, flags


def _payload_bbox(payload) -> tuple:
    """统一取版面单元 bbox：dict 文本块（block['bbox']）与 Table 对象（.bbox）。"""
    return payload["bbox"] if isinstance(payload, dict) else payload.bbox


def _reading_order_key(payload) -> tuple[int, float]:
    """版面块排序键：y 分桶（同行容差）优先，再按 x。"""
    bounding_box = _payload_bbox(payload)
    return (round(bounding_box[1] / _LINE_BUCKET_POINTS), bounding_box[0])


def _render_text_block(block: dict) -> str:
    """文本版面块 → 多行文本（span 间隙补空格，行间保留换行）。"""
    lines = []
    for line in block.get("lines", []):
        rendered_line = _render_line_spans(line.get("spans", []))
        if rendered_line:
            lines.append(rendered_line)
    return "\n".join(lines)


def _render_table(table) -> str:
    """表格 → 行主序文本：每行 = 各单元格文本按列拼接（格内折行折叠成一行）。

    按 (行,列) 而非 (y,x) 排序，根治"表格行内居中小元素插进相邻格折行中间"
    的错乱（多列表格下 √/序号 与折行文本的 y 交错）。跨页截断的行保留原文
    （由上层把续页行文本拼到本块尾部 / 注入表头锚）。
    """
    lines: list[str] = []
    for row in table.extract():
        cells: list[str] = []
        for cell in row:
            if cell is None:
                continue  # 合并单元格的空位跳过
            cell_text = " ".join(str(cell).split())  # 格内折行折叠为一行
            if cell_text:
                cells.append(cell_text)
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _render_line_spans(spans: list[dict]) -> str:
    """一行内多个 span：相邻 span 间隙 ≥ 0.35×字号 时补空格，防词被拼死。"""
    parts: list[str] = []
    previous_end_x: float | None = None
    for span in spans:
        span_text = span.get("text", "")
        span_bounding = span.get("bbox", (0, 0, 0, 0))
        font_size = span.get("size", 10)
        if previous_end_x is not None and span_bounding[0] - previous_end_x > _SPAN_GAP_RATIO * font_size:
            parts.append(" ")
        parts.append(span_text)
        previous_end_x = span_bounding[2]
    return "".join(parts).strip()


def _build_chunk(
    text: str,
    page_number: int,
    block_type: str,
    source_file: str,
    document_title: str,
) -> DocumentChunk:
    metadata = {
        "doc_id": None,
        "title": document_title,
        "page": page_number,
        "section": f"第 {page_number} 页",
        "block_type": block_type,
        "source_file": source_file,
        "format": "pdf",
    }
    return DocumentChunk(text=text, metadata=metadata)
