"""PDF 质量门：逐页判定"这一页 PyMuPDF 够不够用"（移植自 demo1 quality_gate.py）。

红/黄/绿三档 + 四类可解释信号，全部零 LLM 成本：
    红 = 硬伤 → 文本不可信，必须走慢路径（OCR / MinerU / VLM 转录）
    黄 = 软提示 → 文本可用但版面复杂（少文字 / 含表格图片），可选 MinerU 提结构
    绿 = 直接放行 → PyMuPDF 产出够用

信号（阈值可调）：
    1. 文字覆盖率  去空白字符 < 20 → 纯图页/扫描页（红）；20-80 → 文字偏少（黄）
    2. 乱码指纹    拉丁文本主导页，小写占比 < 0.35 → 坏字体乱码（红）
                   （"$SSOH 5HPRWH" 型坏字体全大写无小写；正常英文正文小写 75%+）
    3. 替换/控制符  U+FFFD 或控制字符 > 0 → 提取层报错（红）
    4. 版面复杂度  表格数 / 图片块数 ≥ 1 → 可选 MinerU（黄，软信号）
    中文页无大小写概念 → 跳过乱码指纹，只用覆盖率与替换符。

用法（编排 parse 节点）：
    assess_pdf(path) -> {"stats": {红/黄/绿}, "red_pages": [0 起始页号], "notes": [...]}
一次打开文档完成"逐页取文本 + 判级"，M1 只判级不修补；
红页走 VLM 转录是 M2 的条件边（锚点 = gate 节点），本层不越权。
"""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf

COVERAGE_RED = 20      # 去空白字符 < 20 → 疑似纯图页/扫描页（红）
COVERAGE_YELLOW = 80   # ≥20 但 <80 → 文字偏少，可能含图（黄）
LOWER_RATIO_MIN = 0.35  # 拉丁文本小写占比下限（低于 → 乱码嫌疑红）

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_FFFD_CTRL = re.compile(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]")


def analyze_page(page) -> dict:
    """对单个 pymupdf 页计算四类信号并给出红黄绿判定。"""
    raw = page.get_text()
    stripped = re.sub(r"\s+", "", raw)
    n_chars = len(stripped)
    n_cjk = len(_CJK_RE.findall(raw))
    n_fffd_ctrl = len(_FFFD_CTRL.findall(raw))

    # 拉丁字母计数（数字/符号不计入）
    letters = re.findall(r"[A-Za-z]", raw)
    n_latin = len(letters)
    n_lower = sum(1 for ch in letters if ch.islower())

    reasons = []
    coverage_red = n_chars < COVERAGE_RED
    coverage_yellow = COVERAGE_RED <= n_chars < COVERAGE_YELLOW
    if coverage_red:
        reasons.append(f"覆盖率极低({n_chars}字符,疑似纯图页)")
    elif coverage_yellow:
        reasons.append(f"覆盖率偏低({n_chars}字符,可能含图)")
    if n_fffd_ctrl:
        reasons.append(f"替换/控制符x{n_fffd_ctrl}")

    # 乱码指纹：仅当拉丁字母主导（>150）且页面不是纯中文时启用
    garbled = False
    lower_ratio = None
    if n_latin >= 150 and n_latin > n_cjk:
        lower_ratio = n_lower / n_latin
        if lower_ratio < LOWER_RATIO_MIN:
            garbled = True
            reasons.append(f"乱码指纹(小写占比{lower_ratio:.0%})")

    # 版面复杂度（软信号）
    n_tables = len(page.find_tables().tables)
    n_imgs = sum(1 for block in page.get_text("dict")["blocks"] if block["type"] == 1)

    if coverage_red or n_fffd_ctrl or garbled:
        verdict = "红"
    elif coverage_yellow or n_tables or n_imgs:
        verdict = "黄"
    else:
        verdict = "绿"

    return {
        "chars": n_chars,
        "cjk": n_cjk,
        "latin": n_latin,
        "lower_ratio": lower_ratio,
        "fffd": n_fffd_ctrl,
        "tables": n_tables,
        "imgs": n_imgs,
        "verdict": verdict,
        "reasons": "; ".join(reasons),
    }


def assess_pdf(pdf_path: Path, max_pages: int | None = None) -> dict:
    """打开 PDF 逐页取文本并判红黄绿（一次遍历，页号与溯源 page 对齐为 1 起始）。

    返回：
        stats     {"红": n, "黄": n, "绿": n}
        red_pages 红页号列表（1 起始，供 M2 定位走 VLM 的页）
        notes     红页原因摘要（进摄取报告，让人一眼看懂为什么标红）
    """
    doc = pymupdf.open(pdf_path)
    stats = {"红": 0, "黄": 0, "绿": 0}
    red_pages: list[int] = []
    notes: list[str] = []

    for page_index, page in enumerate(doc, start=1):
        if max_pages and page_index > max_pages:
            break
        result = analyze_page(page)
        stats[result["verdict"]] += 1
        if result["verdict"] == "红":
            red_pages.append(page_index)
            if result["reasons"]:
                notes.append(f"P{page_index}: {result['reasons']}")
    doc.close()

    return {"stats": stats, "red_pages": red_pages, "notes": notes}
