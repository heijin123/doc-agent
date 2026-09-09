"""PDF 表格链/超大表切片回归（2026-09-09 第二轮重构）。

覆盖三个真实语料场景：
1. 中金辐照（2 页短表链）：跨页提案表合并成一块，3.00/3.01-3.03 完整连续
2. 天力锂能年报（页内大表 + 长链）：表格原子化（行不腰斩）、行切段带列名表头、
   无 1-2 字纯数字页码块、table/page 标记不因 pop 错位
3. 奥比中光公告（表单式多页）：每股东独立块、内容自足

运行：.venv/Scripts/python.exe scripts/verify_table_split.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docagent.ingest.chunk_pdf import chunk_pdf_file  # noqa: E402

INBOX = Path(__file__).resolve().parents[1] / "data" / "inbox"
SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"

passed: list[str] = []
failed: list[str] = []


def check(condition: bool, name: str, detail: str = "") -> None:
    if condition:
        passed.append(name)
        print(f"  PASS  {name}")
    else:
        failed.append(name)
        print(f"  FAIL  {name}  {detail}")


def chunks_of(rel_dir: Path, filename: str) -> list:
    path = rel_dir / filename
    chunks, _ = chunk_pdf_file(path, path.name, path.stem)
    return chunks


# ---------------- 1. 中金辐照：2 页短表链合并 ----------------
print("== 1. 中金辐照 2 页短表链（合并完整性）")
chunks = chunks_of(INBOX, "中金辐照：中金辐照股份有限公司关于召开2026年第三次临时股东会的通知（更正后）.pdf")
proposal_block = next(c.text for c in chunks if "提案编码 | 提案名称" in c.text)
check("2.03" in proposal_block, "短表合并-2.03 提案在块内")
check("3.03" in proposal_block, "短表合并-3.03（跨页续行）并入同块")
check("3.01" in proposal_block and "3.02" in proposal_block, "短表合并-3.01/3.02 同在", "无 3.01/3.02")
table_blocks = [c for c in chunks if c.metadata["block_type"] == "table"]
check(len(table_blocks) >= 4, "短表合并-表格被标记为 table 块", f"table 块仅 {len(table_blocks)}")
check(not any(c.metadata["block_type"] == "table" and len(c.text.strip()) <= 4 for c in chunks),
      "短表合并-无纯数字页码被标 table")

# ---------------- 2. 天力锂能：页内大表 + 长链锚 ----------------
print("== 2. 天力锂能（页内大表原子化 + 长链表头锚）")
chunks = chunks_of(SAMPLES, "天力锂能：天力锂能集团股份有限公司2024年年度报告（更正后）.pdf")
research_blocks = [c.text for c in chunks
                   if c.metadata["block_type"] == "table" and c.text.startswith("主要研发项目名称")]
check(len(research_blocks) >= 3, "长链-研发表跨页成多块且每块带表头锚",
      f"{len(research_blocks)} 块均应以列名开头")
check(all("拟达到的目标" in block.splitlines()[0] for block in research_blocks),
      "长链-研发表续块首行 = 列名（自包含）")
# 表格行原子性：任意块内若出现"主要研发项目名称"表，不应把单元格行按句子腰斩
broken = [c.text for c in chunks
          if c.metadata["block_type"] == "table"
          and "4.4V6 系单晶三元正" in c.text and "针对EV、HEV" not in c.text]
check(not broken, "行原子-研发表首数据行单元格未被腰斩", f"{len(broken)} 个可疑块")
# 无页码垃圾块（纯数字 1-4 位独立成块）
page_only = [c for c in chunks if c.text.strip().isdigit() and len(c.text.strip()) <= 4]
check(not page_only, "无页码-纯数字页码块已剔除", f"残留 {len(page_only)}: {[c.text for c in page_only[:5]]}")
# table/page 标记不错位：供应商表（序号|供应商名称 开头）应为 table
supplier_blocks = [c for c in chunks
                   if c.text.startswith("序号 | 供应商名称") and c.metadata["block_type"] == "table"]
check(len(supplier_blocks) >= 1, "标记-供应商明细表标为 table 块", f"{len(supplier_blocks)} 个")
# 表格块不带 1-2 字噪音（页码/标题被误标 table 的回归防线）
noise_tables = [c for c in chunks if c.metadata["block_type"] == "table" and len(c.text.strip()) <= 6
                and not c.text.startswith("主要研发")]
check(not noise_tables, "标记-无小碎片被误标 table", f"{len(noise_tables)}: {[c.text[:20] for c in noise_tables[:5]]}")

# ---------------- 3. 奥比中光：表单式跨页（每行自含键名） ----------------
print("== 3. 奥比中光（表单式跨 5 页，每股东独立块）")
chunks = chunks_of(SAMPLES, "奥比中光：控股股东、实际控制人及其一致行动人、部分董事、高级管理人员减持计划时间届满暨减持股份结果公告.pdf")
holder_blocks = [c.text for c in chunks if "减持计划首次披露日期" in c.text]
check(len(holder_blocks) >= 9, "表单-各股东减持明细独立成块", f"{len(holder_blocks)} 块")
check(all("股东名称" in b and "减持数量" in b for b in holder_blocks),
      "表单-每块自含键名（字段名：值 语义自足）")
# 无跨页误并：不应出现一个块含两个不同股东名（除合并表头行外）
merged = [b for b in holder_blocks if b.count("股东名称") > 1 and "股东名称 | 黄源浩" not in b.split("股东名称", 1)[0]]
check(not merged, "表单-无跨页误并不同股东", f"{len(merged)} 块")

print()
print(f"==== 通过 {len(passed)} / {len(passed) + len(failed)} ====")
sys.exit(1 if failed else 0)
