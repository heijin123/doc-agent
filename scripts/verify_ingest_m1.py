"""M1 摄取流水线编排验证（mock embedding + 临时 Chroma，无需 Key / Redis）。

覆盖 M1 过闸"单文档入库 + 报告雏形"：
1. 真实 3 PDF：质量门接入生效 —— react paper P2 坏字体页 → 红 ≥ 1（活体演示料）；
   报告字段齐全（format/detected_by/quality/stored/timings）
2. 混合格式（generated 6 件）：探测→切块→落库全 ok，stored == chunks
3. 幂等：全量重跑一遍 → 每文档 skipped == chunks、stored == 0、status = skipped
4. 坏文件隔离：损坏 .pdf + 未知 .xyz 混批 → 各自 failed，正常文档不受影响
5. 降级标注：mock provider → degraded = True（R7 无 Key 不崩的落点）

运行：.venv/Scripts/python.exe scripts/verify_ingest_m1.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

# ---- 环境前置（必须在 import docagent 之前）：mock 向量 + 临时 Chroma/图片目录 ----
_TMP_DIR = Path(tempfile.mkdtemp(prefix="verify_ingest_m1_"))
os.environ["DOCAGENT_CHROMA_DIR"] = str(_TMP_DIR / "chroma")
os.environ["DOCAGENT_IMAGES_DIR"] = str(_TMP_DIR / "images")
os.environ["DOCAGENT_IMAGES_DB"] = str(_TMP_DIR / "image_files.db")
os.environ["EMBEDDING_PROVIDER"] = "mock"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docagent.ingest.pipeline import run_document  # noqa: E402

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"
REAL_PDFS = [
    SAMPLES_DIR / "1_hermes_agent_manual_cn.pdf",   # 中文手册 83 页（封面整页图）
    SAMPLES_DIR / "2_react_paper_en.pdf",           # 英文论文 33 页（P2 已知坏字体）
    SAMPLES_DIR / "3_invoice_table_cn.pdf",         # 电子发票 1 页（表格型）
]
GENERATED_DIR = SAMPLES_DIR / "generated"
MIXED_FILES = sorted(GENERATED_DIR.iterdir())  # docx/md/pptx/txt/xlsx/json 各 1

passed_cases: list[str] = []
failed_cases: list[str] = []


def check(condition: bool, case_name: str, detail: str = "") -> None:
    if condition:
        passed_cases.append(case_name)
        print(f"  PASS  {case_name}")
    else:
        failed_cases.append(case_name)
        print(f"  FAIL  {case_name}  {detail}")


def report_summary(report: dict) -> str:
    quality = report.get("quality") or {}
    return (
        f"{report['source_file']} [{report['status']}] "
        f"fmt={report.get('format')} 块={report['chunks']} "
        f"入={report['stored']} 跳={report['skipped']} "
        f"质(红{quality.get('red', '-')}/黄{quality.get('yellow', '-')}/绿{quality.get('green', '-')})"
    )


def ingest_batch(paths: list[Path]) -> list[dict]:
    """批次直跑 run_document（图内已隔离失败，绝不外抛）。"""
    return [run_document(path) for path in paths]


print("=== M1 摄取流水线编排验证（mock embedding / 临时 Chroma）===\n")

# ---- 1. 真实 3 PDF：质量门接入生效 + 报告字段齐全 ----
print("[1] 真实 3 PDF（质量门活体验证）")
pdf_reports = ingest_batch(REAL_PDFS)
for report in pdf_reports:
    print(f"      {report_summary(report)}")

react_report = next(r for r in pdf_reports if "react_paper" in r["source_file"])
hermes_report = next(r for r in pdf_reports if "hermes" in r["source_file"])
invoice_report = next(r for r in pdf_reports if "invoice" in r["source_file"])

for report in pdf_reports:
    check(report["status"] == "ok", f"{report['source_file']} 状态 ok", str(report.get("error")))
    check(report["chunks"] > 0, f"{report['source_file']} 产出块", f"chunks={report['chunks']}")
    check(
        report["format"] == "pdf" and report["detected_by"] in ("extension", "sniff"),
        f"{report['source_file']} 探测字段齐全",
        f"format={report.get('format')} by={report.get('detected_by')}",
    )
    check(
        report["stored"] == report["chunks"] and report["skipped"] == 0,
        f"{report['source_file']} 首跑全量入库",
        f"stored={report['stored']} chunks={report['chunks']}",
    )

check(
    react_report["quality"]["red"] >= 1,
    "react paper 质量门判出坏字体红页（P2 活体）",
    f"red={react_report['quality'].get('red')}",
)
for report in pdf_reports:
    quality = report["quality"]
    check(
        quality["red"] + quality["yellow"] + quality["green"] > 0,
        f"{report['source_file']} 质量门统计生效",
        f"红{quality['red']}/黄{quality['yellow']}/绿{quality['green']}",
    )

print()

# ---- 2. 混合格式（generated 6 件）全链路 ----
print("[2] 混合格式 6 件（docx/md/pptx/txt/xlsx/json）")
mixed_reports = ingest_batch(MIXED_FILES)
for report in mixed_reports:
    print(f"      {report_summary(report)}")
    check(report["status"] == "ok", f"{report['source_file']} 状态 ok", str(report.get("error")))
    check(report["chunks"] > 0, f"{report['source_file']} 产出块", f"chunks={report['chunks']}")
    check(
        report["stored"] == report["chunks"] and report["skipped"] == 0,
        f"{report['source_file']} 首跑全量入库",
        f"stored={report['stored']} chunks={report['chunks']}",
    )

print()

# ---- 3. 幂等：全量重跑 → 每文档全 skip ----
print("[3] 幂等重跑（全量 9 文档 → 应全 skip）")
second_pass = ingest_batch(REAL_PDFS + MIXED_FILES)
for report in second_pass:
    check(
        report["status"] == "skipped" and report["skipped"] == report["chunks"] and report["stored"] == 0,
        f"{report['source_file']} 二次全 skip",
        f"status={report['status']} skipped={report['skipped']} stored={report['stored']}",
    )
print(f"      重跑 9 文档全部命中跳过（共 {sum(r['skipped'] for r in second_pass)} 块未重复入库）")

print()

# ---- 4. 坏文件隔离（不中断批次） ----
print("[4] 坏文件隔离")
broken_pdf = _TMP_DIR / "broken.pdf"
broken_pdf.write_bytes(b"this is not a real pdf file at all" * 20)  # 扩展名 pdf 但内容损坏
# 全 NUL 二进制：utf-8/gbk 解码后可打印占比 < 0.8 → 内容嗅探拒绝（纯文本会被 txt 兜底收下）
unknown_file = _TMP_DIR / "archive.xyz"
unknown_file.write_bytes(bytes(1024))

bad_reports = ingest_batch([broken_pdf, unknown_file, SAMPLES_DIR / "3_invoice_table_cn.pdf"])
broken_report = bad_reports[0]
unknown_report = bad_reports[1]
survivor_report = bad_reports[2]
check(
    broken_report["status"] == "failed" and "PDF" in (broken_report.get("error") or ""),
    "损坏 pdf 判 failed（parse 兜住）",
    str(broken_report.get("error")),
)
check(
    unknown_report["status"] == "failed",
    "未知 .xyz 判 failed（detect 兜住）",
    str(unknown_report.get("error")),
)
check(
    survivor_report["status"] == "skipped",
    "同批正常文档不受影响（幂等 skip）",
    report_summary(survivor_report),
)

print()

# ---- 5. 降级标注 ----
print("[5] 降级标注（mock provider）")
check(
    all(r.get("provider") == "mock" and r.get("degraded") is True for r in pdf_reports),
    "mock 提供方 + degraded=True 落报告",
    f"provider={pdf_reports[0].get('provider')} degraded={pdf_reports[0].get('degraded')}",
)

print()

# ---- 收尾 ----
shutil.rmtree(_TMP_DIR, ignore_errors=True)
print(f"通过 {len(passed_cases)} / {len(passed_cases) + len(failed_cases)}")
if failed_cases:
    print("失败用例：")
    for case in failed_cases:
        print(f"  - {case}")
    sys.exit(1)
print("M1 摄取编排验证全绿")
