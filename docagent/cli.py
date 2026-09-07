"""doc-agent CLI：`python -m docagent.cli ingest <文件或目录...>`

与 FastAPI 端点同构（M1 起 CLI 先行，作开发与回归入口）：
    ingest  摄取一个或多个文档 / 目录（目录取支持的格式，非递归）
报告逐文档打印 + 批次汇总（状态 / 块 / 入库 / 跳过 / 红黄绿 / 耗时 / 降级标注）。

运行示例（项目根）：
    .venv/Scripts/python.exe -m docagent.cli ingest data/samples
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from docagent.ingest.pipeline import run_document

# 支持的摄取格式（与 file_type_detection 一致；旧版 .doc/.xls 在探测层会被拒并提示）
SUPPORTED_SUFFIXES = {".pdf", ".docx", ".doc", ".md", ".txt", ".json", ".xlsx", ".xls", ".pptx"}


def collect_files(paths: list[str]) -> list[Path]:
    """展开参数：文件直收；目录取其内支持的格式（不递归，inbox 平铺语义）。"""
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(p for p in sorted(path.iterdir()) if p.suffix.lower() in SUPPORTED_SUFFIXES)
        elif path.is_file():
            files.append(path)
        else:
            print(f"[跳过] 路径不存在: {raw}")
    return files


def _quality_cell(report: dict) -> str:
    quality = report.get("quality")
    if not quality:
        return "-"
    return f"红{quality['red']}/黄{quality['yellow']}/绿{quality['green']}"


def print_report(report: dict) -> None:
    """单文档报告行（状态/格式/块/入库/跳过/质量/耗时；降级与失败附注）。"""
    status = report["status"]
    mark = {"ok": "OK ", "skipped": "SKIP", "failed": "FAIL"}.get(status, status)
    degraded = " (mock降级)" if report.get("degraded") else ""
    timing = f"{report['elapsed_s']:>5.1f}s"
    line = (
        f"{mark}  {report['source_file']:<32} {str(report.get('format')):<10}"
        f"块{report['chunks']:>4} 入{report['stored']:>4} 跳{report['skipped']:>4}"
        f"  {_quality_cell(report):>10}  {timing}{degraded}"
    )
    print(line)
    if report.get("error"):
        print(f"       原因: {report['error']}")


def cmd_ingest(args: argparse.Namespace) -> int:
    files = collect_files(args.paths)
    if not files:
        print("没有可摄取的文档（支持的格式: pdf/docx/md/txt/json/xlsx/pptx）")
        return 1

    print(f"摄取 {len(files)} 个文档（串行，Send 并行留 M2+）...")
    print("-" * 110)
    reports = []
    batch_start = time.perf_counter()
    for file_path in files:
        report = run_document(file_path)  # 图内已隔离失败，绝不外抛中断批次
        reports.append(report)
        print_report(report)
    batch_elapsed = time.perf_counter() - batch_start
    print("-" * 110)

    total = len(reports)
    ok = sum(1 for r in reports if r["status"] == "ok")
    skipped = sum(1 for r in reports if r["status"] == "skipped")
    failed = sum(1 for r in reports if r["status"] == "failed")
    total_chunks = sum(r["chunks"] for r in reports)
    total_stored = sum(r["stored"] for r in reports)
    total_skipped_blocks = sum(r["skipped"] for r in reports)
    degraded = any(r.get("degraded") for r in reports)

    print(
        f"汇总: {total} 文档 | OK {ok} / SKIP {skipped} / FAIL {failed} | "
        f"块 {total_chunks}（新入 {total_stored} / 命中跳过 {total_skipped_blocks}）| "
        f"批次耗时 {batch_elapsed:.1f}s"
    )
    if degraded:
        print("提示: 本次为 mock 降级向量（未配 DASHSCOPE_API_KEY），仅供链路演示，检索无语义。")
    return 0 if failed == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docagent", description="文档智能 + RAG（doc-agent）CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest_parser = subparsers.add_parser("ingest", help="摄取文档到向量库")
    ingest_parser.add_argument("paths", nargs="+", help="文件或目录路径（可多个）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ingest":
        return cmd_ingest(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
