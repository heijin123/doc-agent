"""M1 收尾冒烟：FastAPI ingest 端点真实链路（需 .env 配好 DASHSCOPE_API_KEY）。

与 verify_ingest_m1.py（mock + 临时库，纯本地）互补：本脚本走真实 DashScope
向量 + 默认库，验证 API 端点契约：
1. /health 正常
2. 上传 txt → status ok、chunks_created ≥ 1、provider=dashscope、degraded=False
3. 同文件重传 → ok + stored=0 / skipped>0（幂等命中）
4. 上传不可解码二进制 .xyz → 200 + status=failed + error（业务失败非网络错误）

上传文件落 DOCAGENT_INBOX_DIR（默认 data/inbox，同名覆盖语义）。

运行：.venv/Scripts/python.exe scripts/verify_server_ingest.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# 环境前置（import docagent 前）：收件目录指向临时目录，避免污染 data/inbox
_TMP_INBOX = Path(tempfile.mkdtemp(prefix="verify_server_inbox_"))
os.environ["DOCAGENT_INBOX_DIR"] = str(_TMP_INBOX)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from docagent.api.server import app  # noqa: E402

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"

passed_cases: list[str] = []
failed_cases: list[str] = []


def check(condition: bool, case_name: str, detail: str = "") -> None:
    if condition:
        passed_cases.append(case_name)
        print(f"  PASS  {case_name}")
    else:
        failed_cases.append(case_name)
        print(f"  FAIL  {case_name}  {detail}")


client = TestClient(app)

print("=== M1 收尾：ingest 端点真实链路冒烟（真 DashScope 向量）===\n")

print("[1] /health")
health_response = client.get("/health")
check(health_response.status_code == 200 and health_response.json()["status"] == "ok", "health 正常")

print()

print("[2] 上传 sample_notes.txt → 真实摄取")
with (SAMPLES_DIR / "generated" / "sample_notes.txt").open("rb") as handle:
    response = client.post("/api/v1/ingest", files={"file": ("sample_notes.txt", handle, "text/plain")})
body = response.json()
print(f"      {body}")
check(response.status_code == 200, "HTTP 200")
check(body["status"] == "ok", "status=ok", str(body))
check(body["chunks_created"] >= 1, "产生切片", f"chunks={body['chunks_created']}")
check(body["provider"] == "dashscope" and body["degraded"] is False, "真实向量（provider=dashscope 未降级）")

print()

print("[3] 同文件重传 → 幂等 skip")
with (SAMPLES_DIR / "generated" / "sample_notes.txt").open("rb") as handle:
    response2 = client.post("/api/v1/ingest", files={"file": ("sample_notes.txt", handle, "text/plain")})
body2 = response2.json()
print(f"      {body2}")
check(body2["status"] == "ok", "重传 status=ok")
check(body2["stored"] == 0 and body2["skipped"] == body["chunks_created"], "全 skip 未重复入库", str(body2))

print()

print("[4] 上传不可解码二进制 .xyz → 业务失败 200 + status=failed")
bad_bytes = bytes(1024)  # 全 NUL：内容嗅探拒绝（纯文本会被 txt 兜底收下）
response3 = client.post("/api/v1/ingest", files={"file": ("archive.xyz", bad_bytes, "application/octet-stream")})
body3 = response3.json()
print(f"      {body3}")
check(response3.status_code == 200, "业务失败仍 HTTP 200（契约）")
check(body3["status"] == "failed" and body3.get("error"), "status=failed + error 说明", str(body3))

print()

import shutil  # noqa: E402

shutil.rmtree(_TMP_INBOX, ignore_errors=True)
print(f"通过 {len(passed_cases)} / {len(passed_cases) + len(failed_cases)}")
if failed_cases:
    print("失败用例：")
    for case in failed_cases:
        print(f"  - {case}")
    sys.exit(1)
print("ingest 端点真实链路冒烟全绿")
