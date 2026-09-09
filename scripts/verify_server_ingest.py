"""M1 收尾冒烟：FastAPI ingest 端点真实链路（需 .env 配好 DASHSCOPE_API_KEY）。

与 verify_ingest_m1.py（mock + 临时库，纯本地）互补：本脚本走真实 DashScope
向量 + 默认库，验证 API 端点契约（2026-09-09 起为异步摄取）：
1. /health 正常
2. 上传 txt → 立即返回 job_id + status=queued；轮询至 done，result.status=ok、
   chunks_created ≥ 1、provider=dashscope、degraded=False
3. 同文件重传 → 轮询至 done，result.status=ok + stored=0 / skipped>0（幂等命中）
4. 上传不可解码二进制 .xyz → 200 + 立即返回 job_id；轮询至 failed（业务失败非网络错误）

上传文件落 DOCAGENT_INBOX_DIR（默认 data/inbox，同名覆盖语义）。

运行：.venv/Scripts/python.exe scripts/verify_server_ingest.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
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


def wait_job(job_id: str, timeout: float = 180) -> dict | None:
    """轮询摄取任务直到终态（done/failed）或超时。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get("/api/v1/ingest/status", params={"job_id": job_id})
        if resp.status_code == 200:
            body = resp.json()
            if body.get("status") in ("done", "failed"):
                return body
        time.sleep(1.5)
    return None


client = TestClient(app)

print("=== M1 收尾：ingest 端点真实链路冒烟（异步摄取 · 真 DashScope 向量）===\n")

print("[1] /health")
health_response = client.get("/health")
check(health_response.status_code == 200 and health_response.json()["status"] == "ok", "health 正常")

print()

print("[2] 上传 sample_notes.txt → 立即返回 job_id，轮询至完成")
with (SAMPLES_DIR / "generated" / "sample_notes.txt").open("rb") as handle:
    response = client.post("/api/v1/ingest", files={"file": ("sample_notes.txt", handle, "text/plain")})
body = response.json()
print(f"      上传响应: {body}")
check(response.status_code == 200, "HTTP 200")
check(body.get("job_id") and body.get("status") == "queued", "立即返回 job_id + status=queued（不阻塞）", str(body))
job = wait_job(body["job_id"])
check(job is not None, "轮询拿到终态")
result = (job or {}).get("result") or {}
print(f"      终态: status={job.get('status') if job else None} result={result}")
check(job is not None and job.get("status") == "done", "任务 done")
check(result.get("status") == "ok", "result.status=ok", str(result))
check(result.get("chunks_created", 0) >= 1, "产生切片", f"chunks={result.get('chunks_created')}")
check(result.get("provider") == "dashscope" and result.get("degraded") is False, "真实向量（provider=dashscope 未降级）")
first_chunks = result.get("chunks_created", 0)

print()

print("[3] 同文件重传 → 幂等 skip")
with (SAMPLES_DIR / "generated" / "sample_notes.txt").open("rb") as handle:
    response2 = client.post("/api/v1/ingest", files={"file": ("sample_notes.txt", handle, "text/plain")})
body2 = response2.json()
check(body2.get("job_id") and body2.get("status") == "queued", "重传也立即返回 job_id", str(body2))
job2 = wait_job(body2["job_id"])
result2 = (job2 or {}).get("result") or {}
print(f"      终态: result={result2}")
check(job2 is not None and result2.get("status") == "ok", "重传 result.status=ok")
check(result2.get("stored") == 0 and result2.get("skipped") == first_chunks, "全 skip 未重复入库", str(result2))

print()

print("[4] 上传不可解码二进制 .xyz → 业务失败仍在后台 job 中体现")
bad_bytes = bytes(1024)  # 全 NUL：内容嗅探拒绝（纯文本会被 txt 兜底收下）
response3 = client.post("/api/v1/ingest", files={"file": ("archive.xyz", bad_bytes, "application/octet-stream")})
body3 = response3.json()
print(f"      上传响应: {body3}")
check(response3.status_code == 200, "上传仍 HTTP 200（契约）")
check(body3.get("job_id") and body3.get("status") == "queued", "落盘成功即返回 job_id（失败在后台）", str(body3))
job3 = wait_job(body3["job_id"])
result3 = (job3 or {}).get("result") or {}
print(f"      终态: status={job3.get('status') if job3 else None} result={result3}")
check(job3 is not None and job3.get("status") == "failed", "任务 failed")
check(result3.get("error"), "result 含 error 说明", str(result3))

print()

import shutil  # noqa: E402

shutil.rmtree(_TMP_INBOX, ignore_errors=True)
print(f"通过 {len(passed_cases)} / {len(passed_cases) + len(failed_cases)}")
if failed_cases:
    print("失败用例：")
    for case in failed_cases:
        print(f"  - {case}")
    sys.exit(1)
print("ingest 端点真实链路冒烟全绿（异步摄取）")
