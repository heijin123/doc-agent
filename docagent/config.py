"""doc-agent 全局配置：路径 / 向量库 / Embedding 提供方。

读取顺序：环境变量优先，`.env`（项目根，未提交）兜底——demo1 经验：
.env 由用户自己管理，代码只做读取，不强制字段对齐。

Embedding 两种提供方（无 Key 也能全链路跑通，摄取报告标注降级）：
    dashscope —— 真实向量（.env 配 DASHSCOPE_API_KEY），推荐
    mock      —— 本地字符 bigram 哈希向量（回归 / 无 Key 演示用，非语义向量）

注意：两 provider 向量维度不同（1024 vs 256），同一 Chroma 库内混用会报
维度错误 → 切换 provider 后需清空 data/chroma_db 重灌（vectorstore 文档同步注明）。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")  # 不存在则静默跳过

# ---- 向量库 ----
# DOCAGENT_CHROMA_DIR 支持测试注入（verify 脚本指向临时目录）
CHROMA_DIR = Path(os.getenv("DOCAGENT_CHROMA_DIR", str(BASE_DIR / "data" / "chroma_db")))
COLLECTION_NAME = "doc_blocks"

# ---- 上传收件目录（API ingest 端点落盘处；同文件重传 = 同名覆盖 + 幂等 skip）----
INBOX_DIR = Path(os.getenv("DOCAGENT_INBOX_DIR", str(BASE_DIR / "data" / "inbox")))

# ---- 图片资源仓库（摄取提取的图文件 + sqlite 登记；不入向量库，只作块的可引用展示）----
# DOCAGENT_IMAGES_DIR / DOCAGENT_IMAGES_DB 支持测试注入（verify 脚本指向临时目录）
IMAGES_DIR = Path(os.getenv("DOCAGENT_IMAGES_DIR", str(BASE_DIR / "data" / "images")))
IMAGES_DB_PATH = Path(os.getenv("DOCAGENT_IMAGES_DB", str(BASE_DIR / "data" / "image_files.db")))

# ---- 切片检查缓存（调试辅助，非入库链路）----
# 切好的块原样落盘一份可读 md（每文档一个文件，覆盖写），摄取后打开即可人工抽查
# 切片质量（跨页表格是否断裂 / 块大小 / page·block_type 标记）；写入失败不影响摄取
CHUNK_CACHE_DIR = Path(os.getenv("DOCAGENT_CHUNK_CACHE_DIR", str(BASE_DIR / "data" / "chunk")))

# ---- Embedding ----
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "dashscope").strip().lower()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
QWEN_EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3")
EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS", "1024"))  # text-embedding-v3 支持维度
MOCK_EMBED_DIMENSIONS = int(os.getenv("MOCK_EMBED_DIMENSIONS", "256"))

# ---- 问答 chat（M3 消费侧：引用溯源答复）----
# 与 embedding/vlm 共用 DASHSCOPE_API_KEY；无 Key 时问答编排降级（见 qa.py）
QWEN_CHAT_MODEL = os.getenv("QWEN_CHAT_MODEL", "qwen-plus")

# ---- VLM 转录（M2 慢路径兜底：质量门红页 → qwen-vl 整页渲染转录）----
# 与 embedding 共用 DASHSCOPE_API_KEY；无 Key 时编排跳过转录（R7 降级不崩）
QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL", "qwen-vl-max")          # 首选：细节转录更全
VLM_FALLBACK_MODEL = os.getenv("VLM_FALLBACK_MODEL", "qwen-vl-plus")  # 失败降级
VLM_PAGE_ZOOM = float(os.getenv("VLM_PAGE_ZOOM", "2.0"))           # 页渲染倍率（≈144dpi）
VLM_MAX_PAGES_PER_DOC = int(os.getenv("VLM_MAX_PAGES_PER_DOC", "20"))  # 成本护栏：单文档转录上限
