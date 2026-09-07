# DocAgent — 文档智能 + RAG 工程

> 面向 **RAG 摄取端（Ingestion）** 的端到端工程：多格式文档（PDF / Word / Excel / PPT / Markdown / JSON / TXT）→ 探测 → 解析 → 质量门 → 定制切片 → 幂等落库 → 可检索知识库。
> 与问答端（Embedding → Rerank → LLM）配套，构成完整的 **文档智能问答工作台**。

```
                    ┌─────────────────────────────────────────────┐
                    │            doc-agent 工作台（Web + API）      │
                    └─────────────────────────────────────────────┘
       上传/摄取                                          问答（M3）
          │                                                 ▲
          ▼                                                 │
┌────────────────────────────┐                   ┌──────────────────────────┐
│    摄取流水线（LangGraph）   │                   │    检索链路（规划中）       │
│  detect → parse → gate     │                   │  向量召回 → RRF → Rerank  │
│  → chunk → store           │                   │  → 引用溯源 → 编排答复      │
└────────────────────────────┘                   └──────────────────────────┘
          │ 幂等 upsert（md5 diff）
          ▼
   ┌──────────────┐
   │ Chroma 向量库 │  id = {doc_id}:{block_seq}
   └──────────────┘
```

---

## 目录

1. [为什么做这个项目](#1-为什么做这个项目)
2. [核心特性](#2-核心特性)
3. [技术栈](#3-技术栈)
4. [架构设计](#4-架构设计)
5. [快速开始](#5-快速开始)
6. [目录结构与文件职责](#6-目录结构与文件职责)
7. [摄取管线设计细节](#7-摄取管线设计细节)
8. [测试与验证](#8-测试与验证)
9. [里程碑 Roadmap](#9-里程碑-roadmap)
10. [配套文档](#10-配套文档)
11. [设计原则](#11-设计原则)

---

## 1. 为什么做这个项目

RAG 检索质量的天花板在**摄取（数据侧）**，但市面上多数 demo 都聚焦检索端（向量化、混合检索、重排），摄取端——**格式识别、异构解析、切块策略、质量把关、增量落库**——才是真正的工程深水区，也往往是生产环境里最先翻车的地方。

本项目的核心命题（区别于常见 RAG demo）：

- **分格式定制切块**：表格按"一行一条记录"切、PPT/PDF 按"一页一个主题"切、Markdown/Word 按"标题层级"切——不同文档形态用不同的知识单元切法，而不是所有格式套同一个字符窗口。
- **质量门前置判级**：PDF 文本层不可用时（扫描页、坏字体、纯图页），先用零成本规则信号标红（红/黄/绿三档），为"快路径够用就走、不够用再上慢路径（OCR / VLM 转录 / MinerU）"的分级路由提供依据。
- **摄取编排与检索编排同一框架**：LangGraph 既驱动消费侧对话 Agent，也驱动摄取侧确定性流水线（状态机 + 条件边），同一框架两种用法。
- **无 Key 也能全链路跑通**：未配置模型 API Key 时自动降级为本地 mock 向量，摄取与验证照常运行，报告明确标注降级状态——不把"能不能跑"押在密钥上。

需求全文见 [docs/requirements.md](docs/requirements.md)（含取舍决策与验收清单）。

## 2. 核心特性

| 能力 | 说明 |
|---|---|
| **7 格式探测** | 扩展名主判 + 内容嗅探兜底（`%PDF` 头 / ZIP 容器内目录判 Office 新版 / OLE 头拒旧版 / 文本解码判 JSON·MD·TXT），不支持格式给出明确错误与补救提示 |
| **4 类定制切片** | 标题结构切（md/docx）、记录切（excel/json）、页切（pdf/ppt）、段落切（txt）；块统一携带溯源字段 `doc_id / page / section / block_type / block_seq` |
| **PDF 质量门** | 逐页红/黄/绿三档判级，四类可解释信号、零 LLM 成本；坏字体页（乱码指纹）、纯图页、超长页自动识别 |
| **LangGraph 摄取编排** | 5 节点状态机流水线（detect → parse → gate → chunk → store），节点内异常一律转为报告而非外抛——单文档失败不拖垮整批 |
| **md5 增量幂等落库** | 块级内容指纹 diff：命中跳过（不重嵌、不重计费）、变化才 upsert；重复摄取同一语料 258 块全 skip 零重复 |
| **Embedding 双通道** | DashScope 真实向量（1024 维，分批 ≤10 + 指数退避重试）/ 本地 mock 哈希向量（无 Key 降级）；两通道互切需清库重灌 |
| **同构双入口** | CLI（`python -m docagent.cli ingest`）与 Web 上传（`POST /api/v1/ingest`）调用同一编排图，报告格式一致 |
| **Web 工作台骨架** | 问答页（M3 接真链路）+ 上传页（多文件/文件夹，已接真实摄取管线） |

## 3. 技术栈

| 层 | 选型 |
|---|---|
| 语言 | Python ≥ 3.13（uv 管理依赖，清华镜像固化于 pyproject） |
| 摄取编排 | LangGraph（StateGraph 状态机，与对话 Agent 同框架） |
| 文档解析 | PyMuPDF（PDF 快路径）/ python-docx / python-pptx / pandas+openpyxl |
| 向量库 | Chroma（持久化，业务主键 `{doc_id}:{block_seq}`） |
| Embedding | DashScope text-embedding-v3（OpenAI 兼容接口）/ mock 本地降级 |
| API | FastAPI + uvicorn，静态页托管 + JSON 接口 |
| 前端 | 原生 HTML/CSS/JS（无构建步骤，两页面） |

## 4. 架构设计

### 4.1 摄取流水线（每文档一条 LangGraph 图）

```
detect ──▶ parse ──▶ gate ──▶ chunk ──▶ store
   │          │        │         │         │
   └──failed──┴──failed┴──failed─┴──failed─┴──▶ END（报告带 error，批次不中断）
```

| 节点 | 职责 | 关键点 |
|---|---|---|
| `detect` | 格式识别 | 复用探测层；不支持格式 → FAIL 出口（报告带原因） |
| `parse` | PDF 一次打开：逐页取文本 + 质量门判级 | 不把 pymupdf 页面对象塞进 state（M2 加 checkpoint 不炸）；非 PDF 空转 |
| `gate` | 红/黄/绿统计整理进报告 | M1 只判级记录；**M2 的「红页 → VLM 转录」条件边锚点挂在这里** |
| `chunk` | 分格式定制切片 | 一行调用纯函数 `chunk_document()`，切片层零改动 |
| `store` | 幂等落库 | md5 diff → 命中 skip / 变化块才 embedding + upsert |

**为什么摄取用 LangGraph（而不是脚本）**：摄取是确定性数据流水线，没有模型自主决策循环——上编排不是为了"智能"，而是借状态机外壳组织步骤：节点 = 步骤、条件边 = 失败/分支路由、state = 逐步累积的逐文档报告。将来文档粒度并行（每文档一个 Send）与断点续跑（checkpoint）是白捡的扩展点。

**分层纪律**：内层切块器 / 质量门 / 落库全部是**纯函数**（可独立测试、不感知 graph），编排只是薄外壳——M2 挂 VLM 条件边时内层仍不动。

### 4.2 Embedding 降级设计

```
EMBEDDING_PROVIDER=dashscope（默认）     未配 DASHSCOPE_API_KEY
        │                                       │
        ▼                                       ▼
  DashScope text-embedding-v3            mock：本地字符 bigram 哈希向量
  （分批 ≤10 条 + 3 次指数退避重试）        （非语义，仅链路回归/演示用）
```

摄取报告与 API 响应均标注 `provider` 与 `degraded`，降级不掩盖。

> ⚠️ 两 provider 向量维度不同（1024 vs 256），**同一 Chroma 库内混用会报维度错误**——切换 provider 后需清空 `data/chroma_db` 重灌。

## 5. 快速开始

### 5.1 环境与安装

需要 Python ≥ 3.13 与 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/heijin123/doc-agent.git
cd doc-agent
uv sync                      # 依赖已固化：清华镜像 + link-mode=copy（见 pyproject.toml）
```

### 5.2 配置（可选）

想用真实向量，在项目根建 `.env`（已被 .gitignore 忽略，不会误提交）：

```bash
# .env
DASHSCOPE_API_KEY=sk-xxxx            # DashScope（阿里云百炼）API Key
EMBEDDING_PROVIDER=dashscope         # dashscope（默认）| mock
# 以下均有默认值，按需覆盖：
# DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# QWEN_EMBEDDING_MODEL=text-embedding-v3
# EMBED_DIMENSIONS=1024
```

不配 Key 也能跑：自动降级 mock 向量，全链路与验证脚本照常工作（报告标注 `degraded=True`）。

### 5.3 CLI 摄取

```bash
# Windows
.venv/Scripts/python.exe -m docagent.cli ingest data/samples/generated
# macOS / Linux
.venv/bin/python -m docagent.cli ingest data/samples/generated
```

输出逐文档报告（表格）：格式 / 页数 / 质量门红黄绿 / 切块 / 入库-跳过 / 耗时 / 降级标注。已入库文档重跑 → 全 skip（幂等）。

### 5.4 启动 Web 工作台

```bash
.venv/Scripts/python.exe -m uvicorn docagent.api.server:app --port 8001
# 打开 http://127.0.0.1:8001/  → 问答页（当前为演示占位）
# 打开 http://127.0.0.1:8001/upload → 文档上传页（多文件，接真实摄取管线）
```

### 5.5 API

| 端点 | 说明 |
|---|---|
| `GET /health` | 存活检查（含 `chat_demo_mode` 状态） |
| `POST /api/v1/ingest` | 单文件上传 → 真实摄取管线，返回逐文档报告 |
| `POST /api/v1/chat` | 问答（当前演示占位，M3 接真实链路） |

```bash
curl -F "file=@data/samples/generated/sample_notes.txt" http://127.0.0.1:8001/api/v1/ingest
```

```json
{
  "filename": "sample_notes.txt",
  "status": "ok",
  "pages_parsed": null,
  "chunks_created": 4,
  "stored": 4,
  "skipped": 0,
  "format": "txt",
  "provider": "dashscope",
  "degraded": false,
  "error": null,
  "elapsed_s": 1.32
}
```

> 契约约定：**业务失败也返回 HTTP 200 + `status=failed`**（网络错误才非 200），前端据此区分「文档没解析成功」与「服务连不上」。

### 5.6 演示语料说明

- `data/samples/generated/`（md/txt/json/docx/xlsx/pptx）由 `scripts/make_demo_samples.py` 生成，已随仓库提交，可直接摄取。
- `data/samples/*.pdf`（真实 PDF，含 83 页中文产品手册、英文论文、A 股公告年报等）**均不随仓库提交**，清单与演示价值见 `data/samples/README.md`。
- 注意：部分验证脚本的 PDF 用例依赖这些未入库的真实 PDF，clone 后需自行放置再跑（见 [§8 测试与验证](#8-测试与验证)）。

## 6. 目录结构与文件职责

```
doc-agent/
├── pyproject.toml                 # 依赖与 uv 配置（清华镜像、link-mode=copy）
├── README.md                      # 本文件
├── .env                           # 本地密钥（gitignore，不入库）
├── .gitignore
│
├── docagent/                      # 主包
│   ├── config.py                  # 全局配置：路径 / 向量库 / Embedding 提供方
│   │                              #   环境变量优先、.env 兜底；provider 与维度说明
│   │
│   ├── ingest/                    # ── 摄取侧（本 demo 的核心增量）──
│   │   ├── file_type_detection.py # 格式探测：DocumentType 枚举；扩展名主判 +
│   │   │                          #   内容嗅探兜底；UnsupportedFileError 带补救提示
│   │   ├── chunk_models.py        # 块统一数据结构（DocumentChunk / ChunkingResult）、
│   │   │                          #   块大小常量（900/1200/80）、通用二次切/合块工具
│   │   ├── chunking.py            # 切块分发入口 chunk_document(path)：探测 → 按格式
│   │   │                          #   路由 → 统一补溯源元数据（编排只认这一个函数）
│   │   ├── chunk_markdown.py      # md/docx 共用：标题结构切（标题链→section，
│   │   │                          #   代码围栏内 # 不误判；含同题块合并坑的规避）
│   │   ├── chunk_plain_text.py    # txt：段落切（不复用 md——txt 里 # 是普通文本）
│   │   ├── chunk_pdf.py           # pdf：一页一块（超长按段/句二次切；纯图页跳过）
│   │   ├── chunk_docx.py          # docx：正文元素归一化为 markdown 流（表格→管道表）
│   │   │                          #   后复用标题结构切
│   │   ├── chunk_excel.py         # xlsx：一行一条记录（sheet→section、行号→page）
│   │   ├── chunk_json.py          # json：按顶层形态切（数组元素/顶层键/整块兜底），
│   │   │                          #   键路径扁平化为 "a.b：值" 文本
│   │   ├── chunk_ppt.py           # pptx：一页一块（标题→section；纯图页跳过）
│   │   ├── pdf_quality.py         # PDF 质量门：逐页红/黄/绿判级（四类信号、零 LLM）
│   │   └── pipeline.py            # ★ LangGraph 摄取编排：IngestState + 5 节点
│   │                              #   detect→parse→gate→chunk→store + 每节点后
│   │                              #   failed 条件边 + run_document() 报告
│   │
│   ├── vectorstore/               # ── 向量侧 ──
│   │   ├── embedding.py           # embedding 提供方：dashscope（分批 10 + 指数退避
│   │   │                          #   重试）/ mock 降级；对外 info() / embed_texts()
│   │   └── store.py               # Chroma 封装：get_collection() /
│   │                              #   upsert_doc_chunks()——md5 幂等 diff→skip/upsert
│   │
│   ├── api/                       # ── 服务侧 ──
│   │   ├── server.py              # FastAPI：页面托管 + /health + POST /api/v1/ingest
│   │   │                          #   （真实管线）+ /api/v1/chat（占位，M3 接真链路）
│   │   └── static/
│   │       ├── chat.html          # 问答页（骨架，M3 接检索与引用展示）
│   │       ├── upload.html        # 上传页（多文件/文件夹，已接 ingest 契约）
│   │       └── style.css          # 两页共用样式
│   │
│   └── cli.py                     # CLI 入口：`python -m docagent.cli ingest <paths|dir>`
│                                  #   与 API 同构（同一 run_document），报告表格输出
│
├── scripts/                       # 生成与验证脚本（产品代码零依赖，可独立跑）
│   ├── make_demo_samples.py       # 生成 6 件演示语料（docx/xlsx/pptx/md/txt/json）
│   ├── verify_chunking.py         # 探测+切块层验证：77 断言全绿
│   ├── verify_ingest_m1.py        # 摄取编排 M1 验证：47 断言全绿（质量门生效/
│   │                              #   混合格式/幂等重跑/坏文件隔离/降级标注）
│   └── verify_server_ingest.py    # server ingest 端点真实链路冒烟：9/9
│
├── docs/
│   ├── requirements.md            # ★ 需求文档：背景/范围/取舍决策 D1-D6/验收清单/里程碑
│   └── coding_standards.md        # 开发守则八条（防过度设计、人类化命名等）
│
└── data/                          # 运行时数据（chroma_db/inbox 均 gitignore）
    └── samples/                   # 演示语料（generated/ 入库；*.pdf 见 README）
```

## 7. 摄取管线设计细节

### 7.1 格式探测（两类信号）

| 信号 | 机制 |
|---|---|
| 扩展名（主判） | 命中即信任，最快路径 |
| 内容嗅探（兜底） | 扩展名缺失/未知/与内容不符时：`%PDF` 头；ZIP 容器（PK）内目录判 docx/xlsx/pptx；OLE 头（D0CF11E0）明确拒绝旧版 Office 并提示转存；文本解码后 JSON 全量解析 → MD 行首 `#` → TXT |

**设计语义**：未知扩展名的可解码文本 = 当 TXT 收下（不是 bug）；要触发"无法识别拒绝"需不可解码二进制。

### 7.2 分格式切片策略（4 类策略 × 7 格式）

| 格式 | 策略 | 知识单元 | 关键溯源 |
|---|---|---|---|
| Markdown / Word | 标题结构切 | 一个标题章节 | `section` = 标题链（"第 2 章 / 2.1 安装"） |
| Excel / JSON | 记录切 | 一行记录 / 一个数组元素 | `section` = sheet 名或 `$[N]` 路径 |
| PDF / PPT | 页切 | 一页 = 一个主题 | `page` = 页码（溯源最常用定位） |
| TXT | 段落切 | 一个段落（空行分隔） | 无章节概念，不强造 section |

通用约束：块长目标 900 字符、硬上界 1200（超界二次切 + 80 字符重叠）；所有策略共用 `DocumentChunk` 结构与溯源 schema，切法差异只体现在 metadata 填充上。

**为什么表格不能按字符窗口切**（demo1 实证）：行与列是作者切好的业务边界，按字符窗口切会把一条记录拦腰切断，检索只能命中半条记录。

### 7.3 PDF 质量门（红/黄/绿）

| 档 | 含义 | 处置 |
|---|---|---|
| 红 | 硬伤：文本不可信（坏字体乱码页、纯图扫描页） | 必须走慢路径（OCR / MinerU / **M2 的 VLM 转录**） |
| 黄 | 软提示：文本可用但版面复杂（少文字 / 含表格图片） | 可选 MinerU 提结构 |
| 绿 | 直接放行 | PyMuPDF 产出够用 |

全部信号零 LLM 成本、阈值可调。**M1 只判级记录**（红页照常入块、不越权），红页的转录回填是 M2 的条件边锚点。

### 7.4 幂等落库模型

- 业务主键 `id = {doc_id}:{block_seq}`（杜绝自增序号——内容变化会导致同文档新旧块 id 错位）。
- metadata 存块文本 md5：`coll.get(ids=本地ids)` 查旧 → md5 相同 skip（不 touch、不重嵌、不重计费）/ 不同或不存在 → embedding + upsert（同 id 覆盖）。
- 只 upsert、永不推断删除；删除是业务层职责（增量补录模型，差集可能误删未更新文档）。

## 8. 测试与验证

| 脚本 | 覆盖 | 语料依赖 | 结果 |
|---|---|---|---|
| `verify_chunking.py` | 探测 + 四策略切块（真实 PDF 对照源文件逐块核验） | `generated/` + `3_invoice_table_cn.pdf` | 77/77 |
| `verify_ingest_m1.py` | 质量门活体生效 / 混合格式全 ok / 幂等重跑全 skip / 坏文件隔离不中断 / mock 降级标注 | `generated/` + 3 个真实 PDF（hermes/react/发票） | 47/47 |
| `verify_server_ingest.py` | ingest 端点真实链路：上传 ok / 重传全 skip / 不可解码文件业务失败 | 仅 `generated/sample_notes.txt`（clone 即可跑） | 9/9 |

```bash
.venv/Scripts/python.exe scripts/verify_server_ingest.py   # clone 后可直接跑
.venv/Scripts/python.exe scripts/verify_ingest_m1.py       # 需先放置 3 个真实 PDF
```

> ⚠️ **语料依赖说明**：真实 PDF 未随仓库提交（见 [§5.6](#56-演示语料说明)），文件清单在 `data/samples/README.md`。`verify_ingest_m1.py` / `verify_chunking.py` 的 PDF 用例需将对应文件放回 `data/samples/` 后才能跑通；缺文件时这两脚本会明确报错，不影响 `generated/` 相关用例的判断。

**已知实践结论**（防复发沉淀）：
- mock 全绿 ≠ 真实可跑——mock 直接替换模块函数，不校验协议/网络。真实大批量 embedding 曾触发 DashScope 断连（Connection error），已加**每批 3 次指数退避重试**（覆盖连接错误/超时/429/5xx）。
- 切分结果必须与源文件交叉验证（曾踩 `MarkdownHeaderTextSplitter` 同标题相邻块合并：73 块→20 块，SKU 张冠李戴）。

## 9. 里程碑 Roadmap

| 里程碑 | 内容 | 状态 |
|---|---|---|
| **M1** | 项目骨架 + 探测/分格式切块 + 质量门 + **LangGraph 摄取编排** + CLI/API 双入口 + 幂等落库 | ✅ 完成（47/47 + 9/9） |
| **M2** | gate 条件边挂 **VLM 转录**（红页 → Qwen-VL 转录回填，`block_type=figure_transcript`）；解析双输出去重；报告补 token 成本 | ⏳ 规划 |
| **M3** | 问答链路：知识库检索 → 引用溯源（doc/page/section）→ 编排答复；chat 端点替换占位 | ⏳ 规划 |
| **M4** | 评估体系：检索命中 / 引用溯源正确率 / 成本聚合 | ⏳ 规划 |

设计取舍与待拍板项见 [docs/requirements.md](docs/requirements.md) 的 D1–D6 与 §9。

## 10. 配套文档

- [docs/requirements.md](docs/requirements.md) — 需求文档（背景与目标 / MVP 范围 / 技术选型与资产复用 / 验收清单 / 里程碑 / 开放决策）
- [docs/coding_standards.md](docs/coding_standards.md) — 本仓库开发守则八条（体现工程观，欢迎指正）
- [data/samples/README.md](data/samples/README.md) — 演示语料清单与解析形态说明

## 11. 设计原则

本仓库按八条硬性守则开发（见 `docs/coding_standards.md`），核心几条：

1. **不过度设计**——只写当前需求的最小结构，不预建抽象/配置/接口壳；需求出现再重构。
2. **人类化命名**——变量全名不缩写、语义自解释，读起来像句子。
3. **模块化 + 分层纪律**——内层纯函数可独立测试、不感知编排；编排只是可换的薄外壳。
4. **确定性优先于赌模型**——能规则判定的不请 LLM（质量门零成本信号即一例）。
5. **降级不掩盖、mock 不冒充**——无 Key 可跑但明确标注 degraded，验证脚本同时覆盖 mock 与真实链路。

---

*该项目为个人学习与求职作品，代码与文档会随学习持续迭代。*
