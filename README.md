# DocAgent — 文档智能 + RAG 工程

> 面向 **RAG 摄取端（Ingestion）** 的端到端工程，附带完整的问答与评估闭环：多格式文档（PDF / Word / Excel / PPT / Markdown / JSON / TXT）→ 探测 → 解析 → 质量门 → **红页 VLM 转录** → **图片提取与引用** → 定制切片 → 幂等落库 → 向量检索问答（引用溯源）→ golden 评估。
> 摄取编排与问答编排均用 **LangGraph**（同一框架两种用法），构成可演示的 **文档智能问答工作台**。

```
摄取侧（LangGraph，每文档一条图）
  upload / CLI ─▶ detect ─▶ parse ─▶ gate ─▶ chunk ─▶ store ─▶ save_images
                                       │   （md5 幂等 upsert）   （图片落盘+sqlite 登记）
                                       │
                                       └ 红页（坏字体/纯图页）经条件边转 vlm 转录，
                                         产物 figure_transcript 追加块流尾部；原文不产块
                                         每节点后 failed → END（单文档失败不拖垮批次）
                                       ▼
   ┌─────────────────────────────── Chroma 向量库 ─────────────────────────────────┐
   │  id = {doc_id}:{block_seq}                                                   │
   │  metadata: title/page/section/block_type/image_ids（逗号串）                  │
   │            + doc_date / doc_year / ingested_at（日期溯源，年份过滤用）          │
   │  text: 含 [IMAGE:xxx] 占位（召回后展示层回填原图）                              │
   └───────────────────────────────────────────────────────────────────────────────┘
                                       ▲  search_docs(query_texts, top_k, where)
问答侧（LangGraph）
  question ─▶ route ─（文档问题）▶ search（年份两段式裁决）▶ answer ─▶ citations
                  └（寒暄）▶ greet（规则先行，不请模型）              └ 幽灵引用剔除
                                                            ─▶ sources（引用溯源契约）

评估侧（M4）
  data/golden 20 条 ─▶ 期望块定位（anchor 锚句）─▶ recall@5 / MRR
     ─▶（--answers，需真 Key）引用可回查率 ─▶ 门槛判定 R1–R7 ─▶ data/reports/*.json
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

RAG 检索质量的天花板在**摄取（数据侧）**，但市面上多数 demo 都聚焦检索端（向量化、混合检索、重排），摄取端——**格式识别、异构解析、质量把关、切块策略、增量落库**——才是真正的工程深水区，也往往是生产环境里最先翻车的地方。

本项目的核心命题（区别于常见 RAG demo）：

- **分格式定制切块**：表格按"一行一条记录"切、PPT/PDF 按"一页一个主题"切、Markdown/Word 按"标题层级"切——不同文档形态用不同的知识单元切法，而不是所有格式套同一个字符窗口。中间表示**按形态分流**（D8）：Word/PPT 归一化为 markdown，Excel/JSON 保留结构切（不转 md，保住"一行一记录"与键路径溯源）。
- **质量门前置判级 + 分级慢路径**：PDF 文本层不可用时（坏字体、纯图页），先用零成本规则信号标红（红/黄/绿三档），红页再走 **Qwen-VL 整页转录**（慢路径）回填为文本块——不盲信不可信的原文。
- **图片不丢、但也不进向量库**（D6/D7）：图片提取后原位置 `[IMAGE:xxx]` 占位符随文本入库，图文件落 `data/images/` + sqlite 登记，召回后展示层回填原图；图内文字由 VLM 转录覆盖。
- **摄取编排与问答编排同一框架**：LangGraph 既驱动消费侧问答图，也驱动摄取侧确定性流水线——状态机 + 条件边，同一框架两种用法。
- **召回年份感知**：metadata 携带文档日期（`doc_date`/`doc_year`）与入库时间（`ingested_at`），提问含显式年份时按年份过滤检索（两段式裁决），**防跨年份报表张冠李戴**。
- **评估体系闭环**：golden 集（query → 期望文档+锚句）一键评估检索 recall@5/MRR，可选答案层校验**引用可回查率**，门槛判定 + 报告落盘。
- **无 Key 也能全链路跑通**：未配置模型 API Key 时自动降级为本地 mock 向量，摄取/问答/验证照常运行，报告明确标注降级状态——不把"能不能跑"押在密钥上。

需求全文见 [docs/requirements.md](docs/requirements.md)（含取舍决策 D1–D8 与验收清单 R1–R7）。

## 2. 核心特性

| 能力 | 说明 |
|---|---|
| **7 格式探测** | 扩展名主判 + 内容嗅探兜底（`%PDF` 头 / ZIP 容器内目录判 Office 新版 / OLE 头拒旧版 / 文本解码判 JSON·MD·TXT），不支持格式给出明确错误与补救提示 |
| **4 类定制切片** | 标题结构切（md/docx）、记录切（excel/json）、页切（pdf/ppt）、段落切（txt）；块统一携带溯源键 `doc_id / title / page / section / block_type / block_seq / format / source_file` |
| **PDF 质量门** | 逐页红/黄/绿三档判级，四类可解释信号、零 LLM 成本；坏字体乱码页 / 纯图页 / 超长页自动识别 |
| **红页 VLM 转录** | gate 条件边：PDF 含红页 → Qwen-VL 整页渲染转录（`block_type=figure_transcript` 追加尾部，老块 id 不动）；无 Key / 失败自动降级为原文入库（R7 不崩） |
| **图片引用链路** | 提取图片生成幂等 `image_id` → 原位 `[IMAGE:xxx]` 占位符随块入库 → 图文件 + sqlite 登记 → 召回后展示层回填原图（`GET /api/images/{id}`） |
| **LangGraph 双图** | 摄取图 7 节点（detect→parse→gate→[vlm]→chunk→store→save_images，节点异常转报告不外抛）+ 问答图（route→search/answer→citations） |
| **md5 增量幂等落库** | 块级内容指纹 diff：命中跳过（不重嵌、不重计费）、变化才 upsert；重复摄取全 skip 零重复；永不推断删除 |
| **日期元数据 + 年份召回** | `doc_date`/`doc_year`（文件名解析）与 `ingested_at`（入库时间）进溯源键；提问含年份 → 两段式检索裁决防张冠李戴 |
| **问答引用溯源** | 回答只许引上下文块 `[n]`；越界/幽灵角标代码层剔除（不赌模型）；sources 契约含 block_id → 引用可回查 |
| **Embedding 双通道** | DashScope 真实向量（1024 维，分批 ≤10 + 指数退避重试）/ 本地 mock 哈希向量（无 Key 降级）；两通道互切需清库重灌 |
| **同构三入口** | CLI（`ingest`/`eval`）、FastAPI（`/api/v1/ingest` + `/api/v1/chat`）、Web 页面调用同一份编排与问答代码 |
| **评估体系** | golden 检索 recall@5/MRR + 答案层引用可回查率 + 门槛判定；`data/reports/eval_report_latest.json` 落盘 |
| **VLM 转录真实闭环** | react 论文 5 红页 qwen-vl 转录（28–48s/页），Apple Remote / keyboard / Front Row 判别词全部命中转录块（R3 实证） |
| **切片质量缓存** | 切完即落盘 `data/chunk/{stem}.md`（逐块带 `块N｜chars｜block_type｜page｜section｜doc_date｜image_ids` 标记），人工抽查切片质量；调试辅助产物，写盘失败不影响主链路 |
| **异步摄取（上传/处理解耦）** | `POST /api/v1/ingest` 落盘即返回 `job_id` + ETA，真正切片/入库在后台线程池并发执行，大文件不再阻塞后续上传；`/status` 轮询 + `/jobs` 回查 |
| **持久化并发安全** | Chroma 进程内单例 client + 落库锁串行化（diff 读+embed+upsert 持锁），多 worker 并发摄取无锁冲突（实测 4 并发零错误） |

## 3. 技术栈

| 层 | 选型 |
|---|---|
| 语言 | Python ≥ 3.13（uv 管理依赖，清华镜像固化于 pyproject） |
| 编排 | LangGraph（摄取状态机 + 问答状态机，与对话 Agent 同框架） |
| 文档解析 | PyMuPDF（PDF 快路径，`get_text("dict")` 版面块定位）/ python-docx / python-pptx / pandas+openpyxl |
| 视觉转录 | Qwen-VL-Max（首选，失败降级 Plus），整页渲染 zoom=2 转录红页 |
| 向量库 | Chroma（持久化，业务主键 `{doc_id}:{block_seq}`） |
| Embedding | DashScope text-embedding-v3 / qwen3.7-text-embedding-flash（OpenAI 兼容，1024 维，可切换）/ mock 本地降级 |
| 问答模型 | Qwen-Plus（OpenAI 兼容 chat，usage 统计成本） |
| 图片仓库 | 文件系统 `data/images/{doc_id}/` + sqlite 登记（python 内置 sqlite3，幂等 INSERT OR IGNORE） |
| API / 前端 | FastAPI + uvicorn / 原生 HTML/CSS/JS（无构建步骤，问答+上传两页） |
| 评估 | 自研 eval 引擎 + `data/golden/qa_golden.json`（锚句定位期望块，零人工标注成本） |

## 4. 架构设计

### 4.1 摄取流水线（每文档一条 LangGraph 图）

```
                 ┌─（有红页）▶ vlm ────────┐
detect ─▶ parse ─┤                          ├─▶ chunk ─▶ store ─▶ save_images ─▶ END
                 └─（无红页 / 非 PDF）──────┘
   │       │       │
   └─ failed 条件边：任意节点后 → END（报告带 error，批次不中断，单文档失败不拖垮整批）
```

| 节点 | 职责 | 关键点 |
|---|---|---|
| `detect` | 格式识别 | 复用探测层；不支持格式 → FAIL 出口（报告带原因） |
| `parse` | PDF 一次打开：逐页取文本 + 质量门判级 | 不把 pymupdf 页面对象塞进 state（checkpoint 不炸）；非 PDF 空转 |
| `gate` | 红/黄/绿统计整理进报告 | 红页号记录在案，驱动 vlm 条件路由 |
| `vlm` | 红页整页渲染 → qwen-vl 转录（M2 慢路径） | 无 Key/失败返回空 + note（编排降级为原文入库），绝不外抛 |
| `chunk` | 分格式定制切片 | 纯函数 `chunk_document()`；红页转录文本注入后**追加尾部**、原文不产块（v2 语义，无孤儿 id） |
| `store` | 幂等落库 | md5 diff → 命中 skip / 变化块才 embedding + upsert；注入 `doc_id`/`ingested_at` |
| `save_images` | 切块提取的图片落盘 + sqlite 登记 | 幂等（同 md5 跳过）；失败只记 note 不 fail 整文档 |

**为什么摄取用 LangGraph（而不是脚本）**：摄取是确定性数据流水线，没有模型自主决策循环——上编排不是为了"智能"，而是借状态机外壳组织步骤：节点 = 步骤、条件边 = 失败/红页路由、state = 逐步累积的逐文档报告。将来文档粒度并行（每文档一个 Send）与断点续跑（checkpoint）是白捡的扩展点。

**分层纪律**：内层切块器 / 质量门 / 落库全部是**纯函数**（可独立测试、不感知 graph），编排只是薄外壳——M2 挂 VLM 条件边、M2a 挂 save_images 节点时内层几乎不动。

### 4.2 问答链路（M3，消费侧）

```
        route ──文档问题──▶ search（向量检索 top-k + 年份裁决）──▶ answer（LLM，限引 [1..k]）──▶ citations ─▶ END
          │
          └──问候/闲聊──▶ greet（规则先行不请模型）──▶ END
```

| 环节 | 机制 |
|---|---|
| `route` | 问候/闲聊关键词规则分流（不触发检索，零成本） |
| `search` | 显式 query embedding → Chroma top-k（`search_docs`）；含显式年份 → 两段式裁决（见 [§7.6](#76-日期元数据与召回年份感知)） |
| `answer` | 上下文块按 `[1..k]` 编号完整进 prompt，只许引用可见块；块头带「来源 + 页码 + 文档日期」（模型自查年份错位） |
| `citations` | 幽灵引用剔除（纯函数）：越界角标删标注不删句子；有效引用按编号重组 sources 契约（block_id/document_name/page_number/snippet/image_ids） |

**回答质量护栏**（demo1 回声治理同款哲学）：状态声明与引用一律**代码层校验、不赌模型**；无有效引用时答案追加"未检索到直接出处"提示。

### 4.3 评估体系（M4）

```
data/golden/qa_golden.json（20 条 × 9 份真实语料，v2.0 定稿）
  → validate_golden（缺字段/重复 id/锚句过短）
  → 期望块定位（anchor 归一化子串匹配全库，命中即期望块集合；缺陷用例不计指标分母）
  → 检索层：recall@5（至少一个期望块进 top-k 的用例占比）+ MRR
  →（--answers 可选，需真 Key）答案层：run_question 逐条 → 引用可回查率（sources.block_id 必属检索块）
  → 门槛判定（R4：真向量 recall@5≥0.8；mock 无语义 SKIP 只保链路）→ data/reports/eval_report_latest.json
```

定稿实测（2026-09-09）：**recall@5 = 0.95（19/20）、MRR = 0.882、答案层 20/20 带引用、可回查率 100%**（20 次 qwen-plus，chat tokens 44k）。q17（振华英文年报净利数字问）为**保留的诚实用例**——表格断行文本 + 语义词在 812 块大库内分布广泛致单路向量召回 miss（检索 miss 时模型引用仍可回查，但答案依据弱——实证"可回查≠答对"，忠实性断言核对留 v1.1），作 rerank/表格增强的评估基线，不以改 query 洗指标。

### 4.4 Embedding 降级设计

```
EMBEDDING_PROVIDER=dashscope（默认）     未配 DASHSCOPE_API_KEY
        │                                       │
        ▼                                       ▼
  DashScope text-embedding-v3            mock：本地字符 bigram 哈希向量
  （分批 ≤10 条 + 3 次指数退避重试）        （非语义，仅链路回归/演示用）
```

摄取报告 / API 响应 / 问答结果均标注 `provider` 与 `degraded`，降级不掩盖。问答 chat 与 VLM 转录共用同一把 Key，无 Key 时各自降级（问答仍走检索链路、转录跳过红页保持原文），全链路不崩（R7）。

> ⚠️ 两 provider 向量维度不同（1024 vs 256），**同一 Chroma 库内混用会报维度错误**——切换 provider 后需清空 `data/chroma_db` 重灌。同理，**metadata 结构变更不触发增量重嵌**（md5 只算文本），存量库需用脚本显式升级（见 §9 的一次性迁移脚本）。

### 4.5 异步摄取架构与并发落库（2026-09-09）

上传与处理解耦：原来 `/api/v1/ingest` 在请求线程内同步跑完整管线，大文件（如 220 页年报）会长时间占住请求、串行上传时一个大文件阻塞其后所有文件。改为三层：

```
上传（请求线程，毫秒级）
  └─ 落盘 inbox + 估算页数/ETA ─▶ 立即返回 { job_id, status:"queued",
        pages, eta_seconds, message:"已接收，后台处理中，预计约 N 分钟 后可查询" }
  └─ executor.submit(_process_job)        # ThreadPoolExecutor（DOCAGENT_INGEST_WORKERS 默认 2）
        └─ 后台线程：run_document（探测→解析→质量门→[vlm]→切片→落库→save_images）
              └─ 结果写回内存 job 注册表 _INGEST_JOBS[job_id]

前端：拿 job_id 即翻「处理中」（不确定动画） ─▶ GET /api/v1/ingest/status?job_id= 轮询
      └─ queued/processing/done/failed + waited_s/remaining_s
      └─ GET /api/v1/ingest/jobs 返回全部任务倒序快照（“最近上传记录”面板回查）
```

**设计要点**
- **ETA 估算不阻塞**：pymupdf 速读页数 × `DOCAGENT_INGEST_SEC_PER_PAGE`（默认 2.0s/页）；非 PDF 按文件大小粗估，仅给提示，不卡上传响应。
- **并发落库安全**：多 worker 同时写同一 Chroma sqlite 会触发 `database is locked`——`store.get_collection()` 改为进程内**单例 client**，并把「diff 读 + embed + upsert」用一把进程级锁 `_store_lock` 串行化（解析/切块仍在锁外并行，仅碰 Chroma 与打 DashScope 的共享段串行）。实测 4 并发各 120 块 → 0 错误、幂等正确、库内 480 块。
- **任务注册表内存态**：`_INGEST_JOBS` 为进程内字典，服务重启清空未完成任务（已知边界，当前不持久化；符合不过度设计）。超 200 条自动回收最旧已完结项。
- **契约一致性**：业务失败仍返回 HTTP 200 + `status=failed`（仅落盘失败才非 200），前端据此区分「文档没解析成功」与「服务连不上」。

> ⚠️ 上传页「最近上传记录」面板的 HTML 容器已就位，但**轮询回填的 JS 逻辑尚未接入**（服务端 `/jobs`、`/status` 已可用）；当前页面上传后能在「处理中」停留并展示 ETA，面板回查待补（见 §12 已知局限）。

## 5. 快速开始

### 5.1 环境与安装

需要 Python ≥ 3.13 与 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/heijin123/doc-agent.git
cd doc-agent
uv sync                      # 依赖已固化：清华镜像 + link-mode=copy（见 pyproject.toml）
```

### 5.2 配置（可选）

想用真实向量与真实问答，在项目根建 `.env`（已被 .gitignore 忽略，不会误提交）：

```bash
# .env
DASHSCOPE_API_KEY=sk-xxxx            # DashScope（阿里云百炼）Key：embedding/chat/VLM 共用
EMBEDDING_PROVIDER=dashscope         # dashscope（默认）| mock
# 以下均有默认值，按需覆盖：
# DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# QWEN_EMBEDDING_MODEL=text-embedding-v3
# EMBED_DIMENSIONS=1024
# QWEN_CHAT_MODEL=qwen-plus          # 问答模型（--answers 评估也用它）
# QWEN_VL_MODEL=qwen-vl-max          # 红页转录首选；VLM_FALLBACK_MODEL=qwen-vl-plus
# VLM_MAX_PAGES_PER_DOC=20           # 转录成本护栏（单文档红页上限）
```

不配 Key 也能跑：自动降级 mock 向量，摄取/问答/验证照常工作（报告标注 `degraded=True`；mock 向量无语义，仅演示链路）。

### 5.3 CLI

```bash
# Windows
.venv/Scripts/python.exe -m docagent.cli ingest data/samples      # 摄取文件或目录（串行）
.venv/Scripts/python.exe -m docagent.cli ingest <path> --rebuild  # 重建：先删旧块再灌（VLM 升级后清孤儿）
.venv/Scripts/python.exe -m docagent.cli eval                      # golden 检索评估 recall@5/MRR + 门槛
.venv/Scripts/python.exe -m docagent.cli eval --answers            # 加答案层：引用可回查率（需真 Key）
# macOS / Linux 把 .venv/Scripts/python.exe 换成 .venv/bin/python
```

输出逐文档报告（表格）：状态 / 格式 / 切块 / 入库-跳过 / 红黄绿 / 耗时 / 降级标注（失败原因与图片附注单独列出）。已入库文档重跑 → 全 skip（幂等）；`--rebuild` 先删该文档旧块再灌（VLM 升级后清孤儿）。

### 5.4 启动 Web 工作台

```bash
.venv/Scripts/python.exe -m uvicorn docagent.api.server:app --port 8001
# 打开 http://127.0.0.1:8001/      → 问答页（真实链路：检索+引用溯源+命中块原图）
# 打开 http://127.0.0.1:8001/upload → 文档上传页（多文件/文件夹，接真实摄取管线）
```

### 5.5 API

| 端点 | 说明 |
|---|---|
| `GET /health` | 存活检查 |
| `POST /api/v1/ingest` | 异步摄取：落盘 inbox 即返回 `job_id` + `eta_seconds`，后台线程池并发处理；业务失败返回 HTTP 200 + `status=failed`（落盘失败才非 200） |
| `GET /api/v1/ingest/status?job_id=` | 轮询任务状态：`queued/processing/done/failed` + `waited_s/remaining_s` |
| `GET /api/v1/ingest/jobs` | 全部任务精简快照（倒序），供「最近上传记录」面板回查；任务注册表内存态，重启清空 |
| `POST /api/v1/chat` | 问答：`{"message": "…"}` → `final_answer`（含 `[n]` 角标）+ `sources`（溯源契约）+ `usage`/`note` |
| `GET /api/images/{image_id}` | 图片资源：命中块内 `[IMAGE:xxx]` 占位由前端按此地址渲染原图 |

```bash
# 异步摄取：上传即返回 job_id + ETA，真正切片/入库在后台跑
curl -F "file=@data/samples/generated/sample_notes.txt" http://127.0.0.1:8001/api/v1/ingest
# 轮询状态（用上面返回的 job_id）
curl "http://127.0.0.1:8001/api/v1/ingest/status?job_id=ing_xxxxxxxxxxxx"
# 查看全部任务（最近上传记录面板数据源）
curl "http://127.0.0.1:8001/api/v1/ingest/jobs"
# 问答
curl -X POST http://127.0.0.1:8001/api/v1/chat -H "Content-Type: application/json" \
     -d '{"message": "Hermes 手册里如何安装插件？"}'
```

ingest 上传响应示例（立即返回，不等处理）：

```json
{
  "job_id": "ing_a1b2c3d4e5f6",
  "filename": "sample_notes.txt",
  "status": "queued",
  "pages": null,
  "eta_seconds": 20,
  "message": "已接收，后台处理中，预计约 20 秒 后可查询"
}
```

轮询 `/status` 终态响应示例（done）：

```json
{
  "job_id": "ing_a1b2c3d4e5f6",
  "filename": "sample_notes.txt",
  "status": "done",
  "pages": null,
  "eta_seconds": 20,
  "waited_s": 18,
  "remaining_s": 0,
  "result": {
    "status": "ok",
    "chunks_created": 4,
    "stored": 4,
    "skipped": 0,
    "format": "txt",
    "provider": "dashscope",
    "degraded": false,
    "elapsed_s": 1.32
  }
}
```

> 契约约定：**业务失败也返回 HTTP 200 + `status=failed`**（网络错误才非 200），前端据此区分「文档没解析成功」与「服务连不上」。轮询同理：处理失败的任务 `status` 为 `failed` 且 `result`/`error` 带原因，HTTP 仍 200。

### 5.6 演示语料说明

- `data/samples/generated/`（md/txt/json/docx/xlsx/pptx + 含图 PDF）由 `scripts/make_demo_samples.py` 生成，已随仓库提交，可直接摄取。
- `data/samples/*.pdf`（真实语料：中文产品手册、react 英文论文、乱版电子发票、A 股公告与年报 ×6）**均不随仓库提交**（gitignore，面试演示用固定样本），清单与演示价值见 `data/samples/README.md`——clone 后需自行放回对应文件，`verify_chunking` / `verify_ingest_m1` / `verify_ingest_m2` 的 PDF 用例与 `data/golden/qa_golden.json` 的锚句才能全跑通（锚句定位依赖库内有原文）。
- 评估资产：`data/golden/qa_golden.json`（**20 条 golden，v2.0 定稿**）入库；运行产物 `data/reports/` 不入库（随跑随变）。

## 6. 目录结构与文件职责

```
doc-agent/
├── pyproject.toml                 # 依赖与 uv 配置（清华镜像、link-mode=copy）
├── README.md                      # 本文件
├── .env                           # 本地密钥（gitignore，不入库）
├── .gitignore
│
├── docagent/                      # 主包
│   ├── config.py                  # 全局配置：路径 / 向量库 / Embedding·Chat·VLM 模型与 Key
│   │                              #   环境变量优先、.env 兜底；provider/维度说明
│   ├── llm.py                     # 问答 chat 封装（OpenAI 兼容 + usage 统计 + httpx 超时/
│   │                              #   指数退避重试；无 Key 返 None 由调用方降级）
│   ├── qa.py                      # ★ 问答编排（M3）：route→search→answer→citations 状态机；
│   │                              #   幽灵引用剔除纯函数 / 年份两段式检索 / sources 契约组装
│   ├── eval.py                    # ★ 评估引擎（M4）：golden 校验→期望块锚句定位→
│   │                              #   recall@5/MRR→答案层可回查率→门槛→报告落盘
│   │
│   ├── ingest/                    # ── 摄取侧（本 demo 的核心增量）──
│   │   ├── file_type_detection.py # 格式探测：DocumentType 枚举；扩展名主判 + 内容嗅探兜底
│   │   ├── chunk_models.py        # ★ 全项目数据契约：DocumentChunk/ChunkingResult/ExtractedImage、
│   │   │                          #   TRACEABILITY_KEYS 溯源键白名单（含 doc_date/doc_year/
│   │   │                          #   image_ids）、块长常量、make_image_id 等纯函数
│   │   ├── chunking.py            # 切块分发入口 chunk_document()：探测→按格式路由→
│   │   │                          #   _complete_chunk_metadata 统一补溯源键（含日期解析注入）
│   │   ├── chunk_markdown.py      # md：标题结构切（标题链→section，代码围栏内 # 不误判）
│   │   ├── chunk_plain_text.py    # txt：段落切（不复用 md——txt 里 # 是普通文本）
│   │   ├── chunk_pdf.py           # pdf：版面块按 y 分桶+x 排序还原阅读序；图原位 [IMAGE:xxx]；
│   │   │                          #   红页转录文本追加尾部；超长按段/句二次切
│   │   ├── chunk_docx.py          # docx：正文元素归一化为 markdown 流（表格→管道表、
│   │   │                          #   段落 drawing→图片）后复用标题结构切
│   │   ├── chunk_excel.py         # xlsx：一行一条记录（sheet→section、行号→page），保留结构切
│   │   ├── chunk_json.py          # json：按顶层形态切，键路径扁平化 "a.b：值"，保留结构切
│   │   ├── chunk_ppt.py           # pptx：一页一块（标题→section；picture 形状→图片）
│   │   ├── pdf_quality.py         # PDF 质量门：逐页红/黄/绿判级（四类信号、零 LLM）
│   │   ├── vlm_transcribe.py      # VLM 转录（M2）：红页整页渲染→qwen-vl-max(降级 plus)转录；
│   │   │                          #   max_pages 成本护栏；无 Key/失败返空不抛
│   │   └── pipeline.py            # ★ LangGraph 摄取编排：IngestState + 7 节点
│   │                              #   detect→parse→gate→[vlm]→chunk→store→save_images +
│   │                              #   failed 条件边 + run_document() 报告（vlm_*/images_* 字段）
│   │
│   ├── images/                    # ── 图片资源仓库（图片不入向量库，只作引用展示）──
│   │   └── repo.py                #   图文件落 data/images/{doc_id}/ + sqlite 登记
│   │                              #   （幂等 INSERT OR IGNORE；永不推断删除）；resolve_file_path
│   │
│   ├── vectorstore/               # ── 向量侧 ──
│   │   ├── embedding.py           # embedding 提供方：dashscope（分批 10 + 指数退避重试）/
│   │   │                          #   mock 降级；对外 info()/embed_texts()
│   │   └── store.py               # Chroma 封装：get_collection()/upsert_doc_chunks()
│   │                              #   ——md5 幂等 diff→skip/upsert + ingested_at 注入；
│   │                              #   delete_doc_blocks()（显式删除，--rebuild 用）；
│   │                              #   search_docs(query, top_k, where)（年份过滤通道）
│   │
│   ├── api/                       # ── 服务侧 ──
│   │   ├── server.py              # FastAPI：页面托管 + /health + POST ingest（异步：落盘即返
│   │   │                          #   job_id+ETA，后台线程池跑真实管线）+ GET /status + GET /jobs
│   │   │                          #   + POST chat（真实问答）+ GET /api/images/{id}；Chroma 单例
│   │   │                          #   client + 落库锁串行化多 worker 并发
│   │   └── static/
│   │       ├── chat.html          # 问答页（M3 真链路：引用标签 + 命中块原图渲染）
│   │       ├── upload.html        # 上传页（多文件/文件夹，已接 ingest 契约）
│   │       └── style.css          # 两页共用样式
│   │
│   └── cli.py                     # CLI 入口：ingest <paths> [--rebuild] / eval [--top-k N]
│                                  #   [--answers]；与 API 同构，报告表格输出
│
├── scripts/                       # 生成/验证/迁移脚本（产品代码零依赖，可独立跑）
│   ├── make_demo_samples.py       # 生成演示语料（md/txt/json/docx/xlsx/pptx + 含图 PDF）
│   ├── verify_chunking.py         # 探测 + 四策略切块：78 断言（真实 PDF 对照源文件逐块核验）
│   ├── verify_images_ref.py       # 图片引用链路（提取/占位符/幂等/回填）：28 断言
│   ├── verify_ingest_m1.py        # 摄取编排 M1：51 断言（质量门生效/混合格式/幂等/坏文件隔离）
│   ├── verify_ingest_m2.py        # 摄取编排 M2（VLM 转录注入 + 图片落盘）：15 断言
│   ├── verify_server_ingest.py    # ingest 端点真实链路冒烟：9/9（clone 即可跑）
│   ├── verify_docqa_m3.py         # 问答链路（引用约束/幽灵剔除/sources 契约/降级）：14 断言
│   ├── verify_docqa_m4.py         # 评估引擎（golden 校验/指标/门槛/usage 聚合）：29 断言
│   ├── verify_date_metadata.py    # 日期元数据（解析/契约/年份过滤/两段式裁决）：37 断言
│   └── upgrade_add_date_metadata.py  # 一次性存量库迁移：只补 metadata 零重嵌（md5 不触发
│                                     #   增量重嵌，需显式升级；详见 §9）
│
├── docs/
│   ├── requirements.md            # ★ 需求文档：背景/范围/取舍决策 D1-D8/验收清单 R1-R7/里程碑
│   └── coding_standards.md        # 开发守则八条（防过度设计、人类化命名等）
│
└── data/                          # 运行时数据（chroma_db / inbox / images / image_files.db /
    │                              #   markdown / chunk / reports 均 gitignore，不入库）
    │                              #   chunk/：切片质量缓存（data/chunk/{stem}.md，调试辅助）
    ├── samples/                   # 演示语料（generated/ 入库；真实 *.pdf 自行放置）
    ├── golden/qa_golden.json      # ★ M4 评估资产：20 条 golden × 9 文档（v2.0 定稿，入库）
    └── reports/                   # 评估报告运行产物（gitignore，随跑随变）
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
| Excel / JSON | 记录切（**保留结构，不转 md**，D8） | 一行记录 / 一个数组元素 | `section` = sheet 名或 `$[N]` 路径 |
| PDF / PPT | 页切 | 一页 = 一个主题 | `page` = 页码（溯源最常用定位） |
| TXT | 段落切 | 一个段落（空行分隔） | 无章节概念，不强造 section |

通用约束：块长目标 900 字符、硬上界 1200（超界二次切 + 80 字符重叠）；所有策略共用 `DocumentChunk` 结构与溯源 schema，切法差异只体现在 metadata 填充上。

**为什么表格不能按字符窗口切**（demo1 实证）：行与列是作者切好的业务边界，按字符窗口切会把一条记录拦腰切断，检索只能命中半条记录。

### 7.3 PDF 质量门（红/黄/绿）+ VLM 慢路径

| 档 | 含义 | 处置 |
|---|---|---|
| 红 | 硬伤：文本不可信（坏字体乱码页、纯图扫描页） | **gate 条件边 → VLM 整页转录**（M2），产物 `figure_transcript` 块 |
| 黄 | 软提示：文本可用但版面复杂（少文字 / 含表格图片） | 可选 MinerU 提结构（v1.1 候选） |
| 绿 | 直接放行 | PyMuPDF 产出够用 |

全部信号零 LLM 成本、阈值可调。**转录 v2 语义**（2026-09-08 修正，防孤儿块）：

1. 红页**原文不可信 → 不产块**（乱码文字入库只会污染检索）；
2. 转录文本作为 `block_type=figure_transcript` 块**追加到文档块流尾部**——老块 id/顺序完全不动（若原位覆盖，转录后块数变化会导致后续 `block_seq` 平移、新旧 id 错位成孤儿）；
3. 历史遗留的乱码块由 CLI `ingest --rebuild` 显式删除重建清理；
4. 无 Key / 转录失败 / 超 `max_pages` 上限 → 报告记 `vlm_note`，该页降级保持原文入库（R7 不崩）。

### 7.4 幂等落库模型

- 业务主键 `id = {doc_id}:{block_seq}`（杜绝自增序号——内容变化会导致同文档新旧块 id 错位）。
- metadata 存块文本 md5：`coll.get(ids=本地ids)` 查旧 → md5 相同 skip（不 touch、不重嵌、不重计费）/ 不同或不存在 → embedding + upsert（同 id 覆盖）。
- `ingested_at`（入库时间）只对**实际写入**的块刷新，skip 的块保留首入值。
- 只 upsert、永不推断删除；删除是业务层职责（`delete_doc_blocks`，增量补录模型差集可能误删未更新文档）。

### 7.5 图片引用链路（D7，图片不进向量库但不丢弃）

```
切块器提取图片 ─▶ image_id = {doc_stem}-{内容md5前10}（内容幂等，同图重跑同 id）
   ├─▶ 文本流原位写 [IMAGE:{image_id}] 占位符（随块文本入向量库，检索不丢上下文）
   ├─▶ 块 metadata.image_ids = "id1,id2"（逗号串，Chroma metadata 只收标量）
   └─▶ images/repo.save_images：图文件落 data/images/{doc_id}/ + sqlite 登记（INSERT OR IGNORE）

召回后：展示层把命中块文本的 [IMAGE:x] 替换为 <img src="/api/images/x">
```

幂等与清理哲学与向量库同款：同图不重写、永不推断删除（孤儿图由业务层清理）。**图内文字由 VLM 转录覆盖**（D6：不做图片向量库——无多模态 embedding 通道、rerank 只吃文本，纯文本单通道是自洽形态）。

### 7.6 日期元数据与召回年份感知

三个溯源键 + 两段式裁决，防跨年报表张冠李戴。

**三个溯源键**（全进 `TRACEABILITY_KEYS` 白名单）：

| 键 | 语义 | 注入点 |
|---|---|---|
| `doc_date` | 文档日期 `YYYY-MM-DD` 或 `YYYY` | 文件名解析（`extract_doc_date`），`chunking._complete_chunk_metadata` 统一注入（所有格式唯一汇合处） |
| `doc_year` | int，供 Chroma `where` 过滤 | 有 `doc_date` 才写 |
| `ingested_at` | 块入库 ISO 时间 | `store.upsert_doc_chunks`，仅实际写入的块刷新 |

**召回两段式裁决**（`qa.search_node`）：

1. 恒做一次**不限年份语义检索**作基准（零额外成本路径）；
2. 问题含显式年份（`extract_year_filter` 正则收集全部年份 → `doc_year $in [...]`）→ 加一次年份过滤检索；
3. **过滤命中与语义 top1 同 doc_id**（该主题在目标年份确有内容）→ 用过滤结果，护栏生效；
4. 过滤**跑题或为空**（如库只有振华 2025 年报、问"振华 2024"会滤到天力 2024）→ **回退语义结果 + note 明示**，防错位的同时不把可答问题滤成"未找到"。

上下文块头带「文档日期」，让模型自查年份错位；sources 契约透传 `doc_date`。局限：`doc_date` 源自文件名（标题无年份的公告无此键，会被年份过滤排除 → 走回退路径）；相对时间词（"去年/最新一期"）初级版不处理。

### 7.7 切片质量缓存（调试辅助，不入库）

`chunk_node` 切完即把块原样落盘 `data/chunk/{文件名stem}.md`（同名覆盖写），每块前置一行 HTML 注释标记：

```
<!-- 块 3 | chars=842 | block_type=paragraph | page=5 | section=第2章/2.1 安装 | doc_date=2026-08-01 | image_ids=img_ab12cd34ef -->
正文内容……
```

用途：切片质量（跨页表格是否断裂、块大小、page/block_type/section 标记、图片引用）直接查向量库不直观，这份可读副本打开即可逐块人工核对。写盘失败只打一行终端提示、不抛异常、不影响主链路（落库）判 failed。

### 7.8 异步摄取与并发落库（2026-09-09）

**为什么异步**：原 `POST /api/v1/ingest` 在请求线程同步跑完整管线（保存→解析→切片→入库），大文件（220 页年报）会长时间占住请求，前端干等「上传中」；串行上传时一个大文件阻塞其后所有文件。改为**上传与处理解耦**：请求线程只落盘 inbox + 估算 ETA，立即返回 `job_id`；真正的切片/入库在 `ThreadPoolExecutor`（`DOCAGENT_INGEST_WORKERS` 默认 2）后台线程跑，多文件互不阻塞。前端拿 `job_id` 即翻「处理中」+ 不确定动画，轮询 `/status` 把状态翻成「已完成/失败」。

**并发落库安全**（多 worker 同时写 Chroma 必炸）：`store.get_collection()` 改为进程内**单例 `PersistentClient`**（同路径多 client → `database is locked`），并把「`collection.get` 读旧 → `embed_texts` 嵌入 → `collection.upsert` 落库」整段用一把进程级锁 `_store_lock` 串行化。解析/切块仍并行，仅碰 Chroma 与打 DashScope 的共享段串行——杜绝写锁争用。实测 4 并发各 120 块 → 0 错误、md5 幂等正确、库内 480 块。

**任务注册表**：内存字典 `_INGEST_JOBS`，`/jobs` 返回倒序快照供「最近上传记录」面板回查；超 200 条自动回收最旧已完结项。重启清空未完成任务（已知边界，不持久化）。

## 8. 测试与验证

| 脚本 | 覆盖 | 语料依赖 | 结果 |
|---|---|---|---|
| `verify_chunking.py` | 探测 + 四策略切块（真实 PDF 对照源文件逐块核验） | `generated/` + 真实 PDF | 78/78 |
| `verify_images_ref.py` | 图片引用链路：提取 / 占位符 / 幂等 / 回填 | `generated/` 含图样本 | 28/28 |
| `verify_ingest_m1.py` | 质量门活体生效 / 混合格式全 ok / 幂等全 skip / 坏文件隔离不中断 / mock 降级标注 | `generated/` + 真实 PDF | 51/51 |
| `verify_ingest_m2.py` | VLM 转录注入（红页不产块/追加尾部/幂等）/ 图片落盘 | `generated/` + 真实 PDF | 15/15 |
| `verify_server_ingest.py` | ingest 端点异步契约：上传即返 job_id / 轮询至终态 / 重传 skip / 坏文件业务失败 | 仅 `generated/`（clone 即跑） | 9/9 |
| `verify_docqa_m3.py` | 问答链路：引用约束 / 幽灵剔除 / sources 契约 / 降级不崩 | 临时库 mock | 14/14 |
| `verify_docqa_m4.py` | 评估引擎：golden 校验 / 指标 / 门槛 / usage 聚合 | 临时库 mock | 29/29 |
| `verify_date_metadata.py` | 日期解析 / 溯源键契约 / where 过滤 / 两段式裁决三场景 | 临时库 mock | 37/37 |

```bash
.venv/Scripts/python.exe scripts/verify_server_ingest.py   # clone 后可直接跑（其余依赖真实语料）
```

> ⚠️ **语料依赖说明**：真实 PDF 未随仓库提交（见 [§5.6](#56-演示语料说明)），文件清单在 `data/samples/README.md`。缺文件时相关脚本明确报错，不影响 `generated/` 用例。**真实链路额外验证过**（配真 Key）：react 5 红页 qwen-vl 转录 28–48s/页，判别词语义检索全部命中转录块；跨年问答四场景行为正确；`eval --answers` 引用可回查率 1.0。

**已知实践结论**（防复发沉淀）：
- mock 全绿 ≠ 真实可跑——mock 直接替换模块函数，不校验协议/网络。真实大批量 embedding 曾触发 DashScope 断连，已加**每批 3 次指数退避重试**；VLM/chat client 均带 httpx 超时（无超时曾 17 分钟挂起）。
- 切分结果必须与源文件交叉验证（曾踩 `MarkdownHeaderTextSplitter` 同标题相邻块合并：73 块→20 块，SKU 张冠李戴）。
- Chroma `collection.query(query_texts=)` 会触发默认 all-MiniLM 模型下载——检索统一走显式 query embedding（`search_docs`）。
- md5 只算文本 → **metadata 结构变更不触发增量重嵌**，存量库升级要显式跑迁移脚本（§9）。
- 年份过滤单独看"空/非空"不够：过滤命中**跑题主题**块时不触发降级且答非所问——必须比对过滤结果与语义 top1 的 doc_id 一致性。

## 9. 里程碑 Roadmap

| 里程碑 | 内容 | 状态 |
|---|---|---|
| **M1** | 项目骨架 + 探测/分格式切块 + 质量门 + **LangGraph 摄取编排**（5 节点）+ CLI/API 双入口 + 幂等落库 | ✅（51/51 + 9/9） |
| **M2** | 图片引用链路（提取/占位符/sqlite 仓库/落盘，D7）+ **VLM 转录**（gate 条件边 → 红页 Qwen-VL 转录，v2 追加尾部语义） | ✅（15/15；真实闭环实证） |
| **M3** | 问答链路：检索 → LLM 生成 → **幽灵引用剔除** → 引用溯源 sources → `/api/v1/chat` 真链路 + 图片展示 | ✅（14/14） |
| **M4** | **评估体系**：golden 定稿 20 条 × 9 文档 + 检索 recall@5/MRR + 答案层引用可回查率 + 门槛 + 报告落盘 | ✅ 定稿（29/29；真实 recall@5=0.95 / MRR=0.882 / 可回查率 100%，q17 诚实用例） |
| — 增量 — | metadata 日期键（`doc_date`/`doc_year`/`ingested_at`）+ 召回年份感知（两段式裁决防张冠李戴）+ 存量库零重嵌迁移 | ✅（37/37 + 真实四场景） |
| — 增量 — | 切片质量缓存（`data/chunk` 落盘）/ 异步摄取（线程池 + job 注册表 + `/status` + `/jobs`）/ Chroma 并发落库加固（单例 client + 落库锁） | ✅（真实链路 4 并发零错误；上传页 jobs 面板 JS 回填待补，见 §12） |

**v1.1 候选**（不在 MVP 承诺内，见 requirements.md）：golden 补负例与多轮追问用例（20 条已达成 R4 线）；MinerU 慢路径接入（D5，测后决定）；混合检索（BM25+RRF）与 Rerank（q17 表格数字 miss 的直接增强方向）；多轮会话记忆；忠实性深度 checker（答案断言 ↔ 引用块核对）；文档粒度并行（Send）与 checkpoint 断点续跑。

**存量库迁移备忘**：metadata 结构变更后旧块不自动升级（md5 不感知 metadata）。模式 = 写一次性脚本显式 `collection.update(ids, metadatas)`（零重嵌），如 `scripts/upgrade_add_date_metadata.py`（1400 块秒级完成，幂等可重跑）。

## 10. 配套文档

- [docs/requirements.md](docs/requirements.md) — 需求文档（背景与目标 / MVP 范围 / 技术选型与资产复用 / 验收清单 R1–R7 / 里程碑 / 开放决策 D1–D8）
- [docs/coding_standards.md](docs/coding_standards.md) — 本仓库开发守则八条（体现工程观，欢迎指正）
- [data/samples/README.md](data/samples/README.md) — 演示语料清单与解析形态说明

## 11. 设计原则

本仓库按八条硬性守则开发（见 `docs/coding_standards.md`），核心几条：

1. **不过度设计**——只写当前需求的最小结构，不预建抽象/配置/接口壳；需求出现再重构。
2. **人类化命名**——变量全名不缩写、语义自解释，读起来像句子。
3. **模块化 + 分层纪律**——内层纯函数可独立测试、不感知编排；编排只是可换的薄外壳。
4. **确定性优先于赌模型**——能规则判定的不请 LLM（质量门零成本信号、幽灵引用代码层剔除、年份裁决纯正则即三例）。
5. **降级不掩盖、mock 不冒充**——无 Key 可跑但明确标注 degraded，验证脚本同时覆盖 mock 与真实链路。
6. **与业务语义对齐**——切片单元 = 作者切好的知识边界（行/页/章节），不按字符窗口硬切。

## 12. 已知局限与待办

- **复杂版面 PDF 的端到端语义切块**（已知待优化）：当前 PDF 是页内切块 + 红页追加尾部，对跨页大表格、被吞的章节标题、单独成块的表格等场景切块边界不够优（中金辐照公告已观测到「片段 1 末尾 + 片段 2 开头被切一起、表格单独成块」）。轻量修 = 识别中文数字章节标题打断聚合；重量修 = 版面层级语义切块。v1.1 候选。
- **上传页「最近上传记录」面板 JS 回填待接**：服务端 `/jobs`、`/status` 已可用，HTML 容器已就位，但轮询把任务结果回填进面板的 JS 逻辑尚未接入（当前上传后页面停在「处理中」+ ETA，关闭页面不影响后台，但需手动刷新/轮询才能看到终态）。这是今天代码 review 的重点待补项。
- **任务注册表内存态**：`_INGEST_JOBS` 不持久化，服务重启丢未完成任务（符合不过度设计，当前不引入 Redis/DB 任务队列）。
- **评估 q17 保留用例**：振华英文年报净利数字问，表格断行文本 + 语义词在大库内分布广 → 单路向量召回 miss；检索 miss 时引用仍可回查但答案依据弱（"可回查 ≠ 答对"），作 rerank/表格增强评估基线，不以改 query 洗指标。
- **年份过滤依赖文件名日期**：标题无年份的公告缺 `doc_date`，会被年份过滤排除走回退；相对时间词（"去年/最新一期"）初级版不处理。

---

*该项目为个人学习与求职作品，代码与文档会随学习持续迭代。*
