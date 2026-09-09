# DocAgent — 文档智能 + RAG 工程（第 2 个 demo）需求文档

> 版本：v2.0（2026-09-09，M1-M4 全落地 + golden 定稿 20 条回写；README/架构图已同步）
> 定位：求职第 2 个 demo 的需求规格。开发模式：**AI 共同开发（结对制）**——本文件是双方对齐的地图，里程碑逐段 review。

---

## 1. 背景与目标

### 1.1 为什么做这个 demo

第 1 个 demo（电商客服 CS Agent）已完成使命：证明 Agent 编排能力（LangGraph Supervisor/Send/interrupt/记忆/评估全链路真实跑通）。但它有两条天花板：
1. **技术增量已见底**——再做只是换皮换量，面试官看到的是同一个能力；
2. **数据墙无解**——客服要真实对话与订单数据，没有就永远停在 demo 感。

第 2 个 demo 换赛道到 **RAG 生产侧（数据侧）**，理由：
- **零真实业务数据依赖**：公开文档（手册/论文/合同模板/报表）就是天然语料，验证不靠客户数据；
- **差异化空白区**：市面上 RAG 作品九成停在"LangChain 包一层检索"，解析层深度（分级路由/质量门/VLM 补全/量化评估）是多数候选人的空白；
- **技术栈延续 + 增量**：消费侧复用 demo1 编排，摄取侧是"数据流水线编排"——同一 LangGraph 框架两种用法。

### 1.2 核心命题

> 把企业"乱七八糟的文档"（PDF/Word/PPT/Excel/图片）变成**可检索、可问答、答案带出处**的结构化知识库。

### 1.3 与 demo1 的分工叙事（面试一句话版）

> demo1 = 对话编排（消费侧 Agent）；demo2 = 文档摄取流水线（生产侧）+ 引用溯源问答。两个 demo 拼起来 = "数据进得来、问题答得出、答案查得到"的完整 RAG 工程能力。

---

## 2. 目标用户与主场景

演示画像：企业里需要从散落文档（产品手册/合同/说明书/报表）快速查答案并**追溯来源**的人（运营/客服/研发/法务）。

| # | 场景 | 动作 | 输出 |
|---|---|---|---|
| S1 | 建库 | 把一批混合格式文档投进 `inbox/`，触发摄取 | 摄取报告：每份文档绿/黄/红页统计、VLM 补全块数、耗时与成本 |
| S2 | 问答 | 自然语言提问（可追问） | 答案 + **引用出处**（文档名/页码/章节/块） |
| S3 | 回归 | 跑内置 golden set | 量化报告：检索命中率、回答忠实性、与上次对比 |

---

## 3. MVP 范围

### 3.1 做（v1.0）

**摄取侧（生产）**——六步流水线：
1. **格式探测**：扩展名 + 内容嗅探（文字版/扫描版/图文混排）
2. **分级解析路由**（质量门驱动）：PyMuPDF 快路径优先 → 质量门逐页判级 → 红页/图块标记
3. **慢路径补全**：红页走 OCR/MinerU；被判 figure 的信息密集块走 **VLM 转录**（demo1 期已实证：8/10 → 10/10）
4. **结构化切片**：块 = 章节/段落/表格/图注 + 图转录；粒度对齐"问题单元"；**不去重合并**（解析层文本/表格双输出会重复，需去重）
5. **元数据与溯源字段**：doc_id / title / page / section / block_type / block_seq
6. **落库 Chroma**：块级向量，id = `{doc_id}:{block_seq}`；幂等 upsert（沿用 demo1 的 md5 增量模型）

**消费侧（问答）**——复用 demo1 编排换工具集：
- supervisor 意图识别 + ReAct 子 agent + 收尾器（demo1 骨架直接迁移）
- 工具集：`search_docs(query, top_k)` / `get_doc_block(block_id)`（纯读，无写工具）
- **引用溯源（本 demo 核心新增）**：答案按句标注出处；实现=检索块 id 传递 + 提示词约束"只引用本次检索到的块" + **代码层校验**（答案里的引用若不在本次检索集合 → 剔除标注，防幽灵引用——沿用 demo1 strip_ghost 教训：不赌模型）
- 多轮上下文：MVP 只做同会话简答（RedisSaver 可复用，但不作为演示重点）

**评估**：
- 摄取侧：解析 golden（复用 demo1 期 eval_parsing 思路，语料扩到混合文档）
- 问答侧：问答 golden（query → 期望来源块集合）→ 检索命中率；忠实性 checker（答案断言能否在引用块中找到）
- 一键回归脚本 `verify_docqa.py`（沿用 demo1 verify_* 确定性断言模式）

**演示形态**（D3 已定）：FastAPI 服务化（复用 demo1 经验：thread 会话、X-User-Id、/health）+ 网页问答工作台（提问 + 引用展示）+ CLI 命令同构（`ingest`/`ask`/`eval`，作开发与回归入口）。

### 3.2 不做（防过度工程）

| 不做 | 原因 |
|---|---|
| 用户系统/权限/多租户 | 单用户演示，demo1 已演示过身份注入 |
| 定时/增量摄取自动化 | 手动触发即可；增量 upsert 逻辑复用但不自动跑 |
| 混合检索（BM25+RRF）/Rerank | 学习计划 D7-D10 之后作为 v1.1 增强并入，MVP 单路向量够演示 |
| 复杂多轮记忆/用户画像 | demo1 已证过，此处喧宾夺主 |
| 摄取任务管理后台（文档上传/批量监控页） | 摄取入口先走 CLI/API；MVP 网页只做问答页（提问 + 引用展示） |

### 3.3 v1.1 候选（不在 MVP 承诺内）

MinerU 本地集成 / 混合检索+RRF / RAGAS 端到端 / DOCX/PPT 解析完善 / 摄取断点续跑。

---

## 4. 技术选型与资产复用

### 4.1 从 demo1/ecommerce-agent 复用的已验证资产

| 资产 | 位置（ecommerce-agent） | 用途 |
|---|---|---|
| 解析快路径 | `app/chunk/pdf_to_markdown.py` + `demo_pymupdf_basic.py` | 摄取侧文字版 PDF → md |
| 质量门 | `app/chunk/quality_gate.py` | 逐页红黄绿判级，驱动解析路由 |
| VLM 转录 | `app/chunk/vlm_describe_figure.py` | figure 块图内文字转录（qwen-vl） |
| 解析评估 | `app/chunk/eval_parsing.py` + `golden_queries.json` | 摄取侧 golden 引擎（扩展语料） |
| Agent 编排骨架 | `app/agents/*` + `app/supervisor.py` + `app/graph.py` | 消费侧问答（换工具集与提示词） |
| 向量库经验 | `app/vectorstore.py`（md5 增量 upsert / id=业务主键） | 块级落库 |
| LLM 封装 | `app/llm.py`（usage 插桩、重试、关键字传参） | 所有 LLM 调用 + 成本归因 |
| 日志/统计模式 | `app/conversation_log.py` / `ticket_store.py` | 摄取任务日志 + 报告 |
| 踩坑手册 | `docs/second_demo_lessons.md` | 全程对照（25+ 条坑防复发） |

### 4.2 新增技术点（第 2 个 demo 的学习增量）

| 增量 | 说明 |
|---|---|
| **LangGraph 数据流水线编排** | 不是对话图，是"状态机式摄取管线"：节点=六步、条件边=质量门分支、错误=单文档隔离 + 重试（体现同框架两种用法） |
| 文档级知识 schema | doc_id 为根的父子结构（doc → page → block），溯源字段落 metadata |
| 引用溯源链路 | 检索块 id → 答案句级标注 → 代码层幽灵引用剔除 |
| 忠实性评估 | 答案断言 ↔ 引用块内容的自动核对（LLM-as-judge 或规则式）。**M4 收尾决策（2026-09-09）**：深度断言核对留 v1.1——MVP 以"引用可回查率 100%"（引用块必属本次检索块，代码层可验）代忠实性闸，先锁"答得出、出处在" |
| 语料工程 | 造 5 类真实公开文档组成的演示语料（含 1 份已知坏字体页文档） |

### 4.3 语言与依赖

Python 3.13+，与 demo1 同栈（.venv 隔离）；新依赖控制在最小集（pymupdf 已有，其余按需）。

---

## 5. 功能需求明细

### 5.1 摄取（S1）

**输入**：`inbox/` 目录放任意混合文档。演示语料建议 5 类（见 D2）：
- 中文产品手册（Hermes 橙皮书类，83 页图文混排）
- 英文论文（含 2.pdf P2 已知坏字体页——**质量门/VLM 的活体演示料**）
- 电子发票（表格型 1 页）
- 合同/协议模板（可选）
- 纯图片页文档（可选，触发 OCR 分支）

**摄取报告字段**（每文档）：文件/格式/页数/快慢路径分配/绿黄红页数/VLM 转录块数/耗时/LLM token 成本；整批汇总。

**鲁棒性要求**（demo1 教训落实）：
- 单文档失败不中断全批（隔离 try + 报告标 FAIL）
- 无 Key 降级不崩（走规则/占位，报告注明"未启用 LLM 增强"）
- 幂等：同语料二次摄取全 skip（md5），报告注明

### 5.2 问答（S2）

**请求**：`question` + 可选 `session`。**响应**：`answer`（含 `[1][2]` 句级引用角标）+ `citations[]`（每引用：block_id/文档/页/章节/原文摘要）+ 耗时/成本。

**引用溯源约束（核心规则）**：
1. 检索返回 top-k 块，携带 block_id 进上下文；
2. 提示词：只允许引用可见块，格式 `[n]`；
3. 代码层校验：解析答案中 `[n]` → 若 n 超出本次可见块集合，**剔除角标与对应引用条目**，不留残句（沿用 strip_ghost_ticket 的"整句删"经验，幽灵引用一律删标注不删答案内容——答案句本身保留，只去掉无法核实的出处）；
4. 无引用可标的答案：明示"以下为综合回答，未找到直接文档出处"。

**越权/越界不做**：不接外部网络、不写任何业务系统（纯本地只读知识库）。

### 5.3 评估（S3）

| 层 | golden | 指标 | 工具 |
|---|---|---|---|
| 解析层 | 每文档 10-15 条判别词 query | content_present / recall | 扩展 eval_parsing |
| 检索层 | 问答 golden：query → 期望块集合 | recall@5 / MRR | verify_docqa.py |
| 回答层 | 同批 query 的答案 | 引用可回查率 100%（M4 实测 19/19 达成）/ 忠实性断言核对 v1.1 | 代码层回查（sources.block_id 必属检索块） |

**回归门槛**（每里程碑过闸）：检索 recall@5 ≥ 0.8；引用可回查率 = 100%；全绿无失败文档。

---

## 6. 非功能需求

| 项 | 目标 | 备注 |
|---|---|---|
| 延迟 | 问答感知延迟 ≤ 10s；快路径摄取 ≤ 2s/页 | demo1 60s 教训：问答侧纯读无写，天然更轻 |
| 成本 | 摄取报告含 token 成本；LLM 只在 VLM 转录/回答处用 | 质量门/切片零模型成本 |
| 可观测 | 摄取任务级日志 + 每步耗时（timings） | 沿用 conversation_log 模式，预留 T0 分段计时的教训 |
| 鲁棒性 | 见 5.1；坏文档不拖垮整批 | demo1 全链路验证过 |
| 可复现 | 一键起服务：uvicorn 启动即问答页可用；CLI `ingest`/`ask`/`eval` 供开发与回归 | 面试演示零环境摩擦 |

---

## 7. 验收清单（可测，最终 gate）

| # | 验收项 | 判据 | 状态（M4 收尾 2026-09-09） |
|---|---|---|---|
| R1 | 混合语料建库 | 5 类文档全入库；报告逐文档可解释 | ✅ verify_ingest_m1/2 全绿；真库 1400 块（手册/论文/发票/公告×4/年报×2 + 生成样本） |
| R2 | 质量门演示 | 含坏字体页文档被标红并走 VLM 补全；报告可见该页"转录回填" | ✅ react 5 红页 qwen-vl 真实转录闭环；verify_ingest_m2 15/15 |
| R3 | 图内文字可答 | 提问 P2 图内内容（Apple Remote 类），能答且带出处（沿用 q04/q05 实证） | ✅ golden q01 常驻回归；判别词真实检索命中 figure_transcript 块 |
| R4 | 问答 golden | 20 条 recall@5 ≥ 0.8；引用可回查率 100% | ✅ golden 定稿 20 条 × 9 文档；真实 recall@5=0.95 / MRR=0.882；答案层可回查率 100%（详见 §8 M4 行） |
| R5 | 幽灵引用拦截 | 诱导模型引用不可见块 → 角标被剔除、无假出处（负例用例） | ✅ verify_docqa_m3 负例 + strip_out_of_range_citations 纯函数单测 |
| R6 | 一键回归 | verify_docqa.py 全绿；报告含耗时与成本 | ✅ verify_docqa_m3/m4/date_metadata 等 8 脚本 261 断言；eval 报告含耗时与 usage 成本 |
| R7 | 降级不崩 | 无 Key 环境跑通 S1/S2，报告标注降级 | ✅ 全套 verify 在 mock/无 Key 环境全绿，报告标注 degraded |

> ⚠️ R4 的 q17（振华英文年报"净利归属股东"数字问）为**保留的诚实用例**：2026-09-09 真实检索 miss（期望块在 top1 文档内但排名>5）——表格断行文本 + 语义词在 812 块大库内分布广泛，单路向量召回不足。门槛 ≥0.8 非满分制，此用例留作 v1.1（rerank/表格增强）的评估基线，不以改 query 洗指标。

---

## 8. 里程碑（结对开发节奏）

| 里程碑 | 内容 | 过闸标准 | 依赖资产 |
|---|---|---|---|
| M1 | ✅ 完成（2026-09-07）：项目骨架（FastAPI + CLI 同构）+ 摄取编排 v1 并入 LangGraph（5 节点 detect→parse→gate→chunk→store，每节点失败路由）+ POST /api/v1/ingest 真实端点 | 单文档入库 + 报告雏形；verify_ingest_m1 47/47 全绿 | pdf_to_markdown / quality_gate / vectorstore 经验 / demo1 server 骨架 |
| M2 | 图片引用链路（✅ 2026-09-08：提取/占位符/image_ids/sqlite 仓库/编排落盘，D7）+ VLM 转录（✅ 2026-09-08 mock 15/15 + **真实链路闭环**：react 5 红页 qwen-vl 真实转录 → 判别词 Apple Remote/keyboard/Front Row 语义检索命中 P2 figure_transcript 块）。**v2 语义**：红页原文不可信不产块、转录块 figure_transcript 追加尾部（老块 id 不动、无平移无孤儿）、历史乱码块由 CLI `--rebuild` 显式重建 | R2/R3 过闸 | vlm_describe_figure + 图片仓库 |
| M3 | 问答 Agent（✅ 2026-09-08：docagent/llm.py chat 封装 + qa.py 问答图 route→greet/search→answer→citations，幽灵引用代码层剔除 strip_out_of_range_citations + sources 契约 + 图片展示端点 /api/images/{id} + chat.html 渲染原图） | R3/R4/R5 过闸 | demo1 agents 骨架 |
| M4 | ✅ 完成（2026-09-09）：**golden 定稿 20 条 × 9 份真实语料**（新增正川/南卫公告年报，q14 年份护栏活体）+ eval 引擎（锚句定位期望块 / recall@5·MRR / 答案层引用可回查率 / 门槛判定 / 报告落盘）+ CLI `eval [--top-k N] [--answers]` + README/架构图同步 + 本文件 v2.0 回写。真实结果：**recall@5=0.95（19/20）、MRR=0.882、答案层 20/20 带引用、引用可回查率 100%**；q17 诚实用例 miss 记录于 §7 警告 | R1-R7 全绿（见 §7 勾选） | eval_parsing 扩展 |
| v1.1（待定） | MinerU 集成 / 混合检索 RRF / 网页工作台 | 视真实场景与时间 | — |

每里程碑结束：**可演示 + 结对 review + 需求文档回写（偏差记录）**。

---

## 9. 开放决策（待拍板，升 v1.0 前）

| # | 决策点 | 默认建议 | 影响 |
|---|---|---|---|
| D1 | 项目命名 | `doc-agent`（目录已建 `D:\workspace\doc-agent`） | 无，可随时改 |
| D2 | 演示语料侧重 | ✅ 已定（2026-09-07）：通用办公文档（中文手册/英文论文/电子发票/合同模板） | 语料落 `data/samples/` |
| D3 | 演示形态优先级 | ✅ 已定（2026-09-07）：API + 网页问答工作台（CLI 同构作开发/回归入口） | M1 起手含 FastAPI 骨架 |
| D4 | 摄取编排是否用 LangGraph | ✅ 已定（2026-09-07）：**是**。摄取无模型自主决策循环，本质 Workflow 非 Agent——上编排不为"智能"，借状态机外壳组织六步（节点=步骤 / 条件边=失败与质量门路由 / state=逐文档报告累积）；同为 demo1 框架两种用法的核心叙事 | M1 已按 5 节点状态机落地 |
| D5 | MinerU 慢路径形态 | MVP 用 VLM 转录覆盖 figure 块（已实证）；MinerU 本地装放 v1.1 | 决定慢路径依赖 |
| D6 | 向量库形态 | ✅ 已定（2026-09-07）：**单文本通道**——图片/图块不做独立图片向量库（无多模态向量），由 VLM 转录为文本块（block_type=figure_transcript）入同一文本库 | 决定 M2 转录产物去向；M1 本就纯文本向量 |
| D7 | 图片的展示引用（图片向量之外的第二角色） | ✅ 已定（2026-09-08）：**图片不进向量检索，但不丢弃**——切块时提取图片生成 image_id（=doc_stem+内容md5前10，幂等去重），写入 sqlite 图片表（doc_id/url/image_id，图文件落 data/images/）；文档流原位写 `[IMAGE:xxx]` 占位符（随块文本入向量库）；metadata 存 `image_ids`（逗号串，Chroma 标量约束）；召回后由展示层把占位符替换为 `<img>`。图片向量仍不做（demo 无高价值/专用文档场景，v1.1 候选） | 决定摄取层图片提取与落盘；M2 范围含图片引用 + VLM 转录两条线 |
| D8 | 中间表示统一策略 | ✅ 已定（2026-09-08）：**按文档形态分流，不一刀切转 md**——线性排版文档（word/ppt）归一化为 md（已实现）；**excel/json 保留结构切**（转 md 丢"一行一记录"/json_path 溯源，倒退）；PDF 快路径直取文本、红页走慢路径（MinerU 仍按 D5 放 v1.1，测试后评估） | 决定切片层架构维持"形态路由"，不重构为全量 md 归一化 |

---

## 10. 参考（demo1 沉淀，本项目直接引用）

- 踩坑手册：`ecommerce-agent/docs/second_demo_lessons.md`（8 章 25+ 坑）
- 解析实证：`ecommerce-agent/app/chunk/`（demo_pymupdf_basic / pdf_to_markdown / quality_gate / vlm_describe_figure / eval_parsing / golden_queries / corpus_mineru）
- 评估报告：`ecommerce-agent/app/chunk/output_md/parsing_eval_report*.md`（8/10 → 10/10 完整故事）
