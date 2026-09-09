# MEMORY.md — doc-agent 项目长期记忆

> 文档智能 + RAG 工程（求职第 2 个 demo；与 ecommerce-agent 协作反转：AI 主写实现、用户定框架/范围/验收，每里程碑交付后用户 review）。恢复上下文第一入口。

## 项目概况
- 位置 `D:\workspace\doc-agent`（Python 3.13+，uv 管理，清华镜像）。运行：`.venv/Scripts/python.exe -m docagent.cli ...`；服务 `uvicorn docagent.api.server:app --port 8001`
- 定位：RAG **摄取端（Ingestion）** 差异化工程——多格式（PDF/Word/Excel/PPT/MD/JSON/TXT）→ 探测→解析→质量门→VLM 转录→图片引用→定制切片→幂等落库→向量问答(引用溯源)→golden 评估。摄取/问答编排都用 LangGraph
- 核心叙事：分格式定制切块 + 质量门前置 + 红页 VLM 慢路径 + 图片不进向量库 + 年份召回 + 评估闭环 + 无 Key 降级

## 核心设计决策
- 分格式定制切块：md/docx 标题结构切、excel/json 记录切(保留结构不转 md，D8)、pdf/ppt 页切、txt 段落切
- 质量门红/黄/绿 + 红页 VLM 整页转录(v2 追加尾部语义，防孤儿块 id)；无 Key/失败降级保持原文
- 图片 `[IMAGE:xxx]` 占位随块入库，图文件落 data/images/{doc_id}/ + sqlite 登记(D6/D7)，图内文字由 VLM 覆盖
- md5 增量幂等 upsert：id=`{doc_id}:{block_seq}`；md5 相同 skip（保留旧向量）；永不推断删除
- 日期元数据 doc_date/doc_year/ingested_at + 年份两段式裁决防跨年张冠李戴
- Embedding 双通道：dashscope(text-embedding-v3 或 qwen3.7-text-embedding-flash，均 1024 维) / mock 哈希降级；维度不同切换需清 chroma_db 重灌
- 评估：data/golden 20 条，recall@5/MRR + 答案层引用可回查率；真实 recall@5=0.95 / MRR=0.882 / 可回查率 100%(q17 保留)

## 近期代码演进（2026-09-09）
- 切片质量缓存：chunk_node 切完落盘 data/chunk/{stem}.md（块头标记 chars/block_type/page/section/doc_date/image_ids），调试辅助
- 异步摄取：POST /api/v1/ingest 落盘即返 job_id+ETA，后台 ThreadPoolExecutor(DOCAGENT_INGEST_WORKERS 默认 2) 跑真实管线；GET /status 轮询 + GET /jobs 回查；ETA=pymupdf页数×DOCAGENT_INGEST_SEC_PER_PAGE(默认 2s)
- Chroma 并发加固：store.get_collection 单例 PersistentClient + _store_lock 串行化 diff读+embed+upsert；4 并发零错误
- 模型：用户 .env 切 QWEN_EMBEDDING_MODEL=qwen3.7-text-embedding-flash（1024 维）
- 已知缺口：upload.html「最近上传记录」面板 HTML 已就位，但轮询回填 JS 未接入（服务端 /jobs 可用）；复杂版面 PDF 端到端语义切块待优化（中金案例）

## 验证脚本
- verify_chunking(78)/verify_images_ref(28)/verify_ingest_m1(51)/verify_ingest_m2(15)/verify_server_ingest(9,异步契约)/verify_docqa_m3(14)/verify_docqa_m4(29)/verify_date_metadata(37)

## 防复发
- 并行 Edit 同文件会写回丢失 → 改完一条再改下一条（串行逐条）
- mock 全绿≠真实可跑；切换 provider 必清 chroma_db 重灌
- 任务注册表 _INGEST_JOBS 内存态，重启丢未完成任务（不过度设计，不持久化）
- 切片质量用 data/chunk 缓存人工核对，不查向量库
