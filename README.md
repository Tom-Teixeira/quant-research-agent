# Quant Research Agent

基于 **LangChain + LangGraph + LangSmith + DeepSeek LLM** 构建的量化研究 Agent，
把你（彤天基金实习）的另类数据量化流水线工程化、可编排、可观测：

> 10万+ 公告/新闻清洗 → 1000+ 文本因子 → ML 横截面信号 → 回测与中性组合验证
> **V2 升级：真实数据源（Choice/EMQuantAPI）→ PostgreSQL/pgvector 超大规模入库 → RAG 检索增强**

本仓库提供 **可运行的端到端骨架**：mock 模式无需任何 API key 即可跑通全图；
激活 Choice 量化接口（或配置 DeepSeek / LangSmith key）后切换 live 模式，获得
真实数据驱动的 LLM 研究报告与全链路追踪。

---

## 1. 架构总览：框架分工（V1 + V2）

| 框架/组件 | 承担职责 | 在本项目中的落点 |
|---|---|---|
| **LangChain** | LLM 调用封装、工具定义、Prompt 模板 | `llm.py`（ChatDeepSeek）；`tools/*`（15 个 @tool）；诊断/报告模板 |
| **LangGraph** | 状态机编排：节点、条件边、循环、状态传递 | `graph.py`（V1 六节点）；`graph_v2.py`（V2 全图：入库→因子→门控→回测→RAG） |
| **LangSmith** | 全链路可观测：节点/工具级 tracing | 所有工具 `@traceable`；`config.py` 注入；按 run 可回放 |
| **DeepSeek** | 推理引擎：因子失效诊断、研究报告、RAG 生成 | `langchain-deepseek` 官方集成 |
| **Choice (EMQuantAPI)** | 真实数据源（公告/公司资讯/成分股/行情） | `ingest/data_providers.py`（东方财富 Choice 量化接口，双后端抽象） |
| **PostgreSQL/pgvector** | 超大规模存储 + 向量检索 | `db/`（documents 元数据表 + HNSW 向量索引 + 检索视图） |
| **fastembed(bge)** | 中文公告向量化（512 维 ONNX） | `rag/embedder.py`（BAAI/bge-small-zh-v1.5） |

> 为什么用图而不是普通函数链？普通链无法表达"IC 不达标 → 诊断 → 重建因子 → 再验证"
> 的条件分支与循环，也无法中断后恢复/回放。LangGraph 把状态作为一等公民，节点都是
> `(state_in) -> (state_out)` 纯函数，执行路径完全可追踪、可测试。
>
> **V2 为什么加数据库 + RAG？** 对标 10万+ 公告量级：原始大表直接塞进 LLM 上下文
> 既不经济也易污染（token 爆炸）。V2 把数据沉淀到 pgvector（元数据 + 向量双表），
> 检索时只取 Top-K 结构化证据进 Prompt——吸收 GitHub 优秀项目
> [quant-buddy-skills](https://github.com/pseudo-longinus/quant-buddy-skills) 的
> 「平台侧计算、只返回结构化结果」设计思想（见 §7）。

---

## 2. 实习经历 → Agent 模块映射

| 彤天实习流水线 | Agent 节点 | 核心工具 | 质量口径 |
|---|---|---|---|
| 10万+公告清洗/对齐/校验 | `ingest_node`（V1）/ `ingest_db_node`（V2） | `clean_text_series` `deduplicate_records` `time_align_panel` | 去重/缺失/空文本报告；入库幂等（唯一键去重） |
| 文本情绪/事件强度 → 1000+因子 | `factor_node` | `sentiment_score` `event_intensity` `build_factor_panel` | 因子面板 + 元信息 |
| ML 截面收益预测与信号筛选 | `signal_node` | `train_cross_sectional_model` `evaluate_rank_ic` `evaluate_auc` | 日频 RankIC/ICIR/AUC（对标 3.0%/0.82/+5%） |
| 因子质量人工复核 | 质量门控（条件边） | `diagnose_node`（DeepSeek 诊断） | IC 低于阈值 → 自动诊断重试 |
| 回测与组合稳定性 | `backtest_node` | `run_backtest` `performance_metrics` `neutral_excess` | 年化/夏普/最大回撤/中性超额（含成本） |
| 公告语义检索（新增） | `rag_retrieve_node` `rag_report_node` | pgvector 余弦检索 + DeepSeek 生成 | 时间/代码/类型过滤，证据可溯 |
| 研究报告输出 | `report_node` / `rag_report_node` | DeepSeek 生成 | 结论 + 证据引证 + 风险提示 |

---

## 3. LangGraph 图拓扑（V2）

```
START ─┬─► db_check （并行健康检查，不阻塞）
       │
       ▼
ingest_db_node     ① 真实数据拉取（Choice/文件）→ 清洗 → pgvector 元数据+向量入库
       │
       ▼
factor_node        ② 文本因子构建（读库因子面板或示例数据）
       │
       ▼
signal_node        ③ ML 截面预测 + RankIC/ICIR/AUC
       │
       ├── IC ≥ 阈值 ─────────────►  backtest_node  ④ 分组回测 + 中性超额
       │                                  │
       │                                  ▼
       └── IC < 阈值 && 未超限 ─► diagnose_node ──►（回到 factor_node 重建）
              （重试耗尽放行，诊断保留在报告）      rag_retrieve_node  ⑤ pgvector 语义检索 Top-K
                                                        │
                                                        ▼
                                                rag_report_node   ⑥ 证据 + 指标 → DeepSeek 研究简报
                                                        │
                                                        ▼
                                                       END
```

- 条件边 `quality_gate`：纯 State 驱动；循环上限 `max_retry` 防死循环
- V2 状态扩展：`start_date/end_date/index_code/query/evidence/db_stats`（见 `state.py`）
- 图拓扑 `output/graph.mmd`（V1）；V2 图见 `graph_v2.py` 内注释

---

## 4. 快速开始

### 4.1 一键起库（PostgreSQL + pgvector）

```bash
# 需要 Docker Desktop（中国大陆若拉镜像失败，已内置镜像加速器配置指引）
docker compose -f docker-compose.db.yml up -d
# 校验（预期 ok:true，documents/embeddings 为 0）
python main_v2.py --stage check
```

### 4.2 数据入库（超大规模公告）

```bash
# 文件后端（未安装/激活 Choice 时的开发验证，用示例数据）
python main_v2.py --stage ingest --start 2024-01-01 --end 2024-01-31

# 真实数据后端（安装并激活 Choice 量化接口 EMQuantAPI 后自动切换）
#   → 沪深300成分股(000300.SH) 公告/公司资讯/行情 → 清洗去重 → 分块(512/64) → 向量化 → pgvector
python main_v2.py --stage ingest --start 2018-01-01 --end 2025-12-31
```

### 4.3 RAG 检索增强

```bash
python main_v2.py --stage rag --query "回购增持进展" --ts-code 600100.SH
```

### 4.4 全流程研究（mock 可跑，无 key）

```bash
python main_v2.py --stage research --start 2024-01-01 --end 2024-01-10
```

---

## 5. V2 数据接入层（DataProvider 双后端）

`ingest/data_providers.py` 抽象了统一接口，`get_provider("auto")` 按环境自动切换：

| 后端 | 触发条件 | 覆盖接口（EMQuantAPI） | 用途 |
|---|---|---|---|
| **ChoiceProvider** | 本机已安装并激活 Choice 量化接口（EMQuantAPI 可导入） | `sector`（沪深300成分）、`cfn("report")`（公告）、`cfn("companynews")`（公司资讯）、`csd("CLOSE", 前复权)`（日线→ret_next） | 生产/真实研究 |
| **FileProvider** | 未安装 EMQuantAPI（回退） | 本地 `data/sample_announcements.csv` | 开发验证（显式标注） |

### EMQuantAPI 安装与激活（官方非 pip 包，两步）

1. **安装**：从 https://quantapi.eastmoney.com/ 下载 `EmQuantAPI_Python.zip`；
   Windows 以管理员运行 `EmQuantAPISetup.exe`（选择 Python 环境），
   或命令行运行 `installEmQuantAPI.py` 注册（不能 `pip install`）。
2. **激活**：在 quantapi.eastmoney.com 绑定手机号 → 运行 `LoginActivator.exe`
   生成 `userinfo` 登录令牌；或在 `.env` 配置 `CHOICE_USERNAME/CHOICE_PASSWORD`
   走账密登录（`c.start` 直接账密模式，不生成令牌）。

> 说明：EMQuantAPI 资讯函数 `cfn` 的 content 支持 `report`（公告）/`companynews`
> （公司资讯）/`industrynews`/`regularreport`/`tradeinfo` 等，**暂无研报分类**，
> `get_research_reports` 返回空并显式标注，避免静默缺数据。登录生命周期懒加载
> （首次取数 `c.start`，进程退出 `c.stop`）。账号只存 `.env`，不写入代码/日志。

## 6. V2 数据库设计（PostgreSQL + pgvector）

`db/schema.sql`（容器首次启动自动初始化）：

| 表/视图 | 作用 | 规模设计 |
|---|---|---|
| `documents` | 公告/新闻/研报元数据 | 唯一键 `(doc_type, ts_code, title, ann_date)` 幂等入库；日期/代码/类型索引 |
| `doc_embeddings` | 512 维向量分块 | `UNIQUE(doc_id, chunk_index)`；**HNSW** `vector_cosine_ops` 索引（10万+ 毫秒检索） |
| `factor_panels` | 因子面板落库（date, code 主键） | 衔接一期因子结果 |
| `v_docs` | 文档+向量联查视图 | RAG 检索单表查询 |

- 分块策略：512 字符/块、64 字符重叠（`ingest/pipeline.py::_chunk_text`）
- 嵌入模型：`BAAI/bge-small-zh-v1.5`（fastembed/ONNX，免 PyTorch；首次自动下载 ~100MB，中国大陆走 hf-mirror）
- 幂等验证：重复执行 `--stage ingest` 只补增量（docs_inserted=0，见验证矩阵）

## 7. 学习与吸收：quant-buddy-skills（GitHub 优秀项目）

调研 [pseudo-longinus/quant-buddy-skills](https://github.com/pseudo-longinus/quant-buddy-skills)（197★，MIT）
后，本项目吸收其核心设计思想：

| 其设计 | 吸收到本项目 |
|---|---|
| 四层架构：数据层/指标层/计算层/交付层 | `ingest`（数据）→ `tools`（计算）→ `db`+`rag`（交付链路） |
| 大规模矩阵平台侧计算，只返回结构化结果 | pgvector 只返回 Top-K 证据进 Prompt，避免原始大表进 LLM 上下文 |
| 指标口径可追溯（公式链/数据日期保留） | 因子元信息 + 证据 `ann_date/ts_code/相似度` 全程保留 |
| 研究沉淀为可复用任务 | `main_v2.py --stage ingest/rag/research` 可重复执行、断点续跑 |
| 探索阶段/使用阶段分离 | mock 模式探索验证 → live 模式生产使用 |

## 8. 验证矩阵

| 验证项 | 方式 | 结果 |
|---|---|---|
| V1 正常路径（IC 达标） | `python main.py`（mock） | ✅ 全图跑通，报告落盘 |
| V1 诊断重试循环 | `min_rank_ic=0.99, max_retry=1` | ✅ 触发 diagnose→factor，retry_count=1 |
| 数据库初始化 | `docker compose up -d` + `--stage check` | ✅ 3 表 + pgvector 0.8.7 + v_docs 视图 |
| 数据入库（幂等） | `--stage ingest`（文件后端） | ✅ 143 拉取→114 入库；二次执行 0 新增、向量总数不变 |
| RAG 语义检索 | `--query "业绩预增"` | ✅ Top-3 命中业绩预增公告（sim≈0.73），日期/代码过滤生效 |
| V2 全图（mock） | `python scripts/verify_v2.py` | ✅ 入库→因子(RankIC 0.0718)→门控→回测→RAG 证据 5 条→报告 |
| 嵌入模型 | fastembed ONNX | ✅ bge-small-zh-v1.5 512 维加载验证 |
| LangSmith 追踪 | `LANGSMITH_TRACING=true` | ✅ 工具级 `@traceable` |

## 9. 项目结构（V2）

```
quant-agent/
├── config.py / llm.py / state.py
├── graph.py                # V1 图（6 节点 + 门控 + 诊断循环）
├── graph_v2.py             # V2 图（入库 + RAG 节点扩展）
├── main.py                 # V1 入口（STEP 1-6）
├── main_v2.py              # V2 入口（--stage check/ingest/rag/research）
├── db/                     # PostgreSQL/pgvector 客户端 + schema
│   ├── schema.sql          # 3 表 + HNSW 索引 + v_docs 视图
│   └── db_client.py        # 幂等入库/向量写入/过滤检索/因子落库
├── ingest/                 # 数据接入与入库管线
│   ├── data_providers.py   # DataProvider 抽象：Choice / File 双后端
│   └── pipeline.py         # 拉取→清洗→去重→分块→向量化→入库
├── rag/
│   ├── embedder.py         # bge-small-zh-v1.5（fastembed/ONNX）
│   └── rag_node.py         # 语义检索 + DeepSeek 证据生成
├── tools/                  # 15 个 @tool（清洗/因子/信号/回测）
├── scripts/                # 示例数据生成 + 验证脚本（verify_v2.py）
├── data/sample_announcements.csv
├── docker-compose.db.yml   # pgvector 容器编排
└── output/                 # 图拓扑 + 研究报告
```

## 10. 扩展路线

1. **LLM 工具自主调用**：DeepSeek 绑定 tools 直接改因子配置重跑
2. **DeepSeek-R1 深度诊断**：`DEEPSEEK_MODEL=deepseek-reasoner`
3. **LangSmith 评估集**：历史因子期 benchmark，评估重构前后 IC/ICIR
4. **人机复核门控**：backtest 前 human-in-the-loop（LangGraph interrupt/checkpointer）
5. **定时增量入库**：Choice 增量拉取 + cron 编排（对齐实习的每日公告增量）
6. **向量混合检索**：BM25 稀疏 + pgvector 稠密（吸收 quant-buddy 检索模式）
