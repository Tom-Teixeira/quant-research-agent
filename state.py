"""LangGraph 状态定义（State Schema）。

图内所有节点通过 State 读写共享数据——这是 LangGraph 与普通函数链的
本质区别：节点是纯函数（state_in -> state_out），由图负责状态的
传递、分支与循环，天然可断点续跑、可回放、可被 LangSmith 追踪。
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import add_messages


class QuantState(TypedDict, total=False):
    # ---- 输入层 ----
    raw_data: object          # pd.DataFrame 原始公告数据（JSON 序列化前为 DataFrame）
    data_path: str            # 数据文件路径（支持切换真实数据）

    # ---- 阶段1: 文本清洗 ----
    cleaned_texts: list       # list[dict]: {date, code, industry, text_clean}
    data_quality_report: dict # {rows, dup_removed, missing_filled, outlier_clipped, n_features}

    # ---- 阶段2: 因子构建 ----
    factors: object           # pd.DataFrame: index=[date, code], cols=因子列
    factor_meta: dict         # {n_factors, factor_names}

    # ---- 阶段3: 信号生成 ----
    signal_df: object         # pd.DataFrame: 预测值 + 实际收益 + 分组
    quality: dict             # {rank_ic, icir, auc, passed, message}

    # ---- 阶段4: 回测 ----
    backtest: dict            # {annual_return, sharpe, max_drawdown, excess_ann, excess_sharpe}

    # ---- 阶段5: LLM 报告 ----
    report: str

    # ---- 控制 ----
    retry_count: int          # LLM 诊断-重试循环计数
    diagnostics: list         # LLM 诊断记录（LangSmith 可追踪）

    # ---- V2: 数据库 + RAG ----
    db_status: dict           # 数据库健康检查结果
    db_stats: dict            # 入库统计 {docs_loaded, docs_inserted, embeddings_written, ...}
    start_date: str           # 数据起始日期（YYYY-MM-DD）
    end_date: str             # 数据结束日期（YYYY-MM-DD）
    index_code: str           # 指数代码（成分股来源，如 000300.SH）
    ts_code: str              # RAG 检索股票代码过滤
    query: str                # RAG 检索查询（用户/上游给定）
    evidence: list            # RAG 检索证据集 [{ann_date, name, title, chunk_text, distance, ...}]


# 图内消息历史（预留：若扩展为对话式 agent 可启用 add_messages reducer）
Messages = Annotated[list, add_messages]
