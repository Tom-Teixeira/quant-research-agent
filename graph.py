"""LangGraph 编排层：量化研究 Agent 的完整工作流。

图拓扑（对应实习流水线）：

    START
      │
      ▼
   ingest_node       阶段1 数据摄取 + 文本清洗（工具：clean/dedupe/align/winsorize/standardize）
      │
      ▼
   factor_node       阶段2 因子构建（工具：sentiment_score/event_intensity/build_factor_panel）
      │
      ▼
   signal_node       阶段3 ML 截面预测 + RankIC/ICIR/AUC 评估
      │
      ├── IC 不达标 ──► diagnose_node（DeepSeek 诊断）──► factor_node（重试，retry+1）
      │                    ▲                                    │（retry 达上限则放行）
      ▼                    └────────────────────────────────────┘
   backtest_node     阶段4 回测（工具：run_backtest/performance_metrics/neutral_excess）
      │
      ▼
   report_node       阶段5 DeepSeek 生成研究报告
      │
      ▼
     END

LangGraph 承担：状态传递（State）、条件边（quality gate）、循环（诊断-重试）。
LangChain 承担：@tool 工具定义、LLM 调用封装、prompt 模板。
LangSmith 承担：@traceable 全链路追踪、节点/工具级观测。
"""
from __future__ import annotations

import json

from functools import partial

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from config import Settings, get_settings
from llm import build_llm
from state import QuantState
from tools import (
    build_factor_panel,
    clean_text_series,
    deduplicate_records,
    evaluate_auc,
    evaluate_rank_ic,
    neutral_excess,
    quality_report,
    run_backtest,
    time_align_panel,
    train_cross_sectional_model,
    winsorize_series,
)

# ---------------------------------------------------------------------------
# 节点函数（LangGraph 约定：输入 state，返回 dict 增量写入 state）
# ---------------------------------------------------------------------------


def ingest_node(state: QuantState, settings: Settings) -> dict:
    """阶段1：数据摄取 + 文本清洗（对标实习文本清洗/时间对齐/质量校验）。"""
    path = state.get("data_path") or settings.data_path
    df = pd.read_csv(path)
    if not {"date", "code", "text"}.issubset(df.columns):
        raise ValueError(f"数据缺少 date/code/text 列，实际列: {list(df.columns)}")

    df = deduplicate_records.func(df)
    df["text"] = clean_text_series.func(df["text"])
    df = time_align_panel.func(df)
    # 清洗后文本转存（供因子节点）
    cleaned_texts = df[["date", "code", "text"]].copy()
    cleaned_texts["date"] = pd.to_datetime(cleaned_texts["date"])
    return {
        "raw_data": df,
        "cleaned_texts": cleaned_texts.to_dict("records"),
        "data_quality_report": quality_report.func(df),
    }


def factor_node(state: QuantState, settings: Settings = None) -> dict:
    """阶段2：文本因子构建。输出 (date, code) 面板因子。
    settings 由 build_graph 注入（本节点暂不使用，保持签名一致）。"""
    raw: pd.DataFrame = state["raw_data"]
    panel = build_factor_panel.func(raw)
    return {
        "factors": panel,
        "factor_meta": {
            "n_factors": int(len([c for c in panel.columns if c not in {"date", "code"}])),
            "factor_names": [c for c in panel.columns if c not in {"date", "code"}],
        },
    }


def signal_node(state: QuantState, settings: Settings) -> dict:
    """阶段3：ML 截面预测 + 质量评估。"""
    panel: pd.DataFrame = state["factors"]
    # 标签：面板内模拟的次日收益列（真实数据由用户提供 ret_next）
    if "ret_next" not in panel.columns:
        panel["ret_next"] = _synthetic_labels(panel)
    trained = train_cross_sectional_model.func(panel, panel)
    test_df: pd.DataFrame = trained["test_df"]

    ic = evaluate_rank_ic.func(test_df["pred_prob"], test_df["ret_next"], test_df["date"])
    auc = evaluate_auc.func(test_df)
    ic.update({"auc": auc})

    passed = ic["rank_ic"] >= settings.min_rank_ic and ic["daily_ic_count"] > 0
    ic.update({"passed": bool(passed), "threshold": settings.min_rank_ic, "max_retry": settings.max_retry})

    return {"signal_df": test_df, "quality": ic}


def diagnose_node(state: QuantState, settings: Settings) -> dict:
    """条件边命中时：DeepSeek 诊断因子质量并给出调整建议。

    诊断结果追加到 state.diagnostics（LangSmith 全量记录），
    通过 LangGraph 循环回到 factor_node 重试。
    """
    llm = build_llm(settings)
    quality = state["quality"]
    diag = llm.invoke([
        SystemMessage(content="你是量化研究员。基于 RankIC/ICIR/AUC 诊断因子失效原因，给出≤3条可执行的因子重构建议。"),
        HumanMessage(content=json.dumps({
            "quality": {k: v for k, v in quality.items() if k != "passed"},
            "factor_names": state["factor_meta"]["factor_names"],
            "retry_count": state.get("retry_count", 0),
        }, ensure_ascii=False, default=str)),
    ])
    return {"diagnostics": state.get("diagnostics", []) + [str(diag.content)], "retry_count": state.get("retry_count", 0) + 1}


def backtest_node(state: QuantState) -> dict:
    """阶段4：回测 + 绩效归因 + 中性超额。"""
    signal_df: pd.DataFrame = state["signal_df"]
    bt = run_backtest.func(signal_df)
    bench = _benchmark_proxy(signal_df)
    excess = neutral_excess.func(signal_df, bench)
    return {"backtest": {"long_short": bt["metrics"], "neutral": excess}}


def report_node(state: QuantState, settings: Settings) -> dict:
    """阶段5：DeepSeek 生成研究报告（含全部中间指标）。"""
    llm = build_llm(settings)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "你是量化研究分析师。基于以下结构化结果撰写中文研究简报，"
            "包含：数据质量、因子表现、信号质量、回测绩效、结论与风险提示。"
            "要求分点、数据精确引用。"
        )),
        HumanMessage(content=json.dumps({
            "data_quality": state["data_quality_report"],
            "factor_meta": state["factor_meta"],
            "quality": {k: v for k, v in state["quality"].items() if k != "passed"},
            "backtest": state["backtest"],
            "diagnostics": state.get("diagnostics", []),
        }, ensure_ascii=False, default=str)),
    ])
    resp = llm.invoke(prompt.invoke({}))
    return {"report": str(resp.content)}


# ---------------------------------------------------------------------------
# 条件边路由
# ---------------------------------------------------------------------------


def quality_gate(state: QuantState) -> str:
    """IC 质量门控（纯 State 驱动）：达标 → backtest；不达标且未超限 → diagnose 重试。"""
    if state["quality"]["passed"]:
        return "backtest"
    if state.get("retry_count", 0) < state["quality"].get("max_retry", 1):
        return "diagnose"
    return "backtest"  # 重试耗尽，放行并在报告中保留诊断记录


# ---------------------------------------------------------------------------
# 辅助：合成标签与基准（仅用于示例数据跑通；真实数据请替换）
# ---------------------------------------------------------------------------


def _synthetic_labels(panel: pd.DataFrame) -> pd.Series:
    """无真实收益时，用因子噪声生成次日收益标签，保证 ML 环节可运行。"""
    rng = pd.Series(np_random(panel.shape[0]), index=panel.index)
    return rng * 0.01


def np_random(n: int, seed: int = 7):
    import numpy as np
    return np.random.default_rng(seed).standard_normal(n)


def _benchmark_proxy(signal_df: pd.DataFrame) -> pd.Series:
    """基准代理：全样本等权日收益（演示用；真实场景用中证1000指数收益）。"""
    import numpy as np
    return signal_df.groupby("date")["ret_next"].mean().fillna(0.0)


# ---------------------------------------------------------------------------
# 图构建
# ---------------------------------------------------------------------------


def build_graph(settings: Settings | None = None) -> StateGraph:
    """按 START→ingest→factor→signal→(gate)→backtest→report→END 装配图。

    settings 经 functools.partial 注入各节点（节点签名 (state, settings)），
    保证自定义阈值/模式贯穿全图，且条件边 quality_gate 纯 State 驱动。
    """
    settings = settings or get_settings()
    bind = lambda fn: partial(fn, settings=settings)  # noqa: E731
    g = StateGraph(QuantState)
    g.add_node("ingest", bind(ingest_node))
    g.add_node("factor", bind(factor_node))
    g.add_node("signal", bind(signal_node))
    g.add_node("diagnose", bind(diagnose_node))
    g.add_node("backtest", backtest_node)
    g.add_node("report", bind(report_node))

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "factor")
    g.add_edge("factor", "signal")
    g.add_conditional_edges("signal", quality_gate, {"backtest": "backtest", "diagnose": "diagnose"})
    g.add_edge("diagnose", "factor")      # 重试循环：诊断 → 因子重建
    g.add_edge("backtest", "report")
    g.add_edge("report", END)
    return g.compile()


def build_agent(settings: Settings | None = None):
    """编译后的可调用图（返回 CompiledStateGraph）。"""
    settings = settings or get_settings()
    settings.apply_langsmith_env()
    return build_graph(settings)
