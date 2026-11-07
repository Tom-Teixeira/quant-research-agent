from __future__ import annotations

import json
from functools import partial

import numpy as np
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
)


def ingest_node(state: QuantState, settings: Settings) -> dict:
    path = state.get("data_path") or settings.data_path
    df = pd.read_csv(path)
    if not {"date", "code", "text"}.issubset(df.columns):
        raise ValueError(f"missing date/code/text columns, got {list(df.columns)}")

    df = deduplicate_records.func(df)
    df["text"] = clean_text_series.func(df["text"])
    df = time_align_panel.func(df)
    cleaned = df[["date", "code", "text"]].copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"])
    return {
        "raw_data": df,
        "cleaned_texts": cleaned.to_dict("records"),
        "data_quality_report": quality_report.func(df),
    }


def factor_node(state: QuantState, settings: Settings = None) -> dict:
    panel = build_factor_panel.func(state["raw_data"])
    names = [c for c in panel.columns if c not in {"date", "code"}]
    return {
        "factors": panel,
        "factor_meta": {"n_factors": len(names), "factor_names": names},
    }


def signal_node(state: QuantState, settings: Settings) -> dict:
    panel = state["factors"]
    if "ret_next" not in panel.columns:
        panel["ret_next"] = _synthetic_labels(panel)
    trained = train_cross_sectional_model.func(panel, panel)
    test_df = trained["test_df"]

    ic = evaluate_rank_ic.func(test_df["pred_prob"], test_df["ret_next"], test_df["date"])
    ic["auc"] = evaluate_auc.func(test_df)

    passed = ic["rank_ic"] >= settings.min_rank_ic and ic["daily_ic_count"] > 0
    ic.update({"passed": bool(passed), "threshold": settings.min_rank_ic, "max_retry": settings.max_retry})
    return {"signal_df": test_df, "quality": ic}


def diagnose_node(state: QuantState, settings: Settings) -> dict:
    llm = build_llm(settings)
    quality = state["quality"]
    diag = llm.invoke([
        SystemMessage(content="Diagnose why the factor IC is below threshold. Give up to three concrete fixes."),
        HumanMessage(content=json.dumps({
            "quality": {k: v for k, v in quality.items() if k != "passed"},
            "factor_names": state["factor_meta"]["factor_names"],
            "retry_count": state.get("retry_count", 0),
        }, default=str)),
    ])
    return {
        "diagnostics": state.get("diagnostics", []) + [str(diag.content)],
        "retry_count": state.get("retry_count", 0) + 1,
    }


def backtest_node(state: QuantState) -> dict:
    signal_df = state["signal_df"]
    bt = run_backtest.func(signal_df)
    excess = neutral_excess.func(signal_df, _benchmark_proxy(signal_df))
    return {"backtest": {"long_short": bt["metrics"], "neutral": excess}}


def report_node(state: QuantState, settings: Settings) -> dict:
    llm = build_llm(settings)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="Write a research summary covering data quality, factors, signal, backtest, and risks. Cite the numbers."),
        HumanMessage(content=json.dumps({
            "data_quality": state["data_quality_report"],
            "factor_meta": state["factor_meta"],
            "quality": {k: v for k, v in state["quality"].items() if k != "passed"},
            "backtest": state["backtest"],
            "diagnostics": state.get("diagnostics", []),
        }, default=str)),
    ])
    resp = llm.invoke(prompt.invoke({}))
    return {"report": str(resp.content)}


def quality_gate(state: QuantState) -> str:
    if state["quality"]["passed"]:
        return "backtest"
    if state.get("retry_count", 0) < state["quality"].get("max_retry", 1):
        return "diagnose"
    return "backtest"


def _synthetic_labels(panel: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.random.default_rng(7).standard_normal(panel.shape[0]),
        index=panel.index,
    ) * 0.01


def _benchmark_proxy(signal_df: pd.DataFrame) -> pd.Series:
    return signal_df.groupby("date")["ret_next"].mean().fillna(0.0)


def build_graph(settings: Settings | None = None) -> StateGraph:
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
    g.add_edge("diagnose", "factor")
    g.add_edge("backtest", "report")
    g.add_edge("report", END)
    return g.compile()


def build_agent(settings: Settings | None = None):
    settings = settings or get_settings()
    settings.apply_langsmith_env()
    return build_graph(settings)
