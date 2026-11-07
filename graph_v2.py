from __future__ import annotations

from functools import partial
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from config import Settings, get_settings
from db.db_client import QuantDB, health_check
from graph import (
    backtest_node,
    build_llm,
    diagnose_node,
    factor_node,
    quality_gate,
    signal_node,
)
from rag.rag_node import retrieve_docs
from state import QuantState


def db_check_node(state: QuantState, settings: Settings) -> dict:
    return {"db_status": health_check()}


def factor_node_v2(state: QuantState, settings: Settings) -> dict:
    import pandas as pd

    db = QuantDB()
    try:
        with db.conn() as conn:
            cur = conn.execute(
                "SELECT date, code, sentiment_mean, sentiment_std, event_mean, event_max, "
                "n_announcements, tone_vol, ret_next FROM factor_panels ORDER BY date"
            )
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
        if rows:
            state["raw_data"] = pd.DataFrame([dict(zip(cols, r)) for r in rows])
            return factor_node(state, settings)
    except Exception:
        pass

    sample = Path(__file__).resolve().parent / "data" / "sample_announcements.csv"
    state["raw_data"] = pd.read_csv(sample)
    return factor_node(state, settings)


def ingest_db_node(state: QuantState, settings: Settings) -> dict:
    from ingest.pipeline import run_ingest_pipeline

    stats = run_ingest_pipeline(
        state.get("start_date", "2018-01-01"),
        state.get("end_date", "2025-12-31"),
        index_code=state.get("index_code", "000300.SH"),
    )
    return {"db_stats": stats}


def rag_retrieve_node(state: QuantState, settings: Settings) -> dict:
    llm = build_llm(settings) if settings.mode == "live" else None
    result = retrieve_docs(
        state.get("query") or "回购增持进展",
        top_k=5,
        start_date=state.get("start_date"),
        end_date=state.get("end_date"),
        ts_code=state.get("ts_code"),
        mode=state.get("rag_mode", "adaptive"),
        llm=llm,
    )
    return {"evidence": result["evidence"], "rag_route": result.get("route")}


def rag_report_node(state: QuantState, settings: Settings) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = build_llm(settings)
    evidence = state.get("evidence", [])
    lines = []
    for h in evidence:
        score = h.get("rrf_score")
        tag = f"RRF {score}" if score is not None else (
            f"sim {round(1 - h['distance'], 4)}" if "distance" in h else "hybrid"
        )
        lines.append(f"- [{h['ann_date']}] {h['name']} {h['title']} ({tag})\n  {str(h.get('chunk_text'))[:180]}")

    payload = {
        "query": state.get("query", ""),
        "rag_route": state.get("rag_route"),
        "metrics": {
            "rank_ic": state["quality"].get("rank_ic"),
            "icir": state["quality"].get("icir"),
            "annual_return": state["backtest"]["long_short"].get("annual_return"),
            "sharpe": state["backtest"]["long_short"].get("sharpe"),
        },
        "evidence": "\n".join(lines) if lines else "(none)",
    }
    resp = llm.invoke([
        SystemMessage(content=(
            "Write a research summary from the metrics and retrieved evidence. "
            "Cite source dates and companies. Do not invent evidence."
        )),
        HumanMessage(content=repr(payload)),
    ])
    return {"report": str(resp.content)}


def build_v2_graph(settings: Settings | None = None) -> StateGraph:
    settings = settings or get_settings()
    bind = lambda fn: partial(fn, settings=settings)  # noqa: E731

    g = StateGraph(QuantState)
    g.add_node("db_check", bind(db_check_node))
    g.add_node("ingest_db", bind(ingest_db_node))
    g.add_node("factor", bind(factor_node_v2))
    g.add_node("signal", bind(signal_node))
    g.add_node("diagnose", bind(diagnose_node))
    g.add_node("backtest", backtest_node)
    g.add_node("rag_retrieve", bind(rag_retrieve_node))
    g.add_node("rag_report", bind(rag_report_node))

    g.add_edge(START, "ingest_db")
    g.add_edge("ingest_db", "factor")
    g.add_edge("factor", "signal")
    g.add_conditional_edges("signal", quality_gate, {"backtest": "backtest", "diagnose": "diagnose"})
    g.add_edge("diagnose", "factor")
    g.add_edge("backtest", "rag_retrieve")
    g.add_edge("rag_retrieve", "rag_report")
    g.add_edge("rag_report", END)
    g.add_edge(START, "db_check")
    g.add_edge("db_check", END)
    return g.compile()


def build_v2_agent(settings: Settings | None = None):
    settings = settings or get_settings()
    settings.apply_langsmith_env()
    return build_v2_graph(settings)
