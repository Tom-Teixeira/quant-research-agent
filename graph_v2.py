"""V2 图：在 V1 量化流水线上并入「数据库入库 + RAG 检索增强」节点。

拓扑（V2）：
  START → ingest_db ─► factor ─► signal ─(gate)─► backtest ─► rag_retrieve ─► rag_report ─► END
                   │         （条件边/诊断循环与 V1 一致）
                   └─► db_check（并行健康检查，失败不阻塞）

RAG 环节：把回测涉及股票的公告/新闻做语义检索，检索证据并入 DeepSeek 最终报告，
实现"数据驱动 + 证据可溯"的量化研究闭环。
"""
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
    ingest_node,
    quality_gate,
    report_node,
    signal_node,
)
from rag.embedder import get_embedder
from rag.rag_node import retrieve_docs
from state import QuantState


# ---------------------------------------------------------------------------
# V2 新增节点
# ---------------------------------------------------------------------------


def db_check_node(state: QuantState, settings: Settings) -> dict:
    """数据库健康检查（只读）。失败仅记录，不阻塞主流程。"""
    hc = health_check()
    return {"db_status": hc}


def factor_node_v2(state: QuantState, settings: Settings) -> dict:
    """V2 因子节点：优先使用数据库因子面板（factor_panels），空则回退示例数据。

    真实大规模场景：面板由入库公告的文本因子离线计算后落库，本节点只做
    「读库 → 训练 → 评估」，避免把大表塞进图状态（吸收 quant-buddy 平台侧计算思想）。
    """
    import pandas as pd

    db = QuantDB()
    try:
        with db.conn() as conn:
            cur = conn.execute("SELECT date, code, sentiment_mean, sentiment_std, event_mean, event_max, n_announcements, tone_vol, ret_next FROM factor_panels ORDER BY date")
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
        if rows:
            panel = pd.DataFrame([dict(zip(cols, r)) for r in rows])
            state["raw_data"] = panel  # 库面板 → V1 因子训练
            return factor_node(state, settings)
    except Exception:
        pass  # 库不可用/无面板 → 回退示例数据

    # 回退：示例公告数据（mock 演示链路，指标虚高仅验管线）
    sample = str(Path(__file__).resolve().parent / "data" / "sample_announcements.csv")
    state["raw_data"] = pd.read_csv(sample)
    return factor_node(state, settings)


def ingest_db_node(state: QuantState, settings: Settings) -> dict:
    """数据入库节点：Provider 拉取 → 清洗 → pgvector 元数据+向量入库。

    幂等设计：重复执行只补增量（对标 10万+ 公告的分批入库）。
    """
    from ingest.pipeline import run_ingest_pipeline

    start = state.get("start_date", "2018-01-01")
    end = state.get("end_date", "2025-12-31")
    index = state.get("index_code", "000300.SH")
    stats = run_ingest_pipeline(start, end, index_code=index)
    return {"db_stats": stats}


def rag_retrieve_node(state: QuantState, settings: Settings) -> dict:
    """RAG 检索：自适应混合检索（HNSW+BM25 RRF，按查询复杂度路由）。"""
    query = state.get("query") or "回购增持进展"
    llm = build_llm(settings) if settings.mode == "live" else None
    result = retrieve_docs(query, top_k=5,
                           start_date=state.get("start_date"),
                           end_date=state.get("end_date"),
                           ts_code=state.get("ts_code"),
                           mode=state.get("rag_mode", "adaptive"),
                           llm=llm)
    return {"evidence": result["evidence"], "rag_route": result.get("route")}


def rag_report_node(state: QuantState, settings: Settings) -> dict:
    """RAG 报告：检索证据 + 量化指标 → DeepSeek 生成带引证的最终报告。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = build_llm(settings)
    evidence = state.get("evidence", [])
    ev_lines = []
    for h in evidence:
        score = h.get("rrf_score")
        score_txt = f"RRF {score}" if score is not None else (
            f"相似度 {round(1 - h['distance'], 4)}" if "distance" in h else "混合"
        )
        ev_lines.append(f"- [{h['ann_date']}] {h['name']} {h['title']} ({score_txt})\n  {str(h.get('chunk_text'))[:180]}")
    payload = {
        "query": state.get("query", ""),
        "rag_route": state.get("rag_route"),
        "quant_metrics": {
            "rank_ic": state["quality"].get("rank_ic"),
            "icir": state["quality"].get("icir"),
            "annual_return": state["backtest"]["long_short"].get("annual_return"),
            "sharpe": state["backtest"]["long_short"].get("sharpe"),
        },
        "evidence": "\n".join(ev_lines) if ev_lines else "（无检索证据）",
    }
    resp = llm.invoke([
        SystemMessage(content=(
            "你是量化研究分析师。综合①量化指标（因子IC/回测绩效）与②公告/新闻检索证据，"
            "撰写最终研究简报：数据与因子表现、证据支撑的关键事件、风险提示、可验证结论。"
            "证据必须标注来源日期与公司，不得编造。"
        )),
        HumanMessage(content=repr(payload)),
    ])
    return {"report": str(resp.content)}


# ---------------------------------------------------------------------------
# V2 图装配
# ---------------------------------------------------------------------------


def build_v2_graph(settings: Settings | None = None) -> StateGraph:
    """V2 全流程图：ingest_db → factor → signal →(gate)→ backtest → rag_retrieve → rag_report。"""
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
    g.add_conditional_edges("signal", quality_gate,
                            {"backtest": "backtest", "diagnose": "diagnose"})
    g.add_edge("diagnose", "factor")      # 诊断重试循环（V1 复用）
    g.add_edge("backtest", "rag_retrieve")
    g.add_edge("rag_retrieve", "rag_report")
    g.add_edge("rag_report", END)
    # 并行健康检查（不阻塞主链）
    g.add_edge(START, "db_check")
    g.add_edge("db_check", END)
    return g.compile()


def build_v2_agent(settings: Settings | None = None):
    settings = settings or get_settings()
    settings.apply_langsmith_env()
    return build_v2_graph(settings)
