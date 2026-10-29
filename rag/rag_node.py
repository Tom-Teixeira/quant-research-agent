"""RAG 检索与生成：自适应混合检索（HNSW+BM25 RRF）+ DeepSeek 生成。

RAG 流程（对标实习：公告/新闻 → 因子 → 研究报告）：
  1. 查询复杂度分类（Adaptive-RAG 路由：direct/simple/complex）
  2. 混合检索：pgvector HNSW 稠密 + tsvector BM25 稀疏 → RRF 融合
     （complex 查询自动扩展子查询、多路召回、二次 RRF 证据重排）
  3. 检索结果拼入 Prompt → DeepSeek 生成结构化投资分析
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from config import Settings
from db.db_client import QuantDB
from llm import build_llm
from rag.adaptive_rag import adaptive_retrieve
from rag.embedder import get_embedder


def retrieve_docs(
    query: str,
    top_k: int = 5,
    start_date: str | None = None,
    end_date: str | None = None,
    ts_code: str | None = None,
    doc_type: str | None = None,
    mode: str = "adaptive",
    llm=None,
) -> dict:
    """自适应混合检索（分类 → 路由 → 混合检索 → 证据重排）。"""
    return adaptive_retrieve(
        query, QuantDB(), get_embedder(), llm=llm, mode=mode,
        top_k=top_k, start_date=start_date, end_date=end_date, ts_code=ts_code,
    )


def rag_generate(
    query: str,
    settings: Settings,
    top_k: int = 5,
    start_date: str | None = None,
    end_date: str | None = None,
    ts_code: str | None = None,
    mode: str = "adaptive",
) -> dict:
    """RAG 生成：自适应检索证据 → DeepSeek 生成分析报告。

    mode: adaptive（默认，按复杂度路由）/ hybrid（固定混合检索）/
          vector（纯向量，兼容 V2 原行为）
    Returns:
        {query, evidence: [...], route, queries, report}
    """
    llm = build_llm(settings) if settings.agent_mode == "live" else None
    result = retrieve_docs(query, top_k=top_k, start_date=start_date,
                           end_date=end_date, ts_code=ts_code,
                           mode=mode, llm=llm)
    hits = result["evidence"]

    if not hits:
        if result.get("route") == "direct":
            # 纯知识性问题：不检索，直接回答
            direct = build_llm(settings)
            resp = direct.invoke([
                SystemMessage(content="你是量化研究助手。用户的问题是概念/方法论类，直接给出准确、结构化的中文回答。"),
                HumanMessage(content=query),
            ])
            return {**result, "query": query, "evidence": [], "report": str(resp.content)}
        return {**result, "query": query, "evidence": [], "report": "（未检索到相关公告/新闻）"}

    evidence_lines = []
    for h in hits:
        score_txt = ""
        if "rrf_score" in h and h["rrf_score"] is not None:
            score_txt = f", RRF:{h['rrf_score']}"
        elif "distance" in h:
            score_txt = f", 相似度:{round(1 - h['distance'], 4)}"
        evidence_lines.append(
            f"- [{h['ann_date']}] {h['name']} {h['title']} "
            f"(类型:{h['doc_type']}{score_txt})\n  {str(h.get('chunk_text'))[:200]}"
        )
    evidence = "\n".join(evidence_lines)

    gen_llm = build_llm(settings)
    resp = gen_llm.invoke([
        SystemMessage(content=(
            "你是量化研究员。基于检索到的公告/新闻证据，撰写中文投资分析简报："
            "①关键事件与时间线 ②对基本面/情绪面的影响 ③潜在风险 ④可验证结论。"
            "只能引用证据中出现的事实，不得编造。检索策略为自适应混合检索"
            "（HNSW 稠密 + BM25 稀疏 RRF 融合），复杂查询已扩展子查询。"
        )),
        HumanMessage(content=json.dumps({
            "query": query,
            "route": result.get("route"),
            "queries_used": result.get("queries", [query]),
            "evidence": evidence,
        }, ensure_ascii=False)),
    ])
    return {**result, "query": query, "evidence": hits, "report": str(resp.content)}
