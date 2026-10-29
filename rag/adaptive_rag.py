# Quant Research Agent - 自适应 RAG（Adaptive-RAG）

# 依据：Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language
#   Models through Question Complexity（arXiv:2503.14420）
# 本实现将公告/新闻检索问答路由为三种策略：
#   direct   —— 纯知识性/无需数据的问题 → 不检索，LLM 直接回答
#   simple   —— 单实体/单一事实 → 单次混合检索（HNSW+BM25 RRF，top_k=5）
#   complex  —— 多实体/对比/趋势/时间区间 → 查询扩展 + 多路混合检索 + 证据重排（top_k=8）
# 路由由 LLM 分类（live 模式）；mock/无 LLM 时使用确定性规则，保证全图可跑。

from __future__ import annotations

import re

from db.db_client import QuantDB, reciprocal_rank_fusion
from rag.segment import segment_chinese

# 复杂度规则特征（mock 模式/LLM 分类失败的确定性 fallback）
_COMPLEX_HINTS = (
    "对比", "比较", "差异", "分析", "趋势", "为什么", "哪些", "多少",
    "区间", "期间", "复盘", "归因", "联动", "组合", "两只", "多家",
    "如何", "怎么看", "汇总",
)
_DIRECT_HINTS = (
    "是什么", "什么是", "概念", "定义", "介绍一下", "解释",
    "rankic", "icir", "sharpe", "夏普", "回测", "因子",
)


def classify_query(query: str, llm=None) -> str:
    """查询复杂度分类：direct / simple / complex。

    live 模式优先 LLM 分类（结构化 JSON 输出）；失败或 mock 走规则。
    """
    q = (query or "").strip()
    if not q:
        return "direct"
    if llm is not None:
        try:
            from langchain_core.output_parsers import JsonOutputParser
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "你是量化公告检索的路由器。判断用户查询需要哪种检索策略：\n"
                    "direct：纯概念/定义/方法论问题，无需检索公告（如'什么是RankIC'）；\n"
                    "simple：单只股票/单一事实/简单事件（如'茅台最近回购进展'）；\n"
                    "complex：多实体、对比、趋势、时间区间分析（如'对比A和B的增持公告，分析三季度趋势'）。\n"
                    "只输出 JSON：{\"route\": \"direct|simple|complex\"}"
                )),
                ("human", "{query}"),
            ])
            chain = prompt | llm | JsonOutputParser()
            route = str(chain.invoke({"query": q}).get("route", "")).strip().lower()
            if route in ("direct", "simple", "complex"):
                return route
        except Exception:
            pass  # LLM 分类失败 → 规则兜底
    low = q.lower()
    if any(h in low for h in _DIRECT_HINTS):
        return "direct"
    if any(h in q for h in _COMPLEX_HINTS):
        return "complex"
    return "simple"


def _expand_query(query: str, llm=None) -> list[str]:
    """复杂查询扩展：原文 + 最多 2 个相关子查询（拆实体/条件）。

    live：LLM 生成 1~2 个具体子查询；
    mock/无 LLM：确定性规则——按并列连词拆分实体、提取股票代码。
    """
    queries = [query]
    if llm is not None:
        try:
            from langchain_core.output_parsers import JsonOutputParser
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "把量化公告检索查询扩展为 1~2 个更具体的子查询（保留股票代码与日期），"
                    "只输出 JSON：{\"queries\": [\"子查询1\", \"子查询2\"]}"
                )),
                ("human", "{query}"),
            ])
            chain = prompt | llm | JsonOutputParser()
            extra = [str(s).strip() for s in chain.invoke({"query": query}).get("queries", [])][:2]
            queries += [e for e in extra if e and e != query]
        except Exception:
            pass
    if len(queries) == 1:
        # 规则扩展（仅当 LLM 未产出时）
        parts = [p.strip() for p in re.split(r"[与和、,，以及]", query) if p.strip()]
        if len(parts) > 1:
            queries += parts[:2]
        for code in re.findall(r"\d{6}\.(?:SH|SZ|BJ)", query, re.IGNORECASE):
            queries.append(code.upper())
    out: list[str] = []
    for q in queries:
        if q and q not in out:
            out.append(q)
    return out[:3]


def adaptive_retrieve(
    query: str,
    db: QuantDB,
    embedder,
    llm=None,
    mode: str = "adaptive",
    top_k: int = 5,
    start_date: str | None = None,
    end_date: str | None = None,
    ts_code: str | None = None,
) -> dict:
    """自适应检索：分类 → 路由 → （多路）混合检索 → RRF 证据重排。

    mode:
      adaptive —— 按查询复杂度动态路由（默认）
      hybrid   —— 固定单次 HNSW+BM25 RRF（跳过分类）
      vector   —— 固定纯向量检索（兼容 V2 原行为）
    Returns:
        {"evidence": [...], "route": "direct|simple|complex",
         "queries": [...], "k": 60}
    """
    route = classify_query(query, llm) if mode == "adaptive" else (
        "complex" if mode == "hybrid" else "simple"
    )
    if route == "direct":
        return {"evidence": [], "route": route, "queries": [query], "k": 60}

    queries = _expand_query(query, llm) if route == "complex" else [query]
    query_vec = embedder.embed([segment_chinese(q) for q in queries])
    k = 60  # RRF 常数 + 单路召回深度
    top = 8 if route == "complex" else top_k

    if mode == "vector":
        dense = db.search(query_vec[0], top_k=top, start_date=start_date,
                          end_date=end_date, ts_code=ts_code)
        return {"evidence": dense, "route": route, "queries": queries, "k": k}

    results = [
        db.hybrid_search(segment_chinese(q), vec, top_k=top, k=k,
                         start_date=start_date, end_date=end_date,
                         ts_code=ts_code)
        for q, vec in zip(queries, query_vec)
    ]
    # 多查询证据合并：按 (doc_id) 二次 RRF（以各查询的融合排序为输入）
    merged = reciprocal_rank_fusion([r["evidence"] for r in results], k=k)
    seen: dict[int, dict] = {}
    for doc_id, _score in merged:
        for r in results:
            hit = next((e for e in r["evidence"] if e["doc_id"] == doc_id), None)
            if hit:
                seen[doc_id] = {**hit, "rrf_score": round(_score, 6)}
                break
        if len(seen) >= top:
            break
    evidence = list(seen.values())[:top]
    return {"evidence": evidence, "route": route, "queries": queries, "k": k}
