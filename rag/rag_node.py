"""RAG 检索与生成：公告语义检索 + DeepSeek 生成投资分析。

RAG 流程（对标实习：公告/新闻 → 因子 → 研究报告）：
  1. 用户/上游给定查询（如"某某公司回购进展"或股票代码）
  2. pgvector 语义检索 top-k 公告（支持时间/代码/类型过滤）
  3. 检索结果拼入 Prompt → DeepSeek 生成结构化投资分析
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from config import Settings
from db.db_client import QuantDB
from llm import build_llm
from rag.embedder import get_embedder


def retrieve_docs(
    query: str,
    top_k: int = 5,
    start_date: str | None = None,
    end_date: str | None = None,
    ts_code: str | None = None,
    doc_type: str | None = None,
) -> list[dict]:
    """向量检索（查询 → 嵌入 → pgvector 余弦相似 top-k）。"""
    vec = get_embedder().embed_one(query)
    return QuantDB().search(vec, top_k=top_k, start_date=start_date,
                            end_date=end_date, ts_code=ts_code, doc_type=doc_type)


def rag_generate(
    query: str,
    settings: Settings,
    top_k: int = 5,
    start_date: str | None = None,
    end_date: str | None = None,
    ts_code: str | None = None,
) -> dict:
    """RAG 生成：检索证据 → DeepSeek 生成分析报告。

    Returns:
        {query, evidence: [...], report}
    """
    hits = retrieve_docs(query, top_k=top_k, start_date=start_date,
                         end_date=end_date, ts_code=ts_code)
    if not hits:
        return {"query": query, "evidence": [], "report": "（未检索到相关公告/新闻）"}

    evidence_lines = []
    for h in hits:
        evidence_lines.append(
            f"- [{h['ann_date']}] {h['name']} {h['title']} "
            f"(类型:{h['doc_type']}, 相似度:{round(1 - h['distance'], 4)})\n  {h['chunk_text'][:200]}"
        )
    evidence = "\n".join(evidence_lines)

    llm = build_llm(settings)
    resp = llm.invoke([
        SystemMessage(content=(
            "你是量化研究员。基于检索到的公告/新闻证据，撰写中文投资分析简报："
            "①关键事件与时间线 ②对基本面/情绪面的影响 ③潜在风险 ④可验证结论。"
            "只能引用证据中出现的事实，不得编造。"
        )),
        HumanMessage(content=json.dumps({
            "query": query,
            "evidence": evidence,
        }, ensure_ascii=False)),
    ])
    return {"query": query, "evidence": hits, "report": str(resp.content)}
