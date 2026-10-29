"""数据管道：Provider → 清洗 → 入库(pgvector) → 分块向量化 → RAG 就绪。

对标实习数据体系构建流程：拉取 → 文本清洗 → 时间对齐 → 入库 → 检索增强。
超大规模（10万+）策略：
  - 分页/分代码拉取，批量 upsert（幂等，可断点续跑）
  - 文本分块（按段落/长度），逐文档向量化后批量写 pgvector HNSW
"""
from __future__ import annotations

import os
from datetime import date

from db.db_client import QuantDB
from ingest.data_providers import get_provider
from rag.embedder import get_embedder
from tools.text_tools import clean_text_series, deduplicate_records, time_align_panel


def _chunk_text(text: str, max_len: int = 512, overlap: int = 64) -> list[str]:
    """滑动窗口分块（中文公告/新闻）：512 字符/块，64 字符重叠。"""
    text = text or ""
    if len(text) <= max_len:
        return [text] if text.strip() else []
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + max_len])
        start += max_len - overlap
    return chunks


def run_ingest_pipeline(
    start_date: str,
    end_date: str,
    index_code: str = "000300.SH",
    provider_name: str = "auto",
    db: QuantDB | None = None,
) -> dict:
    """全量入库管线：成分股 → 公告/新闻/研报 → 清洗去重 → 元数据+向量入库。

    Returns:
        {docs_loaded, docs_inserted, embeddings_written, total_documents, total_embeddings}
    """
    provider = get_provider(provider_name)
    db = db or QuantDB()
    embedder = get_embedder()

    # 1. 成分股与数据拉取
    codes = provider.get_index_constituents(index_code)
    if not codes:
        codes = provider.get_announcements([], start_date, end_date) or []
        raise ValueError("未获取到成分股列表，请检查数据源（Choice 环境或本地文件）")

    announcements = provider.get_announcements(codes, start_date, end_date)
    news = provider.get_news(start_date, end_date)
    reports = provider.get_research_reports(codes, start_date, end_date)
    all_docs = announcements + news + reports
    print(f"[ingest] 拉取: 公告 {len(announcements)} / 新闻 {len(news)} / 研报 {len(reports)}，合计 {len(all_docs)}")

    if not all_docs:
        print("[ingest] 无数据（检查日期范围与数据源）")
        return {"docs_loaded": 0, "docs_inserted": 0, "embeddings_written": 0,
                "total_documents": db.count_documents(), "total_embeddings": db.embedding_count()}

    # 2. 清洗与去重（复用一期文本工具；工具要求 date/code/text 列名）
    df = _docs_to_frame(all_docs)
    df = df.rename(columns={"ann_date": "date", "ts_code": "code"})
    df = deduplicate_records.func(df, keys=("date", "code", "title"))
    df["text"] = clean_text_series.func(df["text"])
    df = time_align_panel.func(df)
    df = df.rename(columns={"date": "ann_date", "code": "ts_code"})

    # 3. 元数据批量入库（幂等；content_seg 为 jieba 分词，供 BM25 稀疏检索）
    from rag.segment import segment_chinese

    doc_records = [
        {
            "doc_type": r["doc_type"], "ts_code": r["ts_code"], "name": r["name"],
            "title": r["title"], "content": r["text"], "ann_date": r["ann_date"],
            "source_url": r.get("source_url", ""),
            "content_seg": segment_chinese(r["text"]),
            "extra": {"chunk_count": len(_chunk_text(r["text"]))},
        }
        for _, r in df.iterrows()
    ]
    inserted = db.upsert_documents(doc_records)

    # 4. 分块 → 向量化 → 批量写 pgvector
    embeddings_written = 0
    with db.conn() as conn:
        for r in df.iterrows():
            _, rec = r
            chunks = _chunk_text(rec["text"])
            if not chunks:
                continue
            vecs = embedder.embed(chunks)
            doc_row = conn.execute(
                "SELECT id FROM documents WHERE doc_type=%s AND ts_code=%s AND title=%s AND ann_date=%s",
                (rec["doc_type"], rec["ts_code"], rec["title"], rec["ann_date"]),
            ).fetchone()
            if doc_row:
                db.upsert_embeddings(doc_row[0], [(i, t, v) for i, (t, v) in enumerate(zip(chunks, vecs))])
                embeddings_written += len(chunks)

    stats = {
        "docs_loaded": len(all_docs),
        "docs_inserted": inserted,
        "embeddings_written": embeddings_written,
        "total_documents": db.count_documents(),
        "total_embeddings": db.embedding_count(),
    }
    print(f"[ingest] 入库完成: {stats}")
    return stats


def _docs_to_frame(docs: list[dict]):
    import pandas as pd

    df = pd.DataFrame(docs)
    if df.empty:
        return df
    df = df.rename(columns={"content": "text"})
    df["text"] = df["text"].fillna(df["title"])
    return df
