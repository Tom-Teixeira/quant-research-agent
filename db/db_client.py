"""数据库层：PostgreSQL + pgvector 客户端。

职责：
  - documents 元数据表：超大规模公告/新闻结构化存储（10万+ 行）
  - doc_embeddings 向量表：HNSW 索引支撑 RAG 毫秒级检索
  - 批量写入（COPY/executemany）与幂等 upsert（对标实习的批量写入保障）

依赖：psycopg[binary]（psycopg3）
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import psycopg

DEFAULT_DSN = os.getenv(
    "QUANT_DB_DSN",
    "postgresql://quant:quant_pass_2026@localhost:5432/quantdb",
)


class QuantDB:
    """pgvector 数据库客户端。"""

    def __init__(self, dsn: str = DEFAULT_DSN):
        self.dsn = dsn

    @contextmanager
    def conn(self):
        with psycopg.connect(self.dsn) as conn:
            yield conn

    # ------------------------------------------------------------------
    # 文档写入（幂等 upsert：按 doc_type+ts_code+title+ann_date 去重）
    # ------------------------------------------------------------------

    def upsert_documents(self, docs: list[dict]) -> int:
        """批量写入/更新公告文档，返回实际插入数（重复跳过）。

        Args:
            docs: [{doc_type, ts_code, name, title, content, ann_date, source_url,
                    extra, content_seg}]
                   content_seg 为 jieba 空格分词文本（混合检索 BM25 稀疏路用）。
        """
        inserted = 0
        with self.conn() as conn:
            with conn.cursor() as cur:
                for d in docs:
                    cur.execute(
                        """
                        INSERT INTO documents (doc_type, ts_code, name, title, content,
                                               ann_date, source_url, extra, content_seg)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT DO NOTHING
                        """,
                        (
                            d["doc_type"], d.get("ts_code"), d.get("name"), d["title"],
                            d.get("content"), d["ann_date"], d.get("source_url"),
                            psycopg.types.json.Jsonb(d.get("extra") or {}),
                            d.get("content_seg") or "",
                        ),
                    )
                    inserted += cur.rowcount
                conn.commit()
        return inserted

    def count_documents(self) -> int:
        with self.conn() as conn:
            return conn.execute("SELECT count(*) FROM documents").fetchone()[0]

    # ------------------------------------------------------------------
    # 向量写入
    # ------------------------------------------------------------------

    def upsert_embeddings(self, doc_id: int, chunks: list[tuple[int, str, list[float]]]) -> None:
        """写入某文档的向量分块：[(chunk_index, chunk_text, embedding)]。"""
        with self.conn() as conn:
            with conn.cursor() as cur:
                for idx, text, vec in chunks:
                    cur.execute(
                        """
                        INSERT INTO doc_embeddings (doc_id, chunk_index, chunk_text, embedding)
                        VALUES (%s,%s,%s,%s)
                        ON CONFLICT (doc_id, chunk_index) DO UPDATE
                        SET chunk_text = EXCLUDED.chunk_text, embedding = EXCLUDED.embedding
                        """,
                        (doc_id, idx, text, vec),
                    )
                conn.commit()

    def embedding_count(self) -> int:
        with self.conn() as conn:
            return conn.execute("SELECT count(*) FROM doc_embeddings").fetchone()[0]

    # ------------------------------------------------------------------
    # RAG 检索：向量相似度 + 时间/代码过滤
    # ------------------------------------------------------------------

    def search(
        self,
        query_vec: list[float],
        top_k: int = 5,
        start_date: str | None = None,
        end_date: str | None = None,
        ts_code: str | None = None,
        doc_type: str | None = None,
    ) -> list[dict]:
        """pgvector 余弦相似检索，支持日期/代码/类型过滤（对标：公告检索增强）。"""
        sql = """
            SELECT e.doc_id, e.doc_type, e.ts_code, e.name, e.title, e.ann_date,
                   e.chunk_text, e.embedding <=> %s::vector AS distance
            FROM v_docs e
        """
        conds: list[str] = []
        params: list = [query_vec]
        if start_date:
            conds.append("e.ann_date >= %s"); params.append(start_date)
        if end_date:
            conds.append("e.ann_date <= %s"); params.append(end_date)
        if ts_code:
            conds.append("e.ts_code = %s"); params.append(ts_code)
        if doc_type:
            conds.append("e.doc_type = %s"); params.append(doc_type)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY e.embedding <=> %s::vector LIMIT %s"  # 再绑一次 query_vec
        params += [query_vec, top_k]

        with self.conn() as conn:
            cur = conn.execute(sql, params)
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
        return [dict(zip(cols, r)) for r in rows]

    # ------------------------------------------------------------------
    # 混合检索：HNSW（稠密余弦）+ tsvector（稀疏 BM25）→ RRF 融合
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query_seg: str,
        query_vec: list[float],
        top_k: int = 5,
        k: int = 60,
        start_date: str | None = None,
        end_date: str | None = None,
        ts_code: str | None = None,
        doc_type: str | None = None,
    ) -> dict:
        """双路召回 + Reciprocal Rank Fusion（RRF）融合。

        稠密路：pgvector HNSW 余弦（v_docs，chunk 级 → doc_id 聚合）
        稀疏路：tsvector 全文（documents.content_tsv，BM25 风格 ts_rank）
        融合：score(doc) = Σ_i 1/(k + rank_i(doc))，k=60（RRF 标准参数）

        Args:
            query_seg: jieba 空格分词后的查询（BM25 稀疏路用）
            query_vec: 查询向量（512 维，bge-small-zh-v1.5）
            k: RRF 常数 + 单路召回深度（默认 60，见 README §6）
        Returns:
            {"evidence": [...], "dense": [...], "sparse": [...], "k": k}
            每项证据含 doc_id/doc_type/ts_code/name/title/ann_date/chunk_text
            与 dense_rank/sparse_rank/rrf_score，全程可追溯。
        """
        # 1) 稠密路：HNSW 余弦取 top-k（chunk 级）
        dense = self.search(query_vec, top_k=k, start_date=start_date,
                            end_date=end_date, ts_code=ts_code, doc_type=doc_type)

        # 2) 稀疏路：tsvector 匹配 + ts_rank 取 top-k（文档级）
        sql = """
            SELECT d.id AS doc_id, d.doc_type, d.ts_code, d.name, d.title, d.ann_date,
                   left(d.content, 200) AS chunk_text,
                   ts_rank(d.content_tsv, q) AS score
            FROM documents d, websearch_to_tsquery('simple', %s) q
            WHERE d.content_tsv @@ q
        """
        conds: list[str] = []
        params: list = [query_seg]
        if start_date:
            conds.append("d.ann_date >= %s"); params.append(start_date)
        if end_date:
            conds.append("d.ann_date <= %s"); params.append(end_date)
        if ts_code:
            conds.append("d.ts_code = %s"); params.append(ts_code)
        if doc_type:
            conds.append("d.doc_type = %s"); params.append(doc_type)
        if conds:
            sql += " AND " + " AND ".join(conds)
        sql += " ORDER BY score DESC LIMIT %s"
        params.append(k)
        sparse_rows: list[dict] = []
        with self.conn() as conn:
            cur = conn.execute(sql, params)
            cols = [d.name for d in cur.description]
            for r in cur.fetchall():
                row = dict(zip(cols, r))
                row["sparse_score"] = float(row.pop("score"))
                sparse_rows.append(row)

        # 3) RRF 融合（按 doc_id 排名求逆）
        ranked = reciprocal_rank_fusion([dense, sparse_rows], k=k)

        # 4) 组装最终证据（保留两路排名与融合分）
        dense_rank = {r["doc_id"]: i + 1 for i, r in enumerate(dense)}
        sparse_rank = {r["doc_id"]: i + 1 for i, r in enumerate(sparse_rows)}
        seen: dict[int, dict] = {}
        for doc_id, _score in ranked:
            for src in (dense, sparse_rows):
                hit = next((r for r in src if r["doc_id"] == doc_id), None)
                if hit:
                    seen[doc_id] = {
                        **{kk: hit[kk] for kk in
                           ("doc_id", "doc_type", "ts_code", "name", "title", "ann_date", "chunk_text")},
                        "dense_rank": dense_rank.get(doc_id),
                        "sparse_rank": sparse_rank.get(doc_id),
                        "rrf_score": round(_score, 6),
                    }
                    break
            if len(seen) >= top_k:
                break
        evidence = [seen[i] for i in list(seen.keys())][:top_k]
        return {"evidence": evidence, "dense": dense, "sparse": sparse_rows, "k": k}

    # ------------------------------------------------------------------
    # 因子面板落库
    # ------------------------------------------------------------------

    def upsert_factor_panel(self, panel_rows: list[tuple]) -> int:
        """写入 (date, code, ...) 因子面板，冲突更新。"""
        with self.conn() as conn:
            with conn.cursor() as cur:
                for r in panel_rows:
                    cur.execute(
                        """
                        INSERT INTO factor_panels
                            (date, code, sentiment_mean, sentiment_std, event_mean, event_max,
                             n_announcements, tone_vol, ret_next)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (date, code) DO UPDATE SET
                            sentiment_mean = EXCLUDED.sentiment_mean,
                            event_mean = EXCLUDED.event_mean,
                            ret_next = EXCLUDED.ret_next
                        """,
                        r,
                    )
                conn.commit()
        return len(panel_rows)


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]], k: int = 60, id_key: str = "doc_id"
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion：score(d) = Σ_i 1/(k + rank_i(d)).

    - 只在单路出现的文档按该路排名计分（另一路无贡献）
    - 输入列表顺序即排名（第 1 个元素 rank=1）
    - 输出按融合分降序的 (doc_id, score) 列表
    参考：Cormack et al., "Reciprocal Rank Fusion Outperforms Condorcet
    and Individual Rank Learning Methods" (SIGIR 2009)。
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            doc_id = item[id_key]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def health_check() -> dict:
    """连通性自检（交付前验证用）。"""
    try:
        db = QuantDB()
        n_docs = db.count_documents()
        n_vec = db.embedding_count()
        return {"ok": True, "documents": n_docs, "embeddings": n_vec}
    except Exception as e:  # 连接失败时给出明确诊断
        return {"ok": False, "error": str(e), "hint": "请先运行 docker compose -f docker-compose.db.yml up -d"}
