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
            docs: [{doc_type, ts_code, name, title, content, ann_date, source_url, extra}]
        """
        inserted = 0
        with self.conn() as conn:
            with conn.cursor() as cur:
                for d in docs:
                    cur.execute(
                        """
                        INSERT INTO documents (doc_type, ts_code, name, title, content, ann_date, source_url, extra)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT DO NOTHING
                        """,
                        (
                            d["doc_type"], d.get("ts_code"), d.get("name"), d["title"],
                            d.get("content"), d["ann_date"], d.get("source_url"),
                            psycopg.types.json.Jsonb(d.get("extra") or {}),
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


def health_check() -> dict:
    """连通性自检（交付前验证用）。"""
    try:
        db = QuantDB()
        n_docs = db.count_documents()
        n_vec = db.embedding_count()
        return {"ok": True, "documents": n_docs, "embeddings": n_vec}
    except Exception as e:  # 连接失败时给出明确诊断
        return {"ok": False, "error": str(e), "hint": "请先运行 docker compose -f docker-compose.db.yml up -d"}
