# Quant Research Agent - 既有库数据回填（混合检索 tsvector 索引）

# 用途：对已初始化的 quant-pgvector 容器（schema.sql 早期版本，无 content_seg/
#       content_tsv）执行 002 迁移并回填分词文本，使 BM25 稀疏检索生效。
# 等价于：psql -f db/migrations/002_hybrid_search.sql + 本脚本回填。
# 执行：python scripts/backfill_tsv.py  [--limit 10000]
# 幂等：可重复执行；content_seg 非空的行跳过。

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.db_client import QuantDB
from rag.segment import segment_chinese


def backfill(limit: int | None = None) -> dict:
    db = QuantDB()
    with db.conn() as conn:
        # 1. 先执行 002 迁移（ALTER + GIN 索引 + 触发器，幂等）
        sql_path = Path(__file__).resolve().parents[1] / "db" / "migrations" / "002_hybrid_search.sql"
        with conn.cursor() as cur:
            cur.execute(sql_path.read_text(encoding="utf-8"))
        conn.commit()

        # 2. 回填 content_seg（jieba 分词）
        where = "WHERE content_seg IS NULL OR content_seg = ''"
        limit_sql = f" ORDER BY id LIMIT {int(limit)}" if limit else ""
        with conn.cursor() as cur:
            cur.execute(f"SELECT id, title, content FROM documents {where}{limit_sql}")
            rows = cur.fetchall()
        updated = 0
        with conn.cursor() as cur:
            for doc_id, title, content in rows:
                seg = segment_chinese(f"{title} {content or ''}")
                if not seg:
                    continue
                cur.execute("UPDATE documents SET content_seg = %s WHERE id = %s", (seg, doc_id))
                updated += 1
        conn.commit()
        # 3. 触发器中未覆盖的存量行：触发器只在 INSERT/UPDATE OF title/content_seg 触发，
        #    UPDATE content_seg 已触发 → tsvector 已重算；此处仅作校验统计
        cur = conn.execute(
            "SELECT count(*) FILTER (WHERE content_tsv IS NOT NULL) AS tsv_ok, count(*) AS total FROM documents"
        ).fetchone()
    return {"scanned": len(rows), "backfilled": updated, "tsv_indexed": cur[0], "total_docs": cur[1]}


def main() -> None:
    parser = argparse.ArgumentParser(description="回填 content_seg 供 BM25 稀疏检索")
    parser.add_argument("--limit", type=int, default=None, help="仅回填前 N 行（分批）")
    args = parser.parse_args()
    import json

    print(json.dumps(backfill(args.limit), ensure_ascii=False))


if __name__ == "__main__":
    main()
