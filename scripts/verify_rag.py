"""验证脚本：RAG 语义检索 + 数据库状态。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.db_client import health_check
from rag.rag_node import retrieve_docs

print("DB:", json.dumps(health_check(), ensure_ascii=False))
for query in ["回购增持进展", "重大资产重组", "业绩预增"]:
    hits = retrieve_docs(query, top_k=3, start_date="2024-01-01", end_date="2024-01-31")
    print(f"\n== 查询「{query}」 top-3 ==")
    for h in hits:
        print(f"  [{h['ann_date']}] {h['name']} | {h['title']} | sim={round(1 - h['distance'], 4)}")
        print(f"      {h['chunk_text'][:80]}")
