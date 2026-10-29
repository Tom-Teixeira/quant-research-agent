"""验证脚本：V2 全图（mock）+ RAG 生成链路。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings
from rag.rag_node import rag_generate

settings = get_settings()
settings.apply_langsmith_env()

# 1) RAG 生成链路（mock LLM：验证 evidence → prompt → 结构化输出）
print("==== RAG 生成（mock） ====")
res = rag_generate("回购增持进展", settings, top_k=3,
                   start_date="2024-01-01", end_date="2024-01-31")
print("证据数:", len(res["evidence"]))
print("报告:", str(res["report"])[:300])

# 2) V2 全图（mock 模式，短窗）
print("\n==== V2 全图（mock） ====")
from graph_v2 import build_v2_agent

agent = build_v2_agent(settings)
out = agent.invoke({
    "data_path": None,
    "start_date": "2024-01-01",
    "end_date": "2024-01-10",
    "index_code": "000300.SH",
    "query": "业绩预增",
})
print("db_stats:", {k: v for k, v in out.get("db_stats", {}).items()})
print("quality:", {k: v for k, v in out.get("quality", {}).items() if k != "passed"})
print("backtest keys:", list(out.get("backtest", {}).keys()))
print("evidence:", len(out.get("evidence", [])))
print("report 前 150 字:", str(out.get("report", ""))[:150])
print("V2 全图 mock 验证 OK")
