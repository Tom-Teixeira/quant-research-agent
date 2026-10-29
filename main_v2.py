"""二期入口：真实数据 → pgvector 入库 → RAG 检索增强 → 因子/回测 → 报告。

用法：
    python main_v2.py --stage ingest    # 拉取数据入库（Choice 或文件）
    python main_v2.py --stage rag       # RAG 检索问答（DeepSeek 生成）
    python main_v2.py --stage research  # 全流程：入库→因子→回测→报告
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from config import get_settings
from db.db_client import QuantDB, health_check
from ingest.data_providers import get_provider
from ingest.pipeline import run_ingest_pipeline
from rag.rag_node import rag_generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Quant Research Agent V2（数据库 + RAG）")
    parser.add_argument("--stage", choices=["check", "ingest", "rag", "research"], default="check")
    parser.add_argument("--start", default="2018-01-01", help="数据起始日期")
    parser.add_argument("--end", default="2025-12-31", help="数据结束日期")
    parser.add_argument("--index", default="000300.SH", help="指数代码")
    parser.add_argument("--query", default="回购增持进展", help="RAG 检索查询")
    parser.add_argument("--ts-code", default="", help="RAG 股票代码过滤")
    parser.add_argument("--mode", choices=["adaptive", "hybrid", "vector"], default="adaptive",
                        help="RAG 检索模式：adaptive=自适应路由(默认) / hybrid=固定混合检索 / vector=纯向量")
    args = parser.parse_args()

    settings = get_settings()
    settings.apply_langsmith_env()

    if args.stage == "check":
        print("数据库健康检查:", json.dumps(health_check(), ensure_ascii=False))

    elif args.stage == "ingest":
        print(f"[stage] 数据入库 {args.start} ~ {args.end}  指数={args.index}  数据源=auto(Choice优先)")
        t0 = time.time()
        stats = run_ingest_pipeline(args.start, args.end, index_code=args.index)
        print(f"[stage] 入库耗时 {time.time() - t0:.1f}s  结果: {json.dumps(stats, ensure_ascii=False)}")

    elif args.stage == "rag":
        print(f"[stage] RAG 检索增强（模式={args.mode}）  查询: {args.query}  代码过滤: {args.ts_code or '全部'}")
        t0 = time.time()
        result = rag_generate(args.query, settings, top_k=5,
                              start_date=args.start, end_date=args.end,
                              ts_code=args.ts_code or None,
                              mode=args.mode)
        print(f"[stage] 路由={result.get('route')} 检索到 {len(result['evidence'])} 条证据，生成耗时 {time.time() - t0:.1f}s")
        if result.get("queries"):
            print(f"[stage] 子查询: {result['queries']}")
        print("\n---- 证据 ----")
        for h in result["evidence"]:
            score = h.get("rrf_score")
            score_txt = f"RRF {score}" if score is not None else (
                f"相似度 {round(1 - h['distance'], 4)}" if "distance" in h else "混合"
            )
            print(f"  [{h['ann_date']}] {h['name']} | {h['title']} ({score_txt})")
        print("\n---- DeepSeek 分析 ----")
        print(result["report"])

    elif args.stage == "research":
        print(f"[stage] 全流程：入库 → 因子 → 回测 → RAG 报告")
        stats = run_ingest_pipeline(args.start, args.end, index_code=args.index)
        print("入库:", json.dumps(stats, ensure_ascii=False))
        # 因子与回测（复用一期图，从库中读取因子面板）
        from graph import build_agent
        agent = build_agent(settings)
        result = agent.invoke({"data_path": _export_panel_csv(settings.output_dir)})
        print("信号质量:", json.dumps({k: v for k, v in result["quality"].items() if k != "passed"}, ensure_ascii=False))
        print("回测:", json.dumps(result["backtest"], ensure_ascii=False))
        out = Path(settings.output_dir) / f"research_v2_{datetime.now():%Y%m%d_%H%M%S}.md"
        out.write_text(f"# 量化研究（V2 数据库+RAG）\n\n{result['report']}", encoding="utf-8")
        print("报告:", out)


def _export_panel_csv(output_dir: str) -> str:
    """把 pgvector 因子面板导出为 CSV（衔接一期回测图）。真实场景直接读库。"""
    import pandas as pd

    db = QuantDB()
    with db.conn() as conn:
        rows = conn.execute("SELECT date, code, sentiment_mean, sentiment_std, event_mean, event_max, n_announcements, tone_vol, ret_next FROM factor_panels").fetchall()
    if not rows:
        # 无因子面板时回退示例数据（保持图可运行）
        return str(Path(__file__).resolve().parent / "data" / "sample_announcements.csv")
    df = pd.DataFrame([dict(r) for r in rows])
    path = Path(output_dir) / "factor_panel_from_db.csv"
    df.to_csv(path, index=False)
    return str(path)


if __name__ == "__main__":
    main()
