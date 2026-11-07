from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import get_settings
from db.db_client import QuantDB, health_check
from ingest.pipeline import run_ingest_pipeline
from rag.rag_node import rag_generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["check", "ingest", "rag", "research"], default="check")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--index", default="000300.SH")
    parser.add_argument("--query", default="回购增持进展")
    parser.add_argument("--ts-code", default="")
    parser.add_argument("--mode", choices=["adaptive", "hybrid", "vector"], default="adaptive")
    args = parser.parse_args()

    settings = get_settings()
    settings.apply_langsmith_env()

    if args.stage == "check":
        print(json.dumps(health_check()))

    elif args.stage == "ingest":
        stats = run_ingest_pipeline(args.start, args.end, index_code=args.index)
        print(json.dumps(stats))

    elif args.stage == "rag":
        result = rag_generate(
            args.query, settings, top_k=5,
            start_date=args.start, end_date=args.end,
            ts_code=args.ts_code or None, mode=args.mode,
        )
        print("route:", result.get("route"))
        for h in result["evidence"]:
            print(f"  [{h['ann_date']}] {h['name']} {h['title']}")
        print(result["report"])

    elif args.stage == "research":
        stats = run_ingest_pipeline(args.start, args.end, index_code=args.index)
        print("ingest:", json.dumps(stats))

        from graph import build_agent
        agent = build_agent(settings)
        result = agent.invoke({"data_path": _panel_csv(settings.output_dir)})
        print("signal:", json.dumps({k: v for k, v in result["quality"].items() if k != "passed"}))
        print("backtest:", json.dumps(result["backtest"]))


def _panel_csv(output_dir: str) -> str:
    path = Path(output_dir) / "factor_panel.csv"
    db = QuantDB()
    with db.conn() as conn:
        rows = conn.execute(
            "SELECT date, code, sentiment_mean, sentiment_std, event_mean, event_max, "
            "n_announcements, tone_vol, ret_next FROM factor_panels"
        ).fetchall()
    if not rows:
        return str(Path(__file__).resolve().parent / "data" / "sample_announcements.csv")
    import pandas as pd
    pd.DataFrame([dict(r) for r in rows]).to_csv(path, index=False)
    return str(path)


if __name__ == "__main__":
    main()
