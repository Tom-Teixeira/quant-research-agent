from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from config import get_settings
from graph import build_agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "live"])
    parser.add_argument("--data", type=str)
    args = parser.parse_args()

    settings = get_settings()
    if args.mode:
        settings.mode = args.mode
    if args.data:
        settings.data_path = args.data
    settings.apply_langsmith_env()

    print(f"mode={settings.mode} model={settings.deepseek_model}")

    agent = build_agent(settings)
    result = agent.invoke({"data_path": settings.data_path})

    print("quality:", json.dumps(result["data_quality_report"]))
    print("factors:", json.dumps(result["factor_meta"]))
    print("signal:", json.dumps({k: v for k, v in result["quality"].items() if k != "passed"}))
    print("backtest:", json.dumps(result["backtest"]))

    out = Path(settings.output_dir) / f"report_{datetime.now():%Y%m%d_%H%M%S}.md"
    out.write_text(result["report"], encoding="utf-8")
    print("saved:", out)


if __name__ == "__main__":
    main()
