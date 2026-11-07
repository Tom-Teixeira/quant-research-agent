from __future__ import annotations

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from langsmith import traceable

COMMISSION = 0.0003
SLIPPAGE = 0.0001
STAMP_TAX = 0.0005
ROUND_TRIP_COST = 2 * (COMMISSION + SLIPPAGE) + STAMP_TAX


@tool
@traceable
def run_backtest(signal_df: pd.DataFrame, n_groups: int = 5, rebalance_cost: float = ROUND_TRIP_COST) -> dict:
    df = signal_df.sort_values("date").reset_index(drop=True)
    df["group"] = df.groupby("date")["pred_prob"].transform(
        lambda x: pd.qcut(x.rank(method="first"), n_groups, labels=False, duplicates="drop")
    )
    daily = df.groupby("date").apply(_long_short_return, include_groups=False)
    daily = daily.reset_index()
    daily.columns = ["date", "ret"]
    daily["ret"] = daily["ret"].fillna(0.0)
    metrics = performance_metrics.func(daily["ret"], rebalance_cost)
    return {"daily_returns": daily, "metrics": metrics}


def _long_short_return(g: pd.DataFrame) -> pd.Series:
    top, bottom = g[g["group"] == g["group"].max()], g[g["group"] == g["group"].min()]
    if len(top) == 0 or len(bottom) == 0:
        return pd.Series({"ret": 0.0})
    return pd.Series({"ret": (top["ret_next"].mean() - bottom["ret_next"].mean()) / 2.0})


@tool
@traceable
def performance_metrics(daily_returns: pd.Series, rebalance_cost: float = ROUND_TRIP_COST) -> dict:
    rets = daily_returns.astype(float).values
    turnover_cost = rebalance_cost * (252 / 21)
    net = rets - turnover_cost / 252
    ann = float((1 + net).prod() ** (252 / len(net)) - 1) if len(net) else 0.0
    sharpe = float(net.mean() / net.std() * np.sqrt(252)) if net.std() > 0 else 0.0
    cum = np.cumprod(1 + net)
    drawdown = float((cum / np.maximum.accumulate(cum) - 1).min())
    return {
        "annual_return": round(ann, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(drawdown, 4),
        "turnover_cost_pa": round(turnover_cost, 4),
        "n_days": int(len(net)),
    }


@tool
@traceable
def neutral_excess(signal_df: pd.DataFrame, benchmark_ret: pd.Series) -> dict:
    daily = run_backtest.func(signal_df)["daily_returns"]
    merged = daily.merge(benchmark_ret.reset_index(), on="date", how="left", suffixes=("", "_bm"))
    bench = merged.get("ret_bm")
    if bench is None or bench.isna().all():
        bench = signal_df.groupby("date")["ret_next"].mean().reindex(daily["date"]).fillna(0.0)
    excess = daily["ret"] - bench.values
    ann = float((1 + excess).prod() ** (252 / len(excess)) - 1) if len(excess) else 0.0
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0
    return {"excess_ann": round(ann, 4), "excess_sharpe": round(sharpe, 4)}
