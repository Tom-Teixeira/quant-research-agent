"""阶段4 工具：组合回测与绩效归因（对标实习"回测，已计入双边交易
佣金、滑点及卖出印花税；年化12.6%、夏普1.35、最大回撤-8.7%、
中证1000中性组合超额8.7%"）。

口径说明：
  - 单边佣金 0.03%（万3）、滑点 0.01%（万1）、卖出印花税 0.05%（2023.8 后减半为 0.05%），
    双边换手成本 ≈ 0.08%/次；
  - 组合按预测分数分 5 组，做多 top 组、做空 bottom 组（多空），
    另输出相对基准的中性超额。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from langsmith import traceable

COMMISSION = 0.0003   # 佣金 万3
SLIPPAGE = 0.0001     # 滑点 万1
STAMP_TAX = 0.0005    # 卖出印花税 万5
ROUND_TRIP_COST = 2 * (COMMISSION + SLIPPAGE) + STAMP_TAX  # ≈ 0.13% 一次换手


@tool
@traceable
def run_backtest(signal_df: pd.DataFrame, n_groups: int = 5, rebalance_cost: float = ROUND_TRIP_COST) -> dict:
    """执行分组回测，输出组合日收益序列与绩效指标。

    Args:
        signal_df: 含 date/code/pred_prob/ret_next 的测试期数据。
        n_groups:  分组数（5 组即 Q1-Q5 多空）。
        rebalance_cost: 单次换手成本比例。

    Returns:
        {daily_returns, metrics}
        metrics: {annual_return, sharpe, max_drawdown, turnover}
    """
    df = signal_df.sort_values("date").reset_index(drop=True)
    df["group"] = df.groupby("date")["pred_prob"].transform(
        lambda x: pd.qcut(x.rank(method="first"), n_groups, labels=False, duplicates="drop")
    )
    # 每日组合：等权做多 top 组、等权做空 bottom 组
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
    long_ret = top["ret_next"].mean()
    short_ret = bottom["ret_next"].mean()
    return pd.Series({"ret": (long_ret - short_ret) / 2.0})


@tool
@traceable
def performance_metrics(daily_returns: pd.Series, rebalance_cost: float = ROUND_TRIP_COST) -> dict:
    """绩效指标：年化收益、夏普、最大回撤、年化换手成本。

    Args:
        daily_returns: 组合日收益序列（未计成本）。
        rebalance_cost: 每次调仓成本。
    """
    rets = daily_returns.astype(float).values
    # 简化的换手假设：每月调仓一次（约 21 个交易日）
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
    """相对基准的中性组合超额（对标"中证1000中性组合年化超额8.7%"）。

    Args:
        signal_df: 含 date/pred_prob/ret_next 的测试期数据。
        benchmark_ret: 基准日收益序列（未提供时用全样本等权代理）。

    Returns:
        {excess_ann, excess_sharpe}
    """
    daily = run_backtest.func(signal_df)["daily_returns"]
    merged = daily.merge(benchmark_ret.reset_index(), on="date", how="left", suffixes=("", "_bm"))
    bench = merged.get("ret_bm")
    if bench is None or bench.isna().all():
        # 用全体股票等权日收益作为基准代理
        bench = signal_df.groupby("date")["ret_next"].mean().reindex(daily["date"]).fillna(0.0)
    excess = daily["ret"] - bench.values
    ann = float((1 + excess).prod() ** (252 / len(excess)) - 1) if len(excess) else 0.0
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0
    return {"excess_ann": round(ann, 4), "excess_sharpe": round(sharpe, 4)}
