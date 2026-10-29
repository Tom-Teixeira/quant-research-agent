"""阶段3 工具：机器学习信号生成与质量评估（对标实习"机器学习模型
开展横截面收益预测与股票信号筛选"，口径：日频 RankIC=3.0%、ICIR=0.82、
AUC 较基线提升 5%）。

实现要点：
  - 严格时间切分（训练在前、测试在后），防未来函数；
  - RankIC = 预测值与实际收益的日度 Spearman 相关均值；
  - ICIR = RankIC 均值 / 标准差（信息比率口径）；
  - AUC 用 top 组多空命中率近似（日频截面无二分类标签时的常用做法）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from langsmith import traceable
from scipy.stats import spearmanr


@tool
@traceable
def train_cross_sectional_model(
    factor_panel: pd.DataFrame,
    label_panel: pd.DataFrame,
    feature_cols: list[str] | None = None,
    split_ratio: float = 0.7,
    seed: int = 42,
) -> dict:
    """训练截面收益预测模型（逻辑回归，工业可替换为 XGBoost/LightGBM）。

    Args:
        factor_panel: (date, code) + 因子列的面板。
        label_panel:  (date, code) + ret_next 列（次日收益，标签）。
        feature_cols: 因子列名，默认取除 date/code 外全部数值列。
        split_ratio:  时间切分比例（前 70% 训练）。

    Returns:
        {model, feature_cols, train_pred, test_pred, test_df}
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # 同源（factor_panel 已含 ret_next）时直接复用，避免 merge 产生 ret_next_x/y
    if "ret_next" in factor_panel.columns:
        data = factor_panel.copy()
    else:
        if "ret_next" not in label_panel.columns:
            raise ValueError("label_panel 必须包含 ret_next 列（次日收益标签）")
        data = factor_panel.merge(
            label_panel[["date", "code", "ret_next"]], on=["date", "code"], how="inner"
        )
    if feature_cols is None:
        feature_cols = [c for c in data.columns if c not in {"date", "code", "ret_next"}]
    data = data.dropna(subset=feature_cols + ["ret_next"]).sort_values("date").reset_index(drop=True)

    dates = pd.to_datetime(data["date"]).sort_values().unique()
    cut = dates[int(len(dates) * split_ratio)]
    data["date"] = pd.to_datetime(data["date"])  # 统一为 Timestamp，避免 str 比较报错
    train_mask, test_mask = data["date"] <= cut, data["date"] > cut

    X_tr, y_tr = data.loc[train_mask, feature_cols], (data.loc[train_mask, "ret_next"] > 0).astype(int)
    X_te = data.loc[test_mask, feature_cols]

    scaler = StandardScaler().fit(X_tr)
    model = LogisticRegression(max_iter=500, random_state=seed).fit(scaler.transform(X_tr), y_tr)

    out = data.copy()
    out["pred_prob"] = np.nan
    out.loc[test_mask, "pred_prob"] = model.predict_proba(scaler.transform(X_te))[:, 1]
    # 训练期也给出预测（供诊断；评估只用测试期）
    out.loc[train_mask, "pred_prob"] = model.predict_proba(scaler.transform(X_tr))[:, 1]

    return {
        "model": model,
        "feature_cols": feature_cols,
        "test_df": out.loc[test_mask].reset_index(drop=True),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }


@tool
@traceable
def evaluate_rank_ic(pred: pd.Series, actual: pd.Series, group_key: pd.Series) -> dict:
    """评估预测质量：日度 RankIC / ICIR。

    Args:
        pred:    预测概率（或预测分数）。
        actual:  实际收益。
        group_key: 日期列（按日计算 Spearman 再平均，即"日频 RankIC"）。

    Returns:
        {rank_ic, icir, daily_ic_count}
    """
    frame = pd.DataFrame({"pred": pred, "actual": actual, "date": group_key}).dropna()
    daily_ics = {}
    for d, g in frame.groupby("date"):
        if len(g) >= 5:  # 截面样本不足时跳过该日
            ic, _ = spearmanr(g["pred"], g["actual"])
            if not np.isnan(ic):
                daily_ics[d] = ic
    ics = np.array(list(daily_ics.values()))
    if len(ics) == 0:
        return {"rank_ic": 0.0, "icir": 0.0, "daily_ic_count": 0}
    mean_ic, std_ic = float(ics.mean()), float(ics.std())
    return {"rank_ic": round(mean_ic, 4), "icir": round(mean_ic / std_ic, 4) if std_ic > 0 else 0.0,
            "daily_ic_count": int(len(ics))}


@tool
@traceable
def evaluate_auc(test_df: pd.DataFrame) -> float:
    """AUC 近似评估：按预测分 5 组，top 组做多/底组做空的胜率
    （日频截面命中率，与实习 AUC 口径一致）。
    """
    df = test_df.copy()
    df["group"] = pd.qcut(df["pred_prob"], 5, labels=False, duplicates="drop")
    top, bottom = df[df["group"] == df["group"].max()], df[df["group"] == df["group"].min()]
    if len(top) == 0 or len(bottom) == 0:
        return 0.5
    long_win = (top["ret_next"] > 0).mean()
    short_win = (bottom["ret_next"] < 0).mean()
    return round(float((long_win + short_win) / 2), 4)
