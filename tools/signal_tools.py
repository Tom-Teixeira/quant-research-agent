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
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if "ret_next" in factor_panel.columns:
        data = factor_panel.copy()
    else:
        if "ret_next" not in label_panel.columns:
            raise ValueError("label_panel needs ret_next")
        data = factor_panel.merge(
            label_panel[["date", "code", "ret_next"]], on=["date", "code"], how="inner"
        )
    if feature_cols is None:
        feature_cols = [c for c in data.columns if c not in {"date", "code", "ret_next"}]
    data = data.dropna(subset=feature_cols + ["ret_next"]).sort_values("date").reset_index(drop=True)

    dates = pd.to_datetime(data["date"]).sort_values().unique()
    cut = dates[int(len(dates) * split_ratio)]
    data["date"] = pd.to_datetime(data["date"])
    train_mask, test_mask = data["date"] <= cut, data["date"] > cut

    X_tr, y_tr = data.loc[train_mask, feature_cols], (data.loc[train_mask, "ret_next"] > 0).astype(int)
    X_te = data.loc[test_mask, feature_cols]

    scaler = StandardScaler().fit(X_tr)
    model = LogisticRegression(max_iter=500, random_state=seed).fit(scaler.transform(X_tr), y_tr)

    out = data.copy()
    out["pred_prob"] = np.nan
    out.loc[test_mask, "pred_prob"] = model.predict_proba(scaler.transform(X_te))[:, 1]
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
    frame = pd.DataFrame({"pred": pred, "actual": actual, "date": group_key}).dropna()
    daily_ics = {}
    for d, g in frame.groupby("date"):
        if len(g) >= 5:
            ic, _ = spearmanr(g["pred"], g["actual"])
            if not np.isnan(ic):
                daily_ics[d] = ic
    ics = np.array(list(daily_ics.values()))
    if len(ics) == 0:
        return {"rank_ic": 0.0, "icir": 0.0, "daily_ic_count": 0}
    mean_ic, std_ic = float(ics.mean()), float(ics.std())
    return {
        "rank_ic": round(mean_ic, 4),
        "icir": round(mean_ic / std_ic, 4) if std_ic > 0 else 0.0,
        "daily_ic_count": int(len(ics)),
    }


@tool
@traceable
def evaluate_auc(test_df: pd.DataFrame) -> float:
    df = test_df.copy()
    df["group"] = pd.qcut(df["pred_prob"], 5, labels=False, duplicates="drop")
    top, bottom = df[df["group"] == df["group"].max()], df[df["group"] == df["group"].min()]
    if len(top) == 0 or len(bottom) == 0:
        return 0.5
    long_win = (top["ret_next"] > 0).mean()
    short_win = (bottom["ret_next"] < 0).mean()
    return round(float((long_win + short_win) / 2), 4)
