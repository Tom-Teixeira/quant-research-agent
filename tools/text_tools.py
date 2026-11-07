from __future__ import annotations

import hashlib
import re
from typing import Iterable

import pandas as pd
from langchain_core.tools import tool
from langsmith import traceable

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_WHITESPACE = re.compile(r"\s+")
_CJK_PUNCT = {"，": ",", "。": ".", "；": ";", "：": ":", "（": "(", "）": ")", "！": "!", "？": "?"}


@tool
@traceable
def clean_text_series(texts: Iterable[str]) -> list[str]:
    cleaned = []
    for t in texts:
        if t is None or (isinstance(t, float) and pd.isna(t)):
            cleaned.append("")
            continue
        s = str(t)
        s = _HTML_TAG.sub(" ", s)
        s = _URL.sub(" ", s)
        for zh, en in _CJK_PUNCT.items():
            s = s.replace(zh, en)
        s = _WHITESPACE.sub(" ", s).strip()
        cleaned.append(s)
    return cleaned


@tool
@traceable
def deduplicate_records(df: pd.DataFrame, keys: tuple[str, ...] = ("date", "code", "title")) -> pd.DataFrame:
    n_before = len(df)
    df = df.drop_duplicates(subset=list(keys), keep="first").reset_index(drop=True)
    df.attrs["dup_removed"] = n_before - len(df)
    return df


@tool
@traceable
def time_align_panel(df: pd.DataFrame, date_col: str = "date", code_col: str = "code") -> pd.DataFrame:
    if not {"date", "code", "text"}.issubset(df.columns):
        raise ValueError("need date/code/text columns")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values([code_col, date_col]).reset_index(drop=True)
    df = df.groupby([code_col, date_col], as_index=False, group_keys=False).tail(1)
    return df.reset_index(drop=True)


@tool
@traceable
def winsorize_series(series: pd.Series, limits: tuple[float, float] = (0.01, 0.99)) -> pd.Series:
    lo, hi = series.quantile(limits[0]), series.quantile(limits[1])
    return series.clip(lower=lo, upper=hi)


@tool
@traceable
def standardize_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        mean, std = out[c].mean(), out[c].std()
        if std == 0 or pd.isna(std):
            out[c] = 0.0
        else:
            out[c] = (out[c] - mean) / std
    return out


@tool
@traceable
def quality_report(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "dup_removed": int(df.attrs.get("dup_removed", 0)),
        "missing_pct": round(float(df.isna().mean().mean() * 100), 2),
        "empty_text_pct": round(float((df["text"].fillna("").astype(str).str.strip() == "").mean() * 100), 2),
    }


def text_fingerprint(text: str) -> str:
    return hashlib.md5(re.sub(r"\s+", "", text).encode("utf-8")).hexdigest()
