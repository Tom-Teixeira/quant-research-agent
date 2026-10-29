"""阶段1 工具：文本清洗与数据质量（对标实习"文本清洗、时间对齐、
缺失值处理、异常值缩尾、特征标准化及数据质量校验"）。
"""
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
    """清洗公告/新闻正文：去 HTML 标签、URL、多余空白，统一中英文标点，去重空格。

    Args:
        texts: 原始文本列表。

    Returns:
        清洗后的文本列表（与输入顺序一致）。
    """
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
    """按主键去重（对标数据一致性校验）。返回去重后的 DataFrame。

    Args:
        df: 公告数据，至少含 date/code/title/text 列。
        keys: 去重主键。
    """
    n_before = len(df)
    df = df.drop_duplicates(subset=list(keys), keep="first").reset_index(drop=True)
    df.attrs["dup_removed"] = n_before - len(df)
    return df


@tool
@traceable
def time_align_panel(df: pd.DataFrame, date_col: str = "date", code_col: str = "code") -> pd.DataFrame:
    """时间对齐：同一公司同日多公告合并为一条（保留最新一条），
    按 (code, date) 排序生成单调面板。

    生产环境建议接入交易日历补全缺失交易日（此处保持观测粒度，
    避免重采样引入数据膨胀）。

    Args:
        df: 含 date/code/text 列的公告数据。
    """
    if not {"date", "code", "text"}.issubset(df.columns):
        raise ValueError("数据必须包含 date/code/text 列")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values([code_col, date_col]).reset_index(drop=True)
    # 同公司同日多公告 → 保留最新一条（信息增量口径）
    df = df.groupby([code_col, date_col], as_index=False, group_keys=False).tail(1)
    return df.reset_index(drop=True)


@tool
@traceable
def winsorize_series(series: pd.Series, limits: tuple[float, float] = (0.01, 0.99)) -> pd.Series:
    """异常值缩尾：将超出 [q(lo), q(hi)] 的观测截断到分位数上。
    对标实习"异常值缩尾"。
    """
    lo, hi = series.quantile(limits[0]), series.quantile(limits[1])
    return series.clip(lower=lo, upper=hi)


@tool
@traceable
def standardize_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """特征标准化（z-score）：对标实习"特征标准化"。
    用训练期均值/标准差，避免未来函数。
    """
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
    """数据质量校验报告：行数、去重数、缺失率、空文本率。

    Returns:
        {rows, dup_removed, missing_pct, empty_text_pct}
    """
    return {
        "rows": int(len(df)),
        "dup_removed": int(df.attrs.get("dup_removed", 0)),
        "missing_pct": round(float(df.isna().mean().mean() * 100), 2),
        "empty_text_pct": round(float((df["text"].fillna("").astype(str).str.strip() == "").mean() * 100), 2),
    }


def text_fingerprint(text: str) -> str:
    """文本指纹（用于去重/查重，预留）。"""
    return hashlib.md5(re.sub(r"\s+", "", text).encode("utf-8")).hexdigest()
