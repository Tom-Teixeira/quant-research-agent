"""阶段2 工具：文本因子构建（对标实习"基于文本情绪、事件强度等
构建1000+另类数据因子"）。

词典法提供确定性基线，LLM 增强留出 DeepSeek 接口——两者可以
按比例叠加，这是 LangChain 工具层支持"确定性 + 生成式"混合的关键。
"""
from __future__ import annotations

import re
from typing import Iterable

import pandas as pd
from langchain_core.tools import tool
from langsmith import traceable

# ---- 情绪词典（中文金融场景，可扩展） ----
POSITIVE_WORDS = {
    "增长", "盈利", "超预期", "增持", "回购", "中标", "签约", "突破", "新高",
    "利好", "上调", "推荐", "改善", "扭亏", "分红", "获批", "投产", "提价", "扩张",
}
NEGATIVE_WORDS = {
    "下滑", "亏损", "低于预期", "减持", "质押", "诉讼", "处罚", "违规", "退市",
    "利空", "下调", "回避", "恶化", "st", "停产", "召回", "仲裁", "冻结", "被查",
}

# ---- 事件强度词典（不同事件对股价影响权重，对标"事件强度"因子） ----
EVENT_WEIGHTS = {
    "收购": 3.0, "重组": 3.0, "增持": 2.0, "回购": 2.0, "中标": 2.0, "分红": 1.5,
    "业绩预增": 2.5, "业绩预减": -2.5, "减持": -2.0, "质押": -1.5, "诉讼": -1.5,
    "处罚": -2.0, "违规": -2.5, "退市风险": -3.0, "停产": -2.0, "被查": -2.0,
}

_WORD_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")


@tool
@traceable
def sentiment_score(text: str) -> float:
    """词典法情感打分：返回 [-1, 1] 区间的情感分数。
    正面词 +1、负面词 -1，归一化到文本长度。

    Args:
        text: 清洗后的公告/新闻文本。
    """
    tokens = _WORD_RE.findall(text or "")
    if not tokens:
        return 0.0
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 4)


@tool
@traceable
def event_intensity(text: str) -> float:
    """事件强度打分：按事件类型加权求和（可正可负）。

    Args:
        text: 清洗后的文本。
    """
    score = 0.0
    for keyword, weight in EVENT_WEIGHTS.items():
        if keyword in (text or ""):
            score += weight
    return round(score, 4)


@tool
@traceable
def build_factor_panel(df: pd.DataFrame, min_occurrence: int = 1) -> pd.DataFrame:
    """将公告级文本聚合为 (date, code) 面板因子。

    因子（每只股票每个交易日一行）:
      - n_announcements : 当日公告数
      - sentiment_mean  : 当日公告平均情感
      - sentiment_std   : 情感波动（信息分歧度）
      - event_mean      : 事件强度均值
      - event_max       : 当日最强事件强度
      - tone_vol        : 5日情感滚动标准差（动量/反转线索）

    Args:
        df: 含 date/code/text 列的清洗后数据。
    """
    rows = []
    for _, r in df.iterrows():
        s = sentiment_score.func(r["text"])
        e = event_intensity.func(r["text"])
        rows.append({"date": r["date"], "code": r["code"], "sentiment": s, "event": e, "n": 1})

    panel = pd.DataFrame(rows)
    daily = (
        panel.groupby(["date", "code"])
        .agg(n_announcements=("n", "sum"), sentiment_mean=("sentiment", "mean"),
             sentiment_std=("sentiment", "std"), event_mean=("event", "mean"),
             event_max=("event", "max"))
        .reset_index()
    )
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)
    daily = daily.sort_values(["code", "date"]).reset_index(drop=True)
    # 5 日滚动波动（tone_vol）
    daily["tone_vol"] = daily.groupby("code")["sentiment_mean"].transform(
        lambda x: x.rolling(5, min_periods=2).std().fillna(0.0)
    )
    daily = daily[daily["n_announcements"] >= min_occurrence]
    return daily.reset_index(drop=True)
