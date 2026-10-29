"""数据接入层：统一 DataProvider 接口。

设计目标（对标实习"围绕公告、研究报告及新闻文本的结构化数据体系"）：
  - TushareProvider：真实数据后端（官方 SDK + token，覆盖公告 anns_d / 新闻 news /
    major_news / 研报 research_report / 沪深300 成分 index_weight / 行情 daily / 交易日历 trade_cal）
  - FileProvider：本地文件后端（开发/无 token 时验证全链路；显式标注为 fallback，
    不替代 Tushare 作为真实数据源）

注意：Tushare MCP（豆包插件）当前授权失效时，本模块走 SDK 直连；
token 由用户在 .env 配置，绝不写入代码或日志。
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class DataProvider(ABC):
    """量化文本数据提供者统一接口。"""

    @abstractmethod
    def get_index_constituents(self, index_code: str, trade_date: str) -> list[str]:
        """获取指数成分股代码列表（如沪深300：000300.SH）。"""

    @abstractmethod
    def get_announcements(self, codes: list[str], start_date: str, end_date: str) -> list[dict]:
        """拉取公告：[{doc_type, ts_code, name, title, ann_date, source_url, content}]"""

    @abstractmethod
    def get_news(self, start_date: str, end_date: str) -> list[dict]:
        """拉取财经新闻正文。"""

    @abstractmethod
    def get_research_reports(self, codes: list[str], start_date: str, end_date: str) -> list[dict]:
        """拉取券商研报（标题+摘要）。"""

    @abstractmethod
    def get_daily_returns(self, codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """拉取日频行情（用于回测标签 ret_next）。"""


class TushareProvider(DataProvider):
    """Tushare SDK 后端（需 .env: TUSHARE_TOKEN）。"""

    def __init__(self, token: str | None = None):
        import tushare as ts

        self.ts = ts
        self.pro = ts.pro_api(token or os.getenv("TUSHARE_TOKEN", ""))
        if not (token or os.getenv("TUSHARE_TOKEN", "")):
            raise ValueError("TushareProvider 需要 TUSHARE_TOKEN（在 .env 配置），当前未设置")

    def get_index_constituents(self, index_code: str = "000300.SH", trade_date: str = "") -> list[str]:
        """指数成分（月频权重表，取最新月份）。trade_date 为空取最新。"""
        df = self.pro.index_weight(index_code=index_code)
        if df is None or df.empty:
            return []
        if trade_date:
            df = df[df["trade_date"] <= trade_date]
        latest = df["trade_date"].max()
        return df[df["trade_date"] == latest]["con_code"].tolist()

    def get_announcements(self, codes: list[str], start_date: str, end_date: str) -> list[dict]:
        rows = []
        for code in codes:
            df = self.pro.anns_d(ts_code=code, start_date=start_date.replace("-", ""),
                                 end_date=end_date.replace("-", ""))
            if df is None or df.empty:
                continue
            for _, r in df.iterrows():
                rows.append({
                    "doc_type": "announcement",
                    "ts_code": r["ts_code"],
                    "name": r.get("name", ""),
                    "title": r["title"],
                    "ann_date": str(r["ann_date"])[:10],
                    "source_url": r.get("url", ""),
                    "content": r["title"],  # 正文需 PDF 解析（可选增强），标题先行
                })
        return rows

    def get_news(self, start_date: str, end_date: str) -> list[dict]:
        df = self.pro.news(src="", start_date=start_date.replace("-", ""),
                           end_date=end_date.replace("-", ""))
        rows = []
        if df is not None and not df.empty:
            for _, r in df.iterrows():
                rows.append({
                    "doc_type": "news",
                    "ts_code": None,
                    "name": r.get("source", ""),
                    "title": r.get("title", ""),
                    "ann_date": str(r["datetime"])[:10],
                    "source_url": "",
                    "content": r.get("content", ""),
                })
        return rows

    def get_research_reports(self, codes: list[str], start_date: str, end_date: str) -> list[dict]:
        rows = []
        for code in codes:
            df = self.pro.research_report(ts_code=code, start_date=start_date.replace("-", ""),
                                          end_date=end_date.replace("-", ""))
            if df is None or df.empty:
                continue
            for _, r in df.iterrows():
                rows.append({
                    "doc_type": "research_report",
                    "ts_code": code,
                    "name": r.get("org_name", ""),
                    "title": r.get("title", ""),
                    "ann_date": str(r["pub_date"])[:10],
                    "source_url": "",
                    "content": r.get("abstract", ""),
                })
        return rows

    def get_daily_returns(self, codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        frames = []
        for code in codes:
            df = self.pro.daily(ts_code=code, start_date=start_date.replace("-", ""),
                                end_date=end_date.replace("-", ""))
            if df is not None and not df.empty:
                frames.append(df[["trade_date", "ts_code", "pct_chg"]])
        if not frames:
            return pd.DataFrame(columns=["date", "code", "ret_next"])
        out = pd.concat(frames, ignore_index=True)
        out.columns = ["date", "code", "ret_next"]
        out["ret_next"] = out["ret_next"] / 100.0  # 百分比 → 小数
        return out


class FileProvider(DataProvider):
    """本地文件后端（fallback：无 Tushare token / MCP 授权失效时验证全链路）。

    显式标注：这是开发验证后端，真实数据请使用 TushareProvider。
    文件格式：CSV（date, code, title, text, [ret_next]），utf-8。
    """

    def __init__(self, path: str | None = None):
        self.path = path or os.getenv("QUANT_DATA_CSV", "data/sample_announcements.csv")

    def _load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

    def get_index_constituents(self, index_code: str = "000300.SH", trade_date: str = "") -> list[str]:
        return self._load()["code"].dropna().unique().tolist()

    def get_announcements(self, codes: list[str], start_date: str, end_date: str) -> list[dict]:
        df = self._load()
        df["date"] = df["date"].astype(str)
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "doc_type": "announcement",
                "ts_code": r["code"],
                "name": r.get("name", r["code"]),
                "title": r["title"],
                "ann_date": str(r["date"]),
                "source_url": "",
                "content": r.get("text", r["title"]),
            })
        return rows

    def get_news(self, start_date: str, end_date: str) -> list[dict]:
        return []  # 文件后端以公告为主

    def get_research_reports(self, codes: list[str], start_date: str, end_date: str) -> list[dict]:
        return []

    def get_daily_returns(self, codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        df = self._load()
        if "ret_next" not in df.columns:
            return pd.DataFrame(columns=["date", "code", "ret_next"])
        df["date"] = df["date"].astype(str)
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        return df[["date", "code", "ret_next"]].rename(columns={"ret_next": "ret_next"})


def get_provider(name: str = "auto") -> DataProvider:
    """按可用性选择 provider：Tushare token 存在 → TushareProvider，否则 FileProvider。"""
    if name == "tushare":
        return TushareProvider()
    if name == "file":
        return FileProvider()
    # auto：优先真实数据
    if os.getenv("TUSHARE_TOKEN"):
        return TushareProvider()
    return FileProvider()
