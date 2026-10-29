"""数据接入层：统一 DataProvider 接口。

设计目标（对标实习"围绕公告、研究报告及新闻文本的结构化数据体系"）：
  - ChoiceProvider：真实数据后端（东方财富 Choice 金融终端量化接口 EMQuantAPI）
    覆盖：公告 c.cfn("report") / 公司资讯 c.cfn("companynews") /
          沪深300 成分 c.sector / 日线行情 c.csd("CLOSE", 前复权) / 交易日历 c.tradedates
  - FileProvider：本地文件后端（开发/无 Choice 环境时验证全链路；
    显式标注为 fallback，不替代 Choice 作为真实数据源）

【EMQuantAPI 安装与激活】（官方非 pip 包，需两步）：
  1) 安装：从 https://quantapi.eastmoney.com/ 下载 EmQuantAPI_Python.zip，
     Windows 以管理员运行 EmQuantAPISetup.exe（选择 Python 环境），
     或命令行运行 installEmQuantAPI.py 注册；
  2) 激活：在 quantapi.eastmoney.com 绑定手机号 → 运行 LoginActivator.exe
     生成 userinfo 登录令牌；或在 .env 配置 CHOICE_USERNAME/CHOICE_PASSWORD
     走账密登录（c.start 直接账密模式，不生成令牌）。

登录生命周期由本模块懒加载管理：首次取数时 c.start，进程退出时 c.stop（atexit）。
本模块绝不把账号/密码写入代码或日志。
"""
from __future__ import annotations

import atexit
import os
import time
from abc import ABC, abstractmethod

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


# ---------------------------------------------------------------------------
# Choice 指数代码 → 板块代码映射（EMQuantAPI 板块函数 sector 用，官方附注9）
# ---------------------------------------------------------------------------
INDEX_SECTOR_MAP = {
    "000300.SH": "009006195",  # 沪深300成份
    "000905.SH": "009006062",  # 中证500成份
    "000852.SH": "009007552",  # 中证1000成份
    "000016.SH": "009007063",  # 上证50指数成份
    "000010.SH": "009007060",  # 上证180指数成份
}

# 单批证券代码数（控制响应体积与频次，避免触发 EMQuantAPI 限频）
_BATCH_SIZE = 50


class ChoiceProvider(DataProvider):
    """Choice EMQuantAPI 后端（真实数据源）。

    前置条件：本机已安装 Choice 金融终端，且已完成量化接口激活
    （userinfo 令牌，或 .env 配置 CHOICE_USERNAME/CHOICE_PASSWORD 账密登录）。

    EMQuantAPI 函数对照（官方 Python 手册 V2.7.7）：
      - 登录/退出   c.start(options) / c.stop()
      - 板块成分    c.sector(板块代码, enddate)          → Data 为证券代码列表
      - 资讯(公告)  c.cfn(codes, "report", eCfnMode_StartToEnd,
                          "starttime=...,endtime=...")   → Data[code] 为记录列表
      - 资讯(新闻)  c.cfn(codes, "companynews", ...)     → 同上
      - 行情序列    c.csd(codes, "CLOSE", start, end,
                          "Period=1,AdjustFlag=3")       → Data[code][0][j] + Dates
      - 交易日历    c.tradedates(start, end)             → Data 为日期列表
    """

    def __init__(self, username: str | None = None, password: str | None = None):
        try:
            import EMQuantAPI as emapi
        except ImportError as e:
            raise RuntimeError(
                "未找到 EMQuantAPI：请从 https://quantapi.eastmoney.com/ 下载 "
                "EmQuantAPI_Python.zip，Windows 运行 EmQuantAPISetup.exe 或命令行 "
                "运行 installEmQuantAPI.py 注册后重试（详见 README §5）"
            ) from e
        self._emapi = emapi
        self._started = False
        self._username = username or os.getenv("CHOICE_USERNAME", "")
        self._password = password or os.getenv("CHOICE_PASSWORD", "")
        self._constituents_cache: list[str] | None = None
        atexit.register(self.stop)

    # ------------------------------------------------------------------
    # 登录生命周期（懒加载：首次取数时连接，进程退出时断开）
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._started:
            return
        opts = ""
        if self._username and self._password:
            # V2.5.4.6+ 支持账密登录模式（不生成 userinfo 令牌）
            opts = f"UserName={self._username},PassWord={self._password}"
        ret = self._emapi.c.start(opts)
        if getattr(ret, "ErrorCode", -1) != 0:
            raise ConnectionError(
                f"Choice 登录失败 (ErrorCode={ret.ErrorCode}, {ret.ErrorMsg})："
                "请确认 ① 本机已安装并登录 Choice 金融终端 ② 已完成量化接口激活"
                "（quantapi.eastmoney.com 绑定手机号 + LoginActivator 生成 userinfo）"
            )
        self._started = True

    def stop(self) -> None:
        if self._started:
            try:
                self._emapi.c.stop()
            finally:
                self._started = False

    def __del__(self):
        self.stop()

    # ------------------------------------------------------------------
    # 数据接口
    # ------------------------------------------------------------------

    def get_index_constituents(self, index_code: str = "000300.SH", trade_date: str = "") -> list[str]:
        """指数成分（板块函数）。trade_date 为空用日期宏 N（最新成分）。"""
        self._ensure_started()
        sector_code = INDEX_SECTOR_MAP.get(index_code, index_code)
        end = trade_date or "N"  # "N" = 最新（官方日期宏）
        ret = self._emapi.c.sector(sector_code, end)
        if getattr(ret, "ErrorCode", -1) != 0:
            raise ConnectionError(
                f"sector 查询失败 ({ret.ErrorCode}, {ret.ErrorMsg})："
                f"index_code={index_code} → 板块代码={sector_code}"
            )
        codes = [c for c in ret.Data if isinstance(c, str)]
        self._constituents_cache = codes
        return codes

    def get_announcements(self, codes: list[str], start_date: str, end_date: str) -> list[dict]:
        """公告（资讯函数 content=report）。cfn 不返回正文，正文需抓 url 解析（可选增强），
        此处标题先行（与一期口径一致）。"""
        self._ensure_started()
        rows = []
        for batch in self._batches(codes):
            ret = self._emapi.c.cfn(
                ",".join(batch), "report", self._emapi.eCfnMode_StartToEnd,
                f"starttime={start_date.replace('-', '')},endtime={end_date.replace('-', '')}",
            )
            if getattr(ret, "ErrorCode", -1) != 0:
                print(f"[choice] cfn(report) 失败: {ret.ErrorCode} {ret.ErrorMsg}（跳过本批 {len(batch)} 只）")
                continue
            for code in ret.Data:
                for rec in ret.Data[code]:
                    rows.append({
                        "doc_type": "announcement",
                        "ts_code": code,
                        "name": rec.get("medianname", ""),
                        "title": rec.get("title", ""),
                        "ann_date": str(rec.get("datetime", ""))[:10],
                        "source_url": rec.get("url", ""),
                        "content": rec.get("title", ""),  # 正文需 PDF 解析（可选增强），标题先行
                    })
        return rows

    def get_news(self, start_date: str, end_date: str) -> list[dict]:
        """公司资讯（资讯函数 content=companynews）。行业资讯 industrynews 可扩展接入。"""
        self._ensure_started()
        codes = self._constituents_cache or self.get_index_constituents()
        rows = []
        for batch in self._batches(codes):
            ret = self._emapi.c.cfn(
                ",".join(batch), "companynews", self._emapi.eCfnMode_StartToEnd,
                f"starttime={start_date.replace('-', '')},endtime={end_date.replace('-', '')}",
            )
            if getattr(ret, "ErrorCode", -1) != 0:
                print(f"[choice] cfn(companynews) 失败: {ret.ErrorCode} {ret.ErrorMsg}（跳过本批）")
                continue
            for code in ret.Data:
                for rec in ret.Data[code]:
                    rows.append({
                        "doc_type": "news",
                        "ts_code": code,
                        "name": rec.get("medianname", ""),
                        "title": rec.get("title", ""),
                        "ann_date": str(rec.get("datetime", ""))[:10],
                        "source_url": rec.get("url", ""),
                        "content": rec.get("title", ""),
                    })
        return rows

    def get_research_reports(self, codes: list[str], start_date: str, end_date: str) -> list[dict]:
        """券商研报：EMQuantAPI 资讯函数暂无研报分类（content 仅 companynews/industrynews/
        report/regularreport/tradeinfo/sectornews）。返回空并在日志标注，避免静默缺数据；
        如需研报可扩展抓取 Choice 终端研报导出或接入第三方源。"""
        print("[choice] 注意: EMQuantAPI 资讯接口无研报分类，get_research_reports 返回空（如需可扩展接入）")
        return []

    def get_daily_returns(self, codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """日线行情 → 次日收益标签 ret_next（前复权收盘价，日频）。

        口径：ret_next_t = close_{t+1}/close_t - 1（公告当日收盘后信号，取下一交易日收益）。
        """
        self._ensure_started()
        frames = []
        for batch in self._batches(codes):
            ret = self._emapi.c.csd(
                ",".join(batch), "CLOSE",
                start_date.replace("-", ""), end_date.replace("-", ""),
                "Period=1,AdjustFlag=3,ShowBlank=0,Order=1",  # 日频、前复权、升序
            )
            if getattr(ret, "ErrorCode", -1) != 0:
                print(f"[choice] csd 失败: {ret.ErrorCode} {ret.ErrorMsg}（跳过本批）")
                continue
            dates = ret.Dates
            for code in ret.Data:
                closes = ret.Data[code][0]  # 指标序号 0 = CLOSE
                for j, d in enumerate(dates):
                    try:
                        close = float(closes[j])
                    except (TypeError, ValueError):
                        continue
                    frames.append({"date": str(d)[:10], "code": code, "close": close})
        if not frames:
            return pd.DataFrame(columns=["date", "code", "ret_next"])
        df = pd.DataFrame(frames).sort_values(["code", "date"]).reset_index(drop=True)
        df["ret_next"] = df.groupby("code")["close"].transform(
            lambda s: s.shift(-1) / s - 1
        )
        return df[["date", "code", "ret_next"]].dropna().reset_index(drop=True)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _batches(self, codes: list[str]):
        for i in range(0, len(codes), _BATCH_SIZE):
            yield codes[i:i + _BATCH_SIZE]
            time.sleep(0.2)  # 限频保护（资讯/序列函数每分钟 700 次上限）


class FileProvider(DataProvider):
    """本地文件后端（fallback：无 Choice 环境/未安装 EMQuantAPI 时验证全链路）。

    显式标注：这是开发验证后端，真实数据请使用 ChoiceProvider。
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
        return df[["date", "code", "ret_next"]]


def get_provider(name: str = "auto") -> DataProvider:
    """按可用性选择 provider：Choice（EMQuantAPI 可导入）→ FileProvider。

    - name="choice"：强制 Choice 后端（未安装时抛错，便于显式排障）
    - name="file"：强制本地文件后端
    - name="auto"：EMQuantAPI 可导入 → ChoiceProvider；否则 FileProvider
    """
    if name == "choice":
        return ChoiceProvider()
    if name == "file":
        return FileProvider()
    # auto：优先真实数据（Choice 终端量化接口）
    try:
        import EMQuantAPI  # noqa: F401
        return ChoiceProvider()
    except ImportError:
        return FileProvider()
