"""生成示例公告数据（跑通全图用）。

真实场景：替换为你自己的公告/新闻 CSV，格式要求：
  date, code, title, text [, ret_next]
其中 ret_next 为次日收益标签（无则脚本会以合成标签兜底）。

生成逻辑：12 只模拟股票 × 14 个交易日，每天每家 0~2 条公告，
保证截面样本 >= 10，确保 RankIC 可按日计算。
"""
from __future__ import annotations

import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

CODES = [f"60{i:04d}.SH" for i in range(100, 112)]  # 12 只模拟沪深股票

EVENT_TEMPLATES = [
    ("业绩预增", "公司预计上半年净利润同比增长{}%，主要受益于主营业务量价齐升，业绩超预期。", 2.5),
    ("中标", "公司公告中标{}亿元重大项目，中标金额创年内新高，利好公司未来营收。", 2.0),
    ("回购", "公司拟以不超过{}亿元回购股份用于股权激励，彰显管理层信心。", 2.0),
    ("增持", "控股股东计划增持公司股份，累计增持比例不超过总股本2%。", 2.0),
    ("分红", "公司董事会提议年度每10股派发现金红利{}元，股息率提升。", 1.5),
    ("常规公告", "公司召开董事会会议，审议通过日常经营相关议案，经营正常。", 0.2),
    ("减持", "股东计划减持公司股份不超过总股本1.5%，短期或对股价形成压力。", -2.0),
    ("诉讼", "公司公告涉及一宗标的额为{}万元的诉讼事项，结果存在不确定性。", -1.5),
    ("处罚", "公司因信息披露违规收到监管警示函，被责令整改。", -2.5),
    ("停产", "公司部分产线因环保检查停产整顿，预计影响当月产量。", -2.0),
]


def build_sample_data(path: str) -> None:
    rows = []
    start = date(2024, 1, 2)
    days = [start + timedelta(days=i) for i in range(20) if (start + timedelta(days=i)).weekday() < 5][:14]

    for d in days:
        active = random.sample(CODES, k=random.randint(10, 12))  # 保证每天截面 >= 10
        for code in active:
            n_docs = random.choices([0, 1, 2], weights=[0.2, 0.6, 0.2])[0]
            for _ in range(n_docs):
                event, tmpl, _ = random.choice(EVENT_TEMPLATES)
                num = round(random.uniform(0.5, 30), 1)
                title = f"{code.split('.')[0]}：{event}公告"
                text = tmpl.format(num)
                rows.append({"date": d.isoformat(), "code": code, "title": title, "text": text})

    df = pd.DataFrame(rows)
    # 合成次日收益标签：公告情感/事件强度 + 噪声（保证 ML 有可学信号）
    # 注意：@tool 包装后的 StructuredTool 不能直接传给 pandas map（签名解析冲突），
    #       通过 .func 访问底层函数（LangChain 工具的真实逻辑）。
    from tools.factor_tools import event_intensity, sentiment_score

    df["_score"] = df["text"].map(lambda t: sentiment_score.func(t) + event_intensity.func(t))
    df["ret_next"] = df.groupby("date")["_score"].transform(
        lambda x: x / (x.abs().sum() + 1e-8) * 0.02
    ) + np.random.normal(0, 0.005, len(df))
    df = df.drop(columns="_score").sort_values(["date", "code"]).reset_index(drop=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"示例数据已生成: {path}  共 {len(df)} 条公告，{df['date'].nunique()} 个交易日，{df['code'].nunique()} 家公司")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    out = Path(__file__).resolve().parents[1] / "data" / "sample_announcements.csv"
    build_sample_data(str(out))
