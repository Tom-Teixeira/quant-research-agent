# Quant Research Agent - 混合检索中文分词（jieba）

# 用途：把中文公告/新闻/查询切分为空格分隔词串，供 PostgreSQL tsvector
#       ('simple' 配置按空白切词) 构建 BM25 稀疏检索索引。
# 说明：'simple' 分词器不识别中文，原始中文会被整体当作一个 token，
#       导致 BM25 稀疏检索失效；应用层分词是本项目的标准解法。
# 依赖：jieba>=0.42（requirements.txt 已声明）
from __future__ import annotations

import re
from functools import lru_cache

_jieba = None

# 股票代码模式（600100.SH / 000300.SZ / 8xxxxx.BJ）：分词前保护为整体
_CODE_RE = re.compile(r"(\d{6})\.(SH|SZ|BJ)", re.IGNORECASE)
_TICKER_RE = re.compile(r"^TICK(\d+)$")


def _get_jieba():
    global _jieba
    if _jieba is None:
        import jieba

        jieba.setLogLevel(60)  # 关闭 jieba 初始化日志
        _jieba = jieba
    return _jieba


@lru_cache(maxsize=4096)
def segment_chinese(text: str) -> str:
    """中文空格分词：'公司拟回购股份' -> '公司 拟 回购 股份'。

    股票代码（600100.SH 等）在分词前被替换为 TICK{n} 占位、分词后还原，
    保证代码作为整体 token 供 BM25 命中代码类查询。
    """
    text = (text or "").strip()
    if not text:
        return ""
    placeholders: list[str] = []

    def keep(m: re.Match) -> str:
        placeholders.append(f"{m.group(1)}.{m.group(2).upper()}")
        return f"TICK{len(placeholders) - 1}"

    masked = _CODE_RE.sub(keep, text)
    jieba = _get_jieba()
    words = []
    for w in jieba.cut(masked):
        w = w.strip()
        if not w:
            continue
        m = _TICKER_RE.fullmatch(w)
        words.append(placeholders[int(m.group(1))] if m else w)
    return " ".join(words)
