# Quant Research Agent - 混合检索/自适应 RAG 验证（无需数据库部分）

# 覆盖：
#  ① RRF 融合公式与排序正确性（reciprocal_rank_fusion）
#  ② 中文分词（segment_chinese，BM25 稀疏路前置）
#  ③ 查询复杂度路由规则（classify_query / _expand_query 的确定性分支）
#  ④ V1 mock 图回归（文件后端，不依赖数据库）
# 数据库端到端（002 迁移 + hybrid_search SQL）需在本机 PG 执行：
#   python scripts/backfill_tsv.py && python main_v2.py --stage rag --mode hybrid

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def test_rrf() -> None:
    """RRF：两路排名（doc A 双路命中、B 仅稠密、C 仅稀疏）→ 融合分排序。"""
    from db.db_client import reciprocal_rank_fusion

    dense = [{"doc_id": 1}, {"doc_id": 2}, {"doc_id": 4}]
    sparse = [{"doc_id": 1}, {"doc_id": 3}, {"doc_id": 4}]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    scores = dict(fused)
    # 公式复算
    expected_1 = 1 / (60 + 1) * 2            # 两路 rank=1
    expected_4 = 1 / (60 + 3) + 1 / (60 + 3)  # 两路 rank=3
    expected_2 = 1 / (60 + 2)
    check("RRF 双路命中叠加", abs(scores[1] - expected_1) < 1e-9,
          f"{scores[1]} vs {expected_1}")
    check("RRF 同排名合并", abs(scores[4] - expected_4) < 1e-9,
          f"{scores[4]} vs {expected_4}")
    check("RRF 单路贡献", abs(scores[2] - expected_2) < 1e-9)
    check("RRF 排序 doc1 居首", fused[0][0] == 1, str(fused))
    # 边界：空列表
    check("RRF 空输入", reciprocal_rank_fusion([]) == [])
    # 一致性：单路时等价于原顺序的逆序分数
    single = reciprocal_rank_fusion([[{"doc_id": 5}, {"doc_id": 6}]], k=60)
    check("RRF 单路顺序", [d for d, _ in single] == [5, 6])


def test_segment() -> None:
    from rag.segment import segment_chinese

    s = segment_chinese("公司拟回购股份用于员工持股计划")
    check("jieba 中文分词", "公司" in s and "回购" in s and "股份" in s, s)
    s2 = segment_chinese("600100.SH 发布回购公告")
    check("保留代码整体", "600100.SH" in s2, s2)
    check("空输入", segment_chinese("") == "")
    check("None 输入", segment_chinese(None) == "")


def test_router() -> None:
    from rag.adaptive_rag import classify_query, _expand_query

    check("direct：概念题", classify_query("什么是RankIC？") == "direct")
    check("direct：定义题", classify_query("请解释一下夏普比率") == "direct")
    check("complex：对比题", classify_query("对比A和B两家公司的回购增持公告") == "complex")
    check("complex：趋势题", classify_query("分析三季度业绩预增的趋势") == "complex")
    check("complex：归因题", classify_query("为什么该股近期连续上涨") == "complex")
    check("simple：单一事件", classify_query("茅台最近回购进展") == "simple")
    check("direct：空查询", classify_query("") == "direct")
    # 查询扩展（无 LLM 时只返回原文，保持确定性）
    check("扩展：无 LLM 单查询", _expand_query("回购增持") == ["回购增持"])


def test_adaptive_retrieve_logic() -> None:
    """adaptive_retrieve 全链逻辑（假 DB/Embedder，不连库）：
    direct → 不检索；simple → 单查询单路；complex → 多查询多路 + 二次 RRF。"""
    from rag.adaptive_rag import adaptive_retrieve

    class FakeDB:
        def __init__(self):
            self.calls = 0

        def hybrid_search(self, query_seg, query_vec, top_k=5, k=60, **kw):
            self.calls += 1
            if "回购" in query_seg and "增持" in query_seg:
                docs = [{"doc_id": 1, "title": "回购A"}, {"doc_id": 2, "title": "增持B"}]
            elif "对比" in query_seg:
                docs = [{"doc_id": 4, "title": "对比实体A"}]
            else:
                docs = [{"doc_id": 9, "title": "其他"}]
            hits = [dict(d, doc_type="announcement", ts_code="600100.SH",
                         name="X", ann_date="2024-01-01", chunk_text="", dense_rank=i + 1)
                    for i, d in enumerate(docs)]
            return {"evidence": hits, "dense": hits, "sparse": [], "k": k}

    class FakeEmbedder:
        def embed(self, texts):
            return [[0.1] * 512] * len(texts)

    db = FakeDB()
    emb = FakeEmbedder()

    # direct：不检索
    r = adaptive_retrieve("什么是RankIC？", db, emb, mode="adaptive")
    check("adaptive direct 不检索", r["route"] == "direct" and r["evidence"] == [])

    # simple：单查询
    db.calls = 0
    r = adaptive_retrieve("回购增持", db, emb, mode="adaptive")
    check("adaptive simple 单查询", r["route"] == "simple" and len(r["queries"]) == 1
          and db.calls == 1, f"{r['route']} {r['queries']} calls={db.calls}")

    # complex：多查询多路召回 + 证据重排（doc1 多查询命中应排前）
    db.calls = 0
    r = adaptive_retrieve("对比A和B的回购增持与员工持股公告", db, emb, mode="adaptive")
    check("adaptive complex 多查询", r["route"] == "complex" and 2 <= len(r["queries"]) <= 3
          and db.calls == len(r["queries"]), f"{r['route']} queries={r['queries']} calls={db.calls}")
    ids = [e["doc_id"] for e in r["evidence"]]
    check("adaptive complex 证据重排 doc1 居首", ids[0] == 1, str(ids))

    # hybrid 模式：固定多路（跳过分类，仍多查询）
    r = adaptive_retrieve("回购增持", db, emb, mode="hybrid")
    check("hybrid 模式路由 complex", r["route"] == "complex")

    # vector 模式：走 db.search 单路
    class FakeDBV(FakeDB):
        def search(self, query_vec, top_k=5, **kw):
            return [{"doc_id": 7, "title": "向量命中"}]

    r = adaptive_retrieve("回购增持", FakeDBV(), emb, mode="vector")
    check("vector 模式单路", r["evidence"][0]["doc_id"] == 7)


def test_v1_mock_graph() -> None:
    """V1 mock 全图回归（文件后端，验证改动未破坏一期链路）。"""
    from config import get_settings
    from graph import build_agent

    settings = get_settings()
    agent = build_agent(settings)
    result = agent.invoke({"data_path": str(
        Path(__file__).resolve().parents[1] / "data" / "sample_announcements.csv")})
    q = result["quality"]
    check("V1 mock 全图跑通", q.get("passed") is not None, str(q))
    check("V1 RankIC 数值", isinstance(q.get("rank_ic"), float))
    check("V1 回测产出", "annual_return" in result["backtest"]["long_short"],
          str(result["backtest"]))


if __name__ == "__main__":
    print("== RRF 融合 ==")
    test_rrf()
    print("== 中文分词 ==")
    test_segment()
    print("== 自适应路由（规则） ==")
    test_router()
    print("== adaptive_retrieve 全链（假 DB） ==")
    test_adaptive_retrieve_logic()
    print("== V1 mock 图回归 ==")
    test_v1_mock_graph()
    print(f"\n结果: {PASS} 通过 / {FAIL} 失败")
    sys.exit(1 if FAIL else 0)
