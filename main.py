"""量化研究 Agent — 主入口（逐步构建演示）。

用法：
    python main.py                 # 默认 mock 模式（无需 API key，验证图编排）
    python main.py --mode live     # 真实 DeepSeek 推理（需 .env 配置 DEEPSEEK_API_KEY）
    python main.py --data <csv>    # 切换到真实公告数据

逐步输出各阶段结果，最终报告写入 output/。
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from config import get_settings
from graph import build_agent


def banner(step: int, title: str, detail: str) -> None:
    print(f"\n{'=' * 64}\n[STEP {step}] {title}\n{detail}\n{'=' * 64}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quant Research Agent (LangGraph + LangChain + LangSmith + DeepSeek)")
    parser.add_argument("--mode", choices=["mock", "live"], help="mock=假LLM验证编排, live=真实DeepSeek")
    parser.add_argument("--data", type=str, help="公告数据 CSV 路径（date,code,title,text）")
    args = parser.parse_args()

    settings = get_settings()
    if args.mode:
        settings.mode = args.mode
    if args.data:
        settings.data_path = args.data
    settings.apply_langsmith_env()

    print(f"运行模式: {settings.mode.upper()}  | 模型: {settings.deepseek_model}  | 数据: {settings.data_path}")
    print(f"LangSmith tracing: {settings.langsmith_tracing and bool(settings.langsmith_api_key)}"
          f" (project: {settings.langsmith_project})")

    # ---- STEP 1: DeepSeek LLM（LangChain 层） ----
    banner(1, "构建 DeepSeek LLM 层（LangChain）",
           "ChatDeepSeek 封装 OpenAI 兼容协议；mock 模式用确定性假 LLM 保证可复现")
    from llm import build_llm
    llm = build_llm(settings)
    print(f"  LLM 实例: {llm.__class__.__name__}")

    # ---- STEP 2: 工具层（LangChain @tool + LangSmith @traceable） ----
    banner(2, "装配工具层（LangChain Tools）",
           "文本清洗/因子构建/信号预测/回测 四组工具，全部 @traceable 可观测")
    import tools
    tool_names = [n for n in tools.__all__]
    print(f"  已注册工具: {', '.join(tool_names)}")

    # ---- STEP 3: 图编排（LangGraph StateGraph） ----
    banner(3, "编排 LangGraph 工作流",
           "ingest → factor → signal →(质量门控)→ backtest → report，含 diagnose 重试循环")
    agent = build_agent(settings)
    try:
        mermaid = agent.get_graph().draw_mermaid()
        Path(settings.output_dir, "graph.mmd").write_text(mermaid, encoding="utf-8")
        print(f"  图已编译，Mermaid 拓扑已导出: {Path(settings.output_dir, 'graph.mmd')}")
    except Exception as e:  # 非阻塞：可视化导出失败不影响运行
        print(f"  （提示: 图拓扑导出跳过 - {e}）")

    # ---- STEP 4: 运行 Agent（LangGraph 执行 + LangSmith 追踪） ----
    banner(4, "运行 Agent（全图执行）",
           "LangGraph 自动完成状态传递、条件分支与循环；LangSmith 记录每个节点/工具调用")
    t0 = time.time()
    result = agent.invoke({"data_path": settings.data_path})
    elapsed = time.time() - t0
    print(f"  执行完成，耗时 {elapsed:.2f}s")

    # ---- STEP 5: 输出各阶段结果 ----
    banner(5, "阶段结果", "数据质量 / 因子 / 信号 / 回测")
    print("■ 数据质量:", json.dumps(result["data_quality_report"], ensure_ascii=False))
    print("■ 因子:", json.dumps(result["factor_meta"], ensure_ascii=False))
    q = {k: v for k, v in result["quality"].items() if k != "passed"}
    print("■ 信号质量:", json.dumps(q, ensure_ascii=False))
    print("■ 回测:", json.dumps(result["backtest"], ensure_ascii=False))
    if result.get("diagnostics"):
        print("■ LLM 诊断记录:")
        for d in result["diagnostics"]:
            print("   -", str(d)[:160])

    # ---- STEP 6: 报告落盘 ----
    banner(6, "DeepSeek 研究报告", "报告写入 output/")
    out_path = Path(settings.output_dir) / f"report_{datetime.now():%Y%m%d_%H%M%S}.md"
    out_path.write_text(f"# Quant Research Agent 报告（{settings.mode} 模式）\n\n{result['report']}", encoding="utf-8")
    print(f"  报告已生成: {out_path}")

    # ---- 汇报 ----
    print("\n" + "=" * 64)
    print("运行摘要:")
    print(f"  数据: {result['data_quality_report']['rows']} 条公告, 去重 {result['data_quality_report']['dup_removed']} 条")
    print(f"  因子: {result['factor_meta']['n_factors']} 个 | RankIC {result['quality']['rank_ic']} | "
          f"ICIR {result['quality']['icir']} | AUC {result['quality']['auc']}")
    print(f"  回测: 年化 {result['backtest']['long_short']['annual_return']} | "
          f"夏普 {result['backtest']['long_short']['sharpe']} | "
          f"最大回撤 {result['backtest']['long_short']['max_drawdown']}")
    print("=" * 64)


if __name__ == "__main__":
    main()
