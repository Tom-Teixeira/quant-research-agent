"""配置加载：DeepSeek / LangSmith / 运行模式。
对应量化实习流程中的"数据质量校验"层——所有外部配置在此集中校验。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（.env 所在位置）
ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")


@dataclass
class Settings:
    """集中式配置，带显式校验，避免 KeyError 静默失败。"""

    # --- DeepSeek ---
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_model: str = field(default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))

    # --- LangSmith ---
    langsmith_api_key: str = field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY", ""))
    langsmith_project: str = field(default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "quant-research-agent"))
    langsmith_tracing: bool = field(
        default_factory=lambda: os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    )

    # --- 运行模式 ---
    mode: str = field(default_factory=lambda: os.getenv("QUANT_AGENT_MODE", "mock").lower())

    # --- 数据路径 ---
    data_path: str = field(default_factory=lambda: str(ROOT_DIR / "data" / "sample_announcements.csv"))
    output_dir: str = field(default_factory=lambda: str(ROOT_DIR / "output"))

    # --- 质量门控阈值（对标 ICIR=0.82 的实盘口径） ---
    min_rank_ic: float = 0.015        # 日频 RankIC 门槛：低于此值触发 LLM 诊断重试
    max_retry: int = 1                # LLM 诊断-重试循环上限（防死循环）

    def __post_init__(self) -> None:
        if self.mode not in {"mock", "live"}:
            raise ValueError(f"QUANT_AGENT_MODE 仅支持 mock/live，收到: {self.mode!r}")
        if self.mode == "live" and not self.deepseek_api_key:
            raise ValueError("live 模式需要 DEEPSEEK_API_KEY，请在 .env 中配置（见 .env.example）")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def apply_langsmith_env(self) -> None:
        """将 LangSmith 配置写入进程环境（LangChain/LangGraph 自动读取）。"""
        os.environ.setdefault("LANGSMITH_TRACING", "true" if self.langsmith_tracing and self.langsmith_api_key else "false")
        if self.langsmith_api_key:
            os.environ.setdefault("LANGSMITH_API_KEY", self.langsmith_api_key)
        os.environ.setdefault("LANGSMITH_PROJECT", self.langsmith_project)


def get_settings() -> Settings:
    return Settings()
