from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")


@dataclass
class Settings:
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_model: str = field(default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))

    langsmith_api_key: str = field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY", ""))
    langsmith_project: str = field(default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "quant-research-agent"))
    langsmith_tracing: bool = field(
        default_factory=lambda: os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    )

    mode: str = field(default_factory=lambda: os.getenv("QUANT_AGENT_MODE", "mock").lower())

    data_path: str = field(default_factory=lambda: str(ROOT_DIR / "data" / "sample_announcements.csv"))
    output_dir: str = field(default_factory=lambda: str(ROOT_DIR / "output"))

    min_rank_ic: float = 0.015
    max_retry: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"mock", "live"}:
            raise ValueError(f"mode must be mock or live, got {self.mode!r}")
        if self.mode == "live" and not self.deepseek_api_key:
            raise ValueError("live mode requires DEEPSEEK_API_KEY")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def apply_langsmith_env(self) -> None:
        os.environ.setdefault(
            "LANGSMITH_TRACING",
            "true" if self.langsmith_tracing and self.langsmith_api_key else "false",
        )
        if self.langsmith_api_key:
            os.environ.setdefault("LANGSMITH_API_KEY", self.langsmith_api_key)
        os.environ.setdefault("LANGSMITH_PROJECT", self.langsmith_project)


def get_settings() -> Settings:
    return Settings()
