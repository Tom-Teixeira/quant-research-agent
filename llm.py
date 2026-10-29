"""DeepSeek LLM 层（LangChain 封装）。

- live 模式:  langchain-deepseek 的 ChatDeepSeek（OpenAI 兼容协议），
              DeepSeek 官方不暴露 LangChain 专属 SDK，此包即官方集成。
- mock 模式:  确定性假 LLM。无 API key 也能端到端跑通 LangGraph，
              用于验证编排逻辑；报告的每个占位段都标注"MOCK"。
"""
from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from config import Settings


class _MockChatDeepSeek(BaseChatModel):
    """确定性假 LLM：按输入长度/关键词返回稳定文本，保证图可复现。"""

    model_name: str = "mock-deepseek"

    @property
    def _llm_type(self) -> str:
        return "mock-deepseek"

    def _generate(self, messages: list, stop: list | None = None, run_manager=None, **kwargs: Any):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        text = _compose_mock(messages)
        return ChatResult(generations=[ChatGeneration(text=text, message=AIMessage(content=text))])

    def bind_tools(self, tools: list[BaseTool | dict], **kwargs: Any) -> BaseChatModel:
        return self  # mock 不真正调用工具

    def bind_functions(self, functions: list, **kwargs: Any) -> BaseChatModel:
        return self


def _compose_mock(messages: list) -> str:
    """拼一段确定性的假分析文本，模拟 DeepSeek 的报告风格。"""
    user_parts = [
        m.content
        for m in messages
        if isinstance(m, HumanMessage) and isinstance(m.content, str)
    ]
    joined = " ".join(user_parts)[:400]
    return (
        "[MOCK-DEEPSEEK] 已接收研究请求："
        + (joined if joined else "（空输入）")
        + "。\n结论：因子信号质量达标，建议维持现有组合权重，"
        + "关注高波动期的因子衰减风险。"
    )


def build_llm(settings: Settings) -> BaseChatModel:
    """按运行模式返回真实 DeepSeek 或 mock 模型。"""
    if settings.mode == "mock":
        return _MockChatDeepSeek()

    from langchain_deepseek import ChatDeepSeek  # 延迟导入，mock 模式无需该包

    return ChatDeepSeek(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        temperature=0.3,
        max_tokens=2048,
        timeout=60,
        max_retries=2,
    )
