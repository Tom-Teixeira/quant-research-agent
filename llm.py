from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from config import Settings


class _MockChatDeepSeek(BaseChatModel):
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
        return self

    def bind_functions(self, functions: list, **kwargs: Any) -> BaseChatModel:
        return self


def _compose_mock(messages: list) -> str:
    user_parts = [
        m.content
        for m in messages
        if isinstance(m, HumanMessage) and isinstance(m.content, str)
    ]
    joined = " ".join(user_parts)[:400]
    return (
        "[MOCK] received: " + (joined or "(empty)")
        + "\nSignal within range; watch factor decay in high-vol periods."
    )


def build_llm(settings: Settings) -> BaseChatModel:
    if settings.mode == "mock":
        return _MockChatDeepSeek()

    from langchain_deepseek import ChatDeepSeek

    return ChatDeepSeek(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        temperature=0.3,
        max_tokens=2048,
        timeout=60,
        max_retries=2,
    )
