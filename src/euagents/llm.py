"""The model interface the graph depends on, with a local Ollama implementation and a scripted one for tests.

Structured outputs use Ollama's schema-constrained decoding, so the model cannot emit malformed JSON.
That removes one whole class of failure but not the important one: schema-valid output can still be wrong
(an invented citation, an irrelevant article). Both kinds are counted in the failure-mode analysis.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from euagents import tracing

T = TypeVar("T", bound=BaseModel)
DEFAULT_MODEL = "qwen3:1.7b"


class LLMOutputError(Exception):
    """The model never produced output that satisfied the schema, even after feedback."""


class LLM(Protocol):
    async def json(self, name: str, system: str, user: str, schema: type[T]) -> T: ...

    async def chat(self, messages: list[Any], tools: list[dict]) -> Any: ...


def _usage(msg: Any) -> dict[str, Any]:
    md = getattr(msg, "response_metadata", {}) or {}
    return {"prompt_tokens": md.get("prompt_eval_count"), "completion_tokens": md.get("eval_count")}


class OllamaLLM:
    """Qwen3-1.7B (or any Ollama model) at temperature 0 with thinking disabled. num_thread is capped so the
    machine stays usable while a run is going."""

    def __init__(self, model: str = DEFAULT_MODEL, num_ctx: int = 4096, num_predict: int = 400, num_thread: int = 3,
                 timeout: float = 180.0, max_format_retries: int = 2, base_url: str | None = None):
        from langchain_ollama import ChatOllama

        self._cls = ChatOllama
        self.model, self.max_format_retries = model, max_format_retries
        self._kw = {"model": model, "temperature": 0, "seed": 7, "num_ctx": num_ctx, "num_predict": num_predict, "reasoning": False,
                    "num_thread": num_thread, "client_kwargs": {"timeout": timeout}, "keep_alive": "10m"}
        if base_url:
            self._kw["base_url"] = base_url

    async def json(self, name: str, system: str, user: str, schema: type[T]) -> T:
        from langchain_core.messages import HumanMessage, SystemMessage

        chat = self._cls(**self._kw, format=schema.model_json_schema())
        messages: list[Any] = [SystemMessage(system), HumanMessage(user)]
        tracer, last_err = tracing.current(), ""
        for attempt in range(self.max_format_retries + 1):
            started = time.time()
            reply = await chat.ainvoke(messages)
            text = reply.content if isinstance(reply.content, str) else str(reply.content)
            try:
                out = schema.model_validate_json(text)
            except ValidationError as exc:
                last_err = str(exc).splitlines()[0]
                tracer.emit("llm", name, started, attempt=attempt, error=f"schema: {last_err}", output=text, **_usage(reply))
                messages += [reply, HumanMessage(f"That did not match the required format ({last_err}). Reply again with valid JSON only.")]
                continue
            tracer.emit("llm", name, started, attempt=attempt, output=text, **_usage(reply))
            return out
        raise LLMOutputError(f"{name}: no valid output after {self.max_format_retries + 1} attempts ({last_err})")

    async def chat(self, messages: list[Any], tools: list[dict]) -> Any:
        started = time.time()
        reply = await self._cls(**self._kw).bind_tools(tools).ainvoke(messages)
        tracing.current().emit("llm", "chat", started, output=reply.content, tool_calls=getattr(reply, "tool_calls", None), **_usage(reply))
        return reply


class ScriptedLLM:
    """Deterministic stand-in for tests: returns queued outputs per call name, and records every prompt it was given."""

    def __init__(self, script: dict[str, list[Any]]):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[tuple[str, str]] = []

    async def json(self, name: str, system: str, user: str, schema: type[T]) -> T:
        self.calls.append((name, user))
        queue = self.script.get(name)
        if not queue:
            raise LLMOutputError(f"{name}: scripted LLM has no more outputs")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        started = time.time()
        out = item if isinstance(item, schema) else schema.model_validate(item)
        tracing.current().emit("llm", name, started, output=out.model_dump_json())
        return out

    async def chat(self, messages: list[Any], tools: list[dict]) -> Any:
        self.calls.append(("chat", str(messages[-1])))
        queue = self.script.get("chat")
        if not queue:
            raise LLMOutputError("chat: scripted LLM has no more outputs")
        return queue.pop(0)
