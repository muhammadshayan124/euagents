"""Tool access over MCP: persistent stdio sessions, retries with backoff, fault injection, tracing.

Two kinds of failure are deliberately kept apart because they need opposite handling:
  ToolTransient  the call did not complete (timeout, broken pipe, injected fault): retrying is reasonable.
  ToolError      the tool ran and said no (unknown article, bad argument): retrying cannot help, so it
                 goes back to the caller, which can fix its arguments.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from euagents import tracing


class ToolError(Exception):
    """The tool executed and reported an error."""


class ToolTransient(Exception):
    """The call did not complete; it may succeed if repeated."""


class FaultInjector:
    """Make the first `fail_first` calls to the named tools fail transiently, to exercise retry logic on purpose."""

    def __init__(self, fail_first: int = 0, tools: set[str] | None = None):
        self.remaining = fail_first
        self.tools = tools

    def maybe_fail(self, tool: str) -> None:
        if self.remaining > 0 and (self.tools is None or tool in self.tools):
            self.remaining -= 1
            raise ToolTransient(f"injected fault on {tool}")


def default_servers(env: dict[str, str] | None = None) -> dict[str, StdioServerParameters]:
    return {
        "law": StdioServerParameters(command=sys.executable, args=["-m", "euagents.servers.lawsearch"], env=env),
        "memo": StdioServerParameters(command=sys.executable, args=["-m", "euagents.servers.memo"], env=env),
    }


def _parse(result: Any) -> Any:
    if result.isError:
        text = " ".join(getattr(b, "text", "") for b in result.content)
        raise ToolError(text.replace("Error executing tool ", "").strip() or "tool error")
    sc = getattr(result, "structuredContent", None)
    if sc:
        return sc["result"] if set(sc) == {"result"} else sc
    out = []
    for block in result.content:
        text = getattr(block, "text", "")
        try:
            out.append(json.loads(text))
        except json.JSONDecodeError:
            out.append(text)
    return out[0] if len(out) == 1 else out


class ToolBox:
    def __init__(self, servers: dict[str, StdioServerParameters] | None = None, retries: int = 2, backoff: float = 0.25,
                 timeout: float = 30.0, faults: FaultInjector | None = None):
        self.servers = servers or default_servers()
        self.retries, self.backoff, self.timeout, self.faults = retries, backoff, timeout, faults
        self._stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}
        self._owner: dict[str, str] = {}  # tool name -> server
        self._schemas: list[dict] = []

    async def __aenter__(self) -> ToolBox:
        for name, params in self.servers.items():
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._sessions[name] = session
            for tool in (await session.list_tools()).tools:
                self._owner[tool.name] = name
                self._schemas.append({"type": "function", "function": {"name": tool.name, "description": tool.description or "", "parameters": tool.inputSchema}})
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    def schemas(self, only: set[str] | None = None) -> list[dict]:
        return [s for s in self._schemas if only is None or s["function"]["name"] in only]

    async def call(self, tool: str, args: dict[str, Any]) -> Any:
        if tool not in self._owner:
            raise ToolError(f"unknown tool {tool!r}")
        tracer = tracing.current()
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with tracer.span("tool", tool, args=args, attempt=attempt) as span:
                    if self.faults:
                        self.faults.maybe_fail(tool)
                    try:
                        result = await asyncio.wait_for(self._sessions[self._owner[tool]].call_tool(tool, args), self.timeout)
                    except TimeoutError as exc:
                        raise ToolTransient(f"{tool} timed out after {self.timeout}s") from exc
                    value = _parse(result)
                    span["result"] = value
                    return value
            except ToolTransient as exc:
                last = exc
                if attempt < self.retries:
                    await asyncio.sleep(self.backoff * (2**attempt) * (0.5 + random.random()))
            # ToolError propagates immediately: repeating the same bad call would fail the same way
        raise ToolTransient(f"{tool} failed after {self.retries + 1} attempts: {last}")
