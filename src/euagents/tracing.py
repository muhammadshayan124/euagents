"""A small tracer: every LLM call, tool call and graph node becomes one JSON line.

Why not just print logs: the failure-mode analysis is computed *from these traces*. Each span records
what went in (summarised), what came out (summarised), how long it took, token counts for LLM calls
and the error if there was one, so a failed run can be diagnosed after the fact without rerunning it.
Spans are written as they finish (JSONL), so a crash or timeout still leaves a usable partial trace.
"""

from __future__ import annotations

import contextvars
import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_current: contextvars.ContextVar[Tracer | None] = contextvars.ContextVar("euagents_tracer", default=None)


def _short(value: Any, limit: int = 400) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"...[+{len(value) - limit} chars]"
    if isinstance(value, dict):
        return {k: _short(v, limit) for k, v in list(value.items())[:20]}
    if isinstance(value, (list, tuple)):
        return [_short(v, limit) for v in list(value)[:20]] + ([f"...[+{len(value) - 20} items]"] if len(value) > 20 else [])
    return value


class Tracer:
    def __init__(self, path: Path | None = None, run_id: str | None = None):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.path = path
        self.spans: list[dict] = []
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, kind: str, name: str, started: float, **fields: Any) -> dict:
        span = {"run": self.run_id, "kind": kind, "name": name, "t": round(started, 3), "ms": round((time.time() - started) * 1000, 1),
                **{k: _short(v) for k, v in fields.items()}}
        self.spans.append(span)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(span, ensure_ascii=False) + "\n")
        return span

    @contextmanager
    def span(self, kind: str, name: str, **fields: Any):
        """Context manager that records a span even if the body raises (the error is captured, then re-raised)."""
        started = time.time()
        extra: dict[str, Any] = {}
        try:
            yield extra
        except BaseException as exc:
            self.emit(kind, name, started, error=f"{type(exc).__name__}: {exc}", **fields, **extra)
            raise
        else:
            self.emit(kind, name, started, **fields, **extra)

    def event(self, name: str, **fields: Any) -> None:
        self.emit("event", name, time.time(), **fields)

    @contextmanager
    def activate(self):
        token = _current.set(self)
        try:
            yield self
        finally:
            _current.reset(token)


def current() -> Tracer:
    """The active tracer, or a throwaway one so library code never has to check for None."""
    return _current.get() or Tracer()


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
