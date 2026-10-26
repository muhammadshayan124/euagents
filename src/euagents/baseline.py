"""The comparison point: one model, free tool use, no scaffolding.

A plain tool-calling loop. The model decides which tool to call, with what arguments, when to stop and what
to cite. This is what most "agent" demos do, and the benchmark asks whether the graph's structure earns its
complexity on a small model.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from euagents import tracing
from euagents.llm import LLM, LLMOutputError
from euagents.tools import ToolBox, ToolError, ToolTransient

CITE_RE = re.compile(r"\b(AI|GDPR|DSA|DATA|NIS2|CRA):(\d+[a-z]?)\b")
SYSTEM = ("You answer questions about EU digital law (AI Act, GDPR, DSA, Data Act, NIS2, Cyber Resilience Act) using the tools. "
          "Search, read the relevant article, then answer in at most 100 words and cite each article you used as ACT:NUMBER, "
          "for example GDPR:17. Only cite articles you have read.")
SYSTEM_DIRECTIVE = (
    "You are a legal research assistant with NO memorised knowledge of the text of EU digital law. You may only answer from tool results. "
    "Procedure, always in this order: (1) call search_articles with short keyword queries; (2) call get_article for the best-matching article; "
    "(3) only then answer in at most 100 words and cite the articles you read as ACT:NUMBER, for example GDPR:17. "
    "Never answer before you have called get_article. Never cite an article you have not read."
)
TOOLS = {"search_articles", "get_article", "list_acts"}


async def run_single_agent(llm: LLM, tools: ToolBox, question: str, max_steps: int = 6, system: str = SYSTEM) -> dict[str, Any]:
    tracer = tracing.current()
    messages: list[Any] = [SystemMessage(system), HumanMessage(question)]
    seen: set[str] = set()  # every article key that appeared in any tool result
    read: set[str] = set()  # articles actually opened with get_article
    notes: list[str] = []
    schemas = tools.schemas(TOOLS)
    for step in range(max_steps):
        try:
            reply = await llm.chat(messages, schemas)
        except LLMOutputError as exc:
            return {"status": "no_answer", "answer": "", "citations": [], "seen": sorted(seen), "read": sorted(read), "steps": step, "notes": [str(exc)]}
        messages.append(reply)
        calls = getattr(reply, "tool_calls", None) or []
        if not calls:
            text = reply.content if isinstance(reply.content, str) else str(reply.content)
            cites = sorted({f"{a}:{n}" for a, n in CITE_RE.findall(text)})
            return {"status": "answered" if text.strip() else "no_answer", "answer": text, "citations": cites, "seen": sorted(seen), "read": sorted(read),
                    "steps": step + 1, "notes": notes}
        for call in calls:
            name, args = call["name"], call.get("args", {})
            with tracer.span("node", "agent_tool_step", tool=name, args=args):
                try:
                    out = await tools.call(name, args)
                    content = json.dumps(out)[:3000]
                    for item in out if isinstance(out, list) else [out]:
                        if isinstance(item, dict) and "key" in item:
                            seen.add(item["key"])
                            if name == "get_article":
                                read.add(item["key"])
                except (ToolError, ToolTransient) as exc:
                    content = f"Error: {exc}"
                    notes.append(f"{name}: {exc}")
            messages.append(ToolMessage(content=content, tool_call_id=call.get("id") or name))
    return {"status": "step_limit", "answer": "", "citations": [], "seen": sorted(seen), "read": sorted(read), "steps": max_steps, "notes": notes}
