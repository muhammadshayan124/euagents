"""Run tasks under a configuration, score them, and classify every failure.

A task is a plain-language question with one gold article (the questions come from the companion `eurag`
benchmark, whose labels are mechanically checked against the legal text). A run SUCCEEDS if the answer cites
the gold article. Success is deliberately a citation test, not an answer-quality judgement: it is objective,
reproducible and cheap, and it measures what a legal-research assistant must get right first, pointing at the
correct provision. It does not tell you the prose is good.

Every failure gets exactly one cause, chosen by checking in pipeline order where the gold article was lost.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from euagents import baseline, graph, tracing
from euagents.llm import LLM
from euagents.tools import ToolBox

CONFIGS = {
    "single_agent": None,  # free tool use with a permissive prompt
    "single_agent_directive": None,  # same loop, prompt that orders search-then-read-then-answer
    "full": graph.Config(),
    "no_planner": graph.Config(use_planner=False),
    "no_verifier": graph.Config(use_verifier=False),
}
CAUSES = ["success", "no_answer", "retrieval_miss", "selection_miss", "ungrounded_citation", "wrong_citation"]


@dataclass(frozen=True)
class Task:
    id: str
    question: str
    gold: tuple[str, ...]


def load_tasks(path: Path) -> list[Task]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(Task(d["id"], d["question"], tuple(d["gold"])))
    return out


def classify(result: dict[str, Any], gold: tuple[str, ...]) -> str:
    """One cause per run. Order matters: it follows the pipeline, so the cause is where the gold article was first lost."""
    g = set(gold)
    cites = set(result.get("citations", []))
    if result.get("status") in ("no_answer", "step_limit", "error", "timeout") or not cites:
        return "no_answer"
    if cites & g:
        return "success"
    grounded = set(result.get("evidence_keys", result.get("read", [])))
    if cites - grounded:
        return "ungrounded_citation"  # cited something that was never retrieved/read: an invented or misremembered key
    candidates = set(result.get("candidate_keys", result.get("seen", [])))
    if not g & candidates:
        return "retrieval_miss"  # the search never surfaced the gold article
    if not g & grounded:
        return "selection_miss"  # it was in the results but the model did not read it
    return "wrong_citation"  # it read the gold article and still cited something else


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def mcnemar_exact(a_ok: list[bool], b_ok: list[bool]) -> tuple[int, int, float]:
    """Exact McNemar test on paired success flags. Returns (a-only wins, b-only wins, two-sided p)."""
    n10 = sum(x and not y for x, y in zip(a_ok, b_ok, strict=True))
    n01 = sum(y and not x for x, y in zip(a_ok, b_ok, strict=True))
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    return n10, n01, float(min(1.0, stats.binomtest(n10, n, 0.5).pvalue))


async def run_task(config_name: str, llm: LLM, tools: ToolBox, task: Task, trace_path: Path, timeout: float = 420.0) -> dict[str, Any]:
    tracer = tracing.Tracer(trace_path, run_id=f"{config_name}-{task.id}")
    started = time.time()
    result: dict[str, Any]
    with tracer.activate():
        try:
            if config_name.startswith("single_agent"):
                prompt = baseline.SYSTEM_DIRECTIVE if config_name == "single_agent_directive" else baseline.SYSTEM
                result = await asyncio.wait_for(baseline.run_single_agent(llm, tools, task.question, system=prompt), timeout)
            else:
                g = graph.build_graph(llm, tools, CONFIGS[config_name])
                state = await asyncio.wait_for(graph.run(g, task.question, f"{config_name}-{task.id}"), timeout)
                result = {"status": state.get("status", "no_answer"), "answer": state.get("answer", ""), "citations": state.get("citations", []),
                          "evidence_keys": list(state.get("evidence", {})), "candidate_keys": list(state.get("candidates", {})),
                          "verified": state.get("verified"), "attempts": state.get("attempts"), "notes": state.get("notes", [])}
        except TimeoutError:
            result = {"status": "timeout", "answer": "", "citations": [], "notes": [f"timeout after {timeout}s"]}
        except Exception as exc:  # noqa: BLE001 - a crashed run is a data point, not a reason to stop the benchmark
            result = {"status": "error", "answer": "", "citations": [], "notes": [f"{type(exc).__name__}: {exc}", traceback.format_exc(limit=3)]}
    spans = tracer.spans
    llm_spans = [s for s in spans if s["kind"] == "llm"]
    result.update({
        "task": task.id, "config": config_name, "gold": list(task.gold), "seconds": round(time.time() - started, 1),
        "llm_calls": len(llm_spans), "tool_calls": sum(s["kind"] == "tool" for s in spans),
        "tool_errors": sum(s["kind"] == "tool" and "error" in s for s in spans),
        "schema_failures": sum("error" in s and str(s["error"]).startswith("schema") for s in llm_spans),
        "prompt_tokens": sum(s.get("prompt_tokens") or 0 for s in llm_spans), "completion_tokens": sum(s.get("completion_tokens") or 0 for s in llm_spans),
    })
    result["cause"] = classify(result, task.gold)
    return result


async def run_config(config_name: str, llm: LLM, tasks: list[Task], out_dir: Path, tools: ToolBox, log=print) -> list[dict]:
    """Run (or resume) one configuration. Finished tasks are read back from disk, so an interrupted run continues where it stopped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path, trace_path = out_dir / f"{config_name}.jsonl", out_dir / "traces" / f"{config_name}.jsonl"
    done = {json.loads(line)["task"]: json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()} if results_path.exists() else {}
    for task in tasks:
        if task.id in done:
            continue
        r = await run_task(config_name, llm, tools, task, trace_path)
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        done[task.id] = r
        log(f"[{config_name}] {task.id}: {r['cause']} ({r['seconds']}s, {r['llm_calls']} LLM calls)")
    return [done[t.id] for t in tasks if t.id in done]


def summarise(results: dict[str, list[dict]]) -> dict[str, Any]:
    out: dict[str, Any] = {"configs": {}}
    for name, rs in results.items():
        n = len(rs)
        ok = sum(r["cause"] == "success" for r in rs)
        lo, hi = wilson(ok, n)
        out["configs"][name] = {
            "n": n, "success": ok / n if n else 0, "ci95": [lo, hi],
            "causes": {c: sum(r["cause"] == c for r in rs) for c in CAUSES},
            "mean_seconds": float(np.mean([r["seconds"] for r in rs])), "mean_llm_calls": float(np.mean([r["llm_calls"] for r in rs])),
            "mean_tokens": float(np.mean([r["prompt_tokens"] + r["completion_tokens"] for r in rs])),
            "schema_failures": int(sum(r["schema_failures"] for r in rs)), "tool_errors": int(sum(r["tool_errors"] for r in rs)),
            "verified_share": float(np.mean([bool(r.get("verified")) for r in rs])) if not name.startswith("single_agent") else None,
        }
    names = list(results)
    out["paired"] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ids = [r["task"] for r in results[a]]
            by_b = {r["task"]: r for r in results[b]}
            common = [t for t in ids if t in by_b]
            a_ok = [next(r for r in results[a] if r["task"] == t)["cause"] == "success" for t in common]
            b_ok = [by_b[t]["cause"] == "success" for t in common]
            n10, n01, p = mcnemar_exact(a_ok, b_ok)
            out["paired"][f"{a} vs {b}"] = {"n": len(common), f"{a}_only": n10, f"{b}_only": n01, "p": p}
    return out
