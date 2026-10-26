"""Failure-mode analysis computed from the saved results and traces (no model calls).

Turns `results/<config>.jsonl` into the tables the README is filled from: success with intervals, the cause of every failure, what
the verifier did, how often the pipeline fell back, and what each configuration costs in calls, tokens and seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from euagents import evaluate

LABEL = {
    "full": "Graph: planner + verifier + approval gate", "no_planner": "Graph without the planner (raw question as the query)",
    "no_verifier": "Graph without the verifier", "single_agent": "Single agent, free tool use (permissive prompt)",
    "single_agent_directive": "Single agent, directive prompt (must search, then read)",
}
CAUSE_TEXT = {
    "retrieval_miss": "the search never surfaced the gold article",
    "selection_miss": "the gold article was a candidate but the model did not choose to read it",
    "wrong_citation": "the model read the gold article and still cited a different one",
    "ungrounded_citation": "the model cited an article it never retrieved (invented or remembered)",
    "no_answer": "no usable answer (step limit, timeout, error, or no citation)",
}


def load(results_dir: Path) -> dict[str, list[dict]]:
    out = {}
    for name in LABEL:
        p = results_dir / f"{name}.jsonl"
        if p.exists():
            out[name] = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return out


def main_table(res: dict[str, list[dict]]) -> str:
    lines = ["| configuration | tasks | success (95% CI) | LLM calls / task | tokens / task | seconds / task |", "|---|---|---|---|---|---|"]
    for name, rs in res.items():
        ok = sum(r["cause"] == "success" for r in rs)
        lo, hi = evaluate.wilson(ok, len(rs))
        lines.append(f"| {LABEL[name]} | {len(rs)} | {ok / len(rs):.0%} ({lo:.0%}-{hi:.0%}) | {np.mean([r['llm_calls'] for r in rs]):.1f} | "
                     f"{np.mean([r['prompt_tokens'] + r['completion_tokens'] for r in rs]):,.0f} | {np.mean([r['seconds'] for r in rs]):.0f} |")
    return "\n".join(lines)


def cause_table(res: dict[str, list[dict]]) -> str:
    causes = ["retrieval_miss", "selection_miss", "wrong_citation", "ungrounded_citation", "no_answer"]
    lines = ["| configuration | success | " + " | ".join(c.replace("_", " ") for c in causes) + " |", "|---|---|" + "---|" * len(causes)]
    for name, rs in res.items():
        n = len(rs)
        lines.append(f"| {LABEL[name]} | {sum(r['cause'] == 'success' for r in rs)} | " + " | ".join(str(sum(r["cause"] == c for r in rs)) for c in causes) + f" |  (of {n})")
    lines += ["", "What each cause means: " + "; ".join(f"**{c.replace('_', ' ')}**: {t}" for c, t in CAUSE_TEXT.items()) + "."]
    return "\n".join(lines)


def paired_table(res: dict[str, list[dict]]) -> str:
    lines = ["| comparison (same tasks) | A only succeeds | B only succeeds | exact McNemar p |", "|---|---|---|---|"]
    pairs = [("full", "no_planner"), ("full", "no_verifier"), ("full", "single_agent"), ("full", "single_agent_directive"), ("single_agent_directive", "single_agent")]
    for a, b in pairs:
        if a in res and b in res:
            ids = [r["task"] for r in res[a]]
            by_b = {r["task"]: r for r in res[b]}
            common = [i for i in ids if i in by_b]
            ok_a = [next(r for r in res[a] if r["task"] == i)["cause"] == "success" for i in common]
            ok_b = [by_b[i]["cause"] == "success" for i in common]
            n10, n01, p = evaluate.mcnemar_exact(ok_a, ok_b)
            lines.append(f"| {a} vs {b} (n={len(common)}) | {n10} | {n01} | {p:.3f} |")
    return "\n".join(lines)


def pipeline_behaviour(res: dict[str, list[dict]]) -> str:
    out = []
    full = res.get("full", [])
    if full:
        redrafted = [r for r in full if (r.get("attempts") or 1) > 1]
        unver = [r for r in full if not r.get("verified")]
        notes = [n for r in full for n in r.get("notes", [])]
        count = lambda s: sum(s in n for n in notes)  # noqa: E731
        out += [
            f"- **Verifier.** {len(redrafted)} of {len(full)} runs needed a second draft ({sum(r['cause'] == 'success' for r in redrafted)} of those succeeded); "
            f"{len(unver)} ended without passing verification (those are never saved).",
            f"- **Fallbacks.** planner failed {count('planner failed')}x, selector proposed non-candidate keys {count('not candidates')}x, selector failed {count('selector failed')}x, "
            f"fell back to top search results {count('fell back')}x; schema failures {sum(r['schema_failures'] for r in full)}; tool errors {sum(r['tool_errors'] for r in full)}.",
        ]
    sa = res.get("single_agent", [])
    if sa:
        no_tools = [r for r in sa if r["tool_calls"] == 0]
        out.append(f"- **Free tool use.** In {len(no_tools)} of {len(sa)} permissive-prompt runs the model made no tool call at all and answered from memory.")
    sd = res.get("single_agent_directive", [])
    if sd:
        no_tools = [r for r in sd if r["tool_calls"] == 0]
        out.append(f"- **Directive prompt.** {len(no_tools)} of {len(sd)} runs still made no tool call; mean tool calls per run {np.mean([r['tool_calls'] for r in sd]):.1f}.")
    return "\n".join(out)


def examples(res: dict[str, list[dict]], tasks: dict[str, str], per_cause: int = 2) -> str:
    out = []
    for cause in ("retrieval_miss", "selection_miss", "ungrounded_citation", "wrong_citation"):
        shown = 0
        for name in ("full", "single_agent_directive", "single_agent", "no_planner", "no_verifier"):
            for r in res.get(name, []):
                if r["cause"] == cause and shown < per_cause:
                    q = tasks.get(r["task"], r["task"])
                    out.append(f"- **{cause.replace('_', ' ')}** ({name}, {r['task']}): \"{q}\" gold {r['gold']}, cited {r['citations'][:3]}")
                    shown += 1
    return "\n".join(out)


def build(results_dir: Path, readme: Path, tasks_file: Path) -> None:
    res = load(results_dir)
    tasks = {t.id: t.question for t in evaluate.load_tasks(tasks_file)}
    sections = {"main": main_table(res), "causes": cause_table(res), "paired": paired_table(res), "behaviour": pipeline_behaviour(res), "examples": examples(res, tasks)}
    text = readme.read_text(encoding="utf-8")
    for name, body in sections.items():
        a, b = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
        i, j = text.find(a), text.find(b)
        if i == -1 or j < i:
            raise ValueError(f"README is missing the {name} markers")
        text = text[: i + len(a)] + "\n" + body + "\n" + text[j:]
    readme.write_text(text, encoding="utf-8")
