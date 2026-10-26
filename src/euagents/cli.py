from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from euagents import evaluate, graph, law, tracing
from euagents.llm import DEFAULT_MODEL, OllamaLLM
from euagents.tools import ToolBox, default_servers

app = typer.Typer(help="Multi-agent research assistant over EU digital law.", no_args_is_help=True)
console = Console(highlight=False)
RESULTS = Path("results")
TASKS = Path("tasks.jsonl")


@app.command()
def fetch() -> None:
    """Download the six acts from the EU Publications Office."""
    for p in law.fetch():
        console.print(f"saved {p}")


@app.command()
def ask(
    question: str,
    model: Annotated[str, typer.Option(help="Ollama model")] = DEFAULT_MODEL,
    memos: Annotated[Path, typer.Option(help="Where approved memos are saved")] = Path("memos"),
    trace: Annotated[Path, typer.Option(help="JSONL trace file")] = Path("results/ask-trace.jsonl"),
) -> None:
    """Answer one question. The answer is shown to you, and nothing is saved unless you approve it."""
    asyncio.run(_ask(question, model, memos, trace))


async def _ask(question: str, model: str, memos: Path, trace: Path) -> None:
    env = {"EUAGENTS_MEMOS": str(memos.resolve())}
    tracer = tracing.Tracer(trace, run_id=uuid.uuid4().hex[:8])
    async with ToolBox(default_servers(env)) as tools:
        g = graph.build_graph(OllamaLLM(model), tools, graph.Config(human="interrupt"))
        thread = tracer.run_id
        with tracer.activate():
            state = await graph.run(g, question, thread)
            while "__interrupt__" in state:
                review = state["__interrupt__"][0].value
                console.print(f"\n[bold]Proposed answer[/bold]\n{review['answer']}\n[dim]sources: {', '.join(review['citations'] or [])}[/dim]")
                if review["problems"]:
                    console.print(f"[yellow]verifier: {'; '.join(review['problems'])}[/yellow]")
                ok = typer.confirm("Save this as a memo?", default=False)
                fb = "" if ok else typer.prompt("What should be changed? (blank to stop)", default="")
                state = await graph.run(g, question, thread, resume={"approved": ok, "feedback": fb})
    console.print(f"\nstatus: {state.get('status')}" + (f"  saved: {state['saved']['saved']}" if state.get("saved") else ""))
    for n in state.get("notes", []):
        console.print(f"[dim]note: {n}[/dim]")
    console.print(f"[dim]trace: {trace}[/dim]")


@app.command("eval")
def eval_cmd(
    config: Annotated[str, typer.Argument(help="single_agent | full | no_planner | no_verifier")],
    n: Annotated[int, typer.Option(help="Number of tasks (a fixed stratified sample, same for every config)")] = 30,
    model: Annotated[str, typer.Option(help="Ollama model")] = DEFAULT_MODEL,
    out: Annotated[Path, typer.Option(help="Results directory")] = RESULTS,
) -> None:
    """Run (or resume) one configuration over the task set."""
    tasks = sample_tasks(n)
    asyncio.run(_eval(config, model, tasks, out))


def sample_tasks(n: int) -> list[evaluate.Task]:
    """A fixed sample: every k-th task, so each act is represented and every configuration sees the identical tasks."""
    tasks = evaluate.load_tasks(TASKS)
    if n >= len(tasks):
        return tasks
    step = len(tasks) / n
    return [tasks[int(i * step)] for i in range(n)]


async def _eval(config: str, model: str, tasks: list[evaluate.Task], out: Path) -> None:
    import tempfile

    env = {"EUAGENTS_MEMOS": tempfile.mkdtemp(prefix="euagents-memos-")}  # evaluation never writes into the real memo directory
    async with ToolBox(default_servers(env)) as tools:
        await evaluate.run_config(config, OllamaLLM(model), tasks, out, tools, log=console.print)


@app.command()
def readme(
    out: Annotated[Path, typer.Option(help="Results directory")] = RESULTS,
    target: Annotated[Path, typer.Option(help="README with <!-- name:start/end --> markers")] = Path("README.md"),
) -> None:
    """Fill the README's result sections from the saved results (so the text cannot drift from the data)."""
    from euagents import analyze

    analyze.build(out, target, TASKS)
    console.print(f"updated {target}")


@app.command()
def report(out: Annotated[Path, typer.Option(help="Results directory")] = RESULTS) -> None:
    """Summarise every finished configuration: success, failure causes, cost, paired tests."""
    results = {}
    for p in sorted(out.glob("*.jsonl")):
        rs = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rs and "cause" in rs[0]:
            results[p.stem] = rs
    summary = evaluate.summarise(results)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    t = Table("config", "n", "success (95% CI)", *evaluate.CAUSES[1:], "s/task", "LLM calls")
    for name, c in summary["configs"].items():
        t.add_row(name, str(c["n"]), f"{c['success']:.0%} ({c['ci95'][0]:.0%}-{c['ci95'][1]:.0%})", *[str(c["causes"][k]) for k in evaluate.CAUSES[1:]],
                  f"{c['mean_seconds']:.0f}", f"{c['mean_llm_calls']:.1f}")
    console.print(t)
    for pair, v in summary["paired"].items():
        console.print(f"{pair}: n={v['n']} p={v['p']:.3f}  {[(k, x) for k, x in v.items() if k.endswith('_only')]}")


if __name__ == "__main__":
    app()
