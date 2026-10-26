from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from euagents import evaluate, graph, law, tracing
from euagents.llm import LLMOutputError, ScriptedLLM
from euagents.tools import FaultInjector, ToolBox, ToolError, ToolTransient, default_servers

DATA = Path(__file__).resolve().parents[1] / "data"
needs_data = pytest.mark.skipif(not (DATA / "GDPR.xhtml").exists(), reason="run `euagents fetch` first")


@pytest.fixture(scope="module")
def index():
    return law.Index(law.load(DATA))


@asynccontextmanager
async def box(tmp_path, **kw):
    env = {"EUAGENTS_DATA": str(DATA), "EUAGENTS_MEMOS": str(tmp_path / "memos")}
    kw.setdefault("backoff", 0.0)
    async with ToolBox(default_servers(env), **kw) as tb:
        tb.memo_dir = tmp_path / "memos"
        yield tb


# -- corpus -------------------------------------------------------------------------------------

@needs_data
def test_corpus_counts_and_search(index):
    assert len(index.articles) == 472
    top = index.search("erasure of personal data", "GDPR", 3)
    assert top and top[0][0].act == "GDPR" and "17" in {a.number for a, _ in top}
    assert index.get("gdpr", "17").title.lower().startswith("right to erasure")
    assert index.get("GDPR", "9999") is None
    assert index.search("zzzzqqq") == []


# -- tracing ------------------------------------------------------------------------------------

def test_tracer_records_success_and_errors_and_writes_jsonl(tmp_path):
    t = tracing.Tracer(tmp_path / "t.jsonl", run_id="r1")
    with t.span("node", "ok", x=1) as s:
        s["result"] = "done"
    with pytest.raises(ValueError), t.span("tool", "boom"):
        raise ValueError("bad")
    rows = tracing.load(tmp_path / "t.jsonl")
    assert [r["name"] for r in rows] == ["ok", "boom"]
    assert rows[0]["result"] == "done" and "error" not in rows[0]
    assert rows[1]["error"].startswith("ValueError")
    assert tracing.current().run_id != "r1"  # nothing is active outside the context manager
    with t.activate():
        assert tracing.current() is t


# -- tools / MCP --------------------------------------------------------------------------------

@needs_data
async def test_mcp_tools_work_over_stdio(tmp_path):
    async with box(tmp_path) as toolbox:
        hits = await toolbox.call("search_articles", {"query": "right to erasure", "act": "GDPR", "k": 3})
        assert hits[0]["key"].startswith("GDPR:")
        art = await toolbox.call("get_article", {"act": "GDPR", "number": "17"})
        assert "erasure" in art["text"].lower()


@needs_data
async def test_tool_error_is_not_retried_and_is_traced(tmp_path):
    async with box(tmp_path) as toolbox:
        t = tracing.Tracer(tmp_path / "x.jsonl")
        with t.activate(), pytest.raises(ToolError, match="no article"):
            await toolbox.call("get_article", {"act": "GDPR", "number": "999"})
        calls = [s for s in t.spans if s["kind"] == "tool"]
        assert len(calls) == 1 and "error" in calls[0]  # one attempt: repeating a bad call cannot help


@needs_data
async def test_transient_faults_are_retried_until_they_clear(tmp_path):
    env = {"EUAGENTS_DATA": str(DATA), "EUAGENTS_MEMOS": str(tmp_path)}
    async with ToolBox(default_servers(env), retries=2, backoff=0.0, faults=FaultInjector(fail_first=2)) as tb:
        t = tracing.Tracer()
        with t.activate():
            out = await tb.call("list_acts", {})
        assert len(out) == 6
        assert [s.get("attempt") for s in t.spans if s["kind"] == "tool"] == [0, 1, 2]
    async with ToolBox(default_servers(env), retries=1, backoff=0.0, faults=FaultInjector(fail_first=5)) as tb:
        with pytest.raises(ToolTransient, match="after 2 attempts"):
            await tb.call("list_acts", {})


@needs_data
async def test_memo_server_refuses_overwrite_and_stays_in_its_directory(tmp_path):
    async with box(tmp_path) as toolbox:
        r = await toolbox.call("save_memo", {"title": "../../etc/passwd test", "body": "x"})
        assert r["saved"].endswith(".md") and "/" not in r["saved"] and ".." not in r["saved"]
        assert (toolbox.memo_dir / r["saved"]).exists()
        with pytest.raises(ToolError, match="already exists"):
            await toolbox.call("save_memo", {"title": "../../etc/passwd test", "body": "y"})
        with pytest.raises(ToolError):
            await toolbox.call("save_memo", {"title": "!!!", "body": "y"})


# -- verifier -----------------------------------------------------------------------------------

EVID = {"GDPR:17": {"heading": "h", "text": "The data subject shall have the right to obtain from the controller the erasure of personal data concerning him or her without undue delay"}}


def test_verifier_catches_the_failure_modes_it_exists_for():
    good = "The data subject can obtain erasure of personal data from the controller without undue delay."
    assert graph.verify_answer(good, ["GDPR:17"], EVID) == []
    assert any("not retrieved" in p for p in graph.verify_answer(good, ["GDPR:99"], EVID))  # invented citation
    assert any("not a valid" in p for p in graph.verify_answer(good, ["Article 17"], EVID))
    assert any("empty" in p for p in graph.verify_answer("", ["GDPR:17"], EVID))
    assert any("no citations" in p for p in graph.verify_answer(good, [], EVID))
    assert any("vocabulary" in p for p in graph.verify_answer("Ships must carry lifeboats and flares for emergency maritime rescue operations.", ["GDPR:17"], EVID))


def test_citation_normalisation_extracts_keys_and_keeps_junk_for_the_verifier():
    raw = ["[AI:50] AI Act, Article 50: Transparency obligations", "AI:50", "see GDPR:17.", "Article 17"]
    assert graph.normalise_citations(raw) == ["AI:50", "GDPR:17", "Article 17"]
    assert graph.verify_answer("whatever the cited text says", ["Article 17"], EVID)  # junk still fails verification


# -- graph --------------------------------------------------------------------------------------

def plan(act="GDPR"):
    return {"queries": ["erasure personal data"], "act": act}


def answer(cites, text="The data subject can obtain erasure of personal data from the controller without undue delay."):
    return {"answer": text, "citations": cites}


@needs_data
async def test_happy_path_saves_a_memo_after_approval(tmp_path):
    async with box(tmp_path) as toolbox:
        llm = ScriptedLLM({"plan": [plan()], "select": [{"keys": ["GDPR:17"]}], "draft": [answer(["GDPR:17"])]})
        g = graph.build_graph(llm, toolbox, graph.Config(human="auto"))
        s = await graph.run(g, "When can I demand deletion of my data?", "t1")
        assert s["status"] == "saved" and s["verified"] and s["attempts"] == 1
        assert (toolbox.memo_dir / s["saved"]["saved"]).read_text(encoding="utf-8").count("GDPR:17") >= 1


@needs_data
async def test_invented_citation_triggers_a_redraft_with_feedback(tmp_path):
    async with box(tmp_path) as toolbox:
        llm = ScriptedLLM({"plan": [plan()], "select": [{"keys": ["GDPR:17"]}], "draft": [answer(["GDPR:99"]), answer(["GDPR:17"])]})
        s = await graph.run(graph.build_graph(llm, toolbox, graph.Config(human="auto")), "q", "t2")
        assert s["attempts"] == 2 and s["status"] == "saved" and s["citations"] == ["GDPR:17"]
        second_prompt = [u for n, u in llm.calls if n == "draft"][1]
        assert "was not retrieved" in second_prompt  # the verifier's complaint reached the model


@needs_data
async def test_persistent_hallucination_is_not_saved(tmp_path):
    async with box(tmp_path) as toolbox:
        llm = ScriptedLLM({"plan": [plan()], "select": [{"keys": ["GDPR:17"]}], "draft": [answer(["GDPR:99"])] * 3})
        s = await graph.run(graph.build_graph(llm, toolbox, graph.Config(human="auto", max_attempts=3)), "q", "t3")
        assert s["attempts"] == 3 and s["status"] == "rejected" and not s.get("saved")
        assert not list(toolbox.memo_dir.glob("*.md"))  # nothing unverified reached the side-effecting tool


@needs_data
async def test_planner_and_selector_failures_fall_back_and_are_recorded(tmp_path):
    async with box(tmp_path) as toolbox:
        llm = ScriptedLLM({"plan": [LLMOutputError("plan: bad")], "select": [{"keys": ["GDPR:555"]}], "draft": [answer(["GDPR:17"])]})
        s = await graph.run(graph.build_graph(llm, toolbox, graph.Config(human="off")), "erasure of personal data", "t4")
        assert any("planner failed" in n for n in s["notes"])
        assert any("not candidates" in n for n in s["notes"]) and any("fell back" in n for n in s["notes"])
        assert s["status"] in ("answered", "unverified")  # human=off never saves


@needs_data
async def test_human_in_the_loop_approve(tmp_path):
    async with box(tmp_path) as toolbox:
        llm = ScriptedLLM({"plan": [plan()], "select": [{"keys": ["GDPR:17"]}], "draft": [answer(["GDPR:17"])]})
        g = graph.build_graph(llm, toolbox, graph.Config(human="interrupt"))
        paused = await graph.run(g, "q", "h1")
        assert "__interrupt__" in paused and not list(toolbox.memo_dir.glob("*.md"))  # nothing saved while waiting
        assert paused["__interrupt__"][0].value["citations"] == ["GDPR:17"]
        done = await graph.run(g, "q", "h1", resume={"approved": True})
        assert done["status"] == "saved" and list(toolbox.memo_dir.glob("*.md"))


@needs_data
async def test_human_rejection_sends_feedback_back_to_the_drafter(tmp_path):
    async with box(tmp_path) as toolbox:
        llm = ScriptedLLM({"plan": [plan()], "select": [{"keys": ["GDPR:17"]}], "draft": [answer(["GDPR:17"], "Short."), answer(["GDPR:17"])]})
        g = graph.build_graph(llm, toolbox, graph.Config(human="interrupt"))
        await graph.run(g, "q", "h2")
        paused = await graph.run(g, "q", "h2", resume={"approved": False, "feedback": "mention the time limit"})
        assert "__interrupt__" in paused  # asked again about the second draft
        assert "mention the time limit" in [u for n, u in llm.calls if n == "draft"][1]
        done = await graph.run(g, "q", "h2", resume={"approved": True})
        assert done["status"] == "saved" and done["attempts"] == 2


@needs_data
async def test_tool_failure_during_research_degrades_instead_of_crashing(tmp_path):
    env = {"EUAGENTS_DATA": str(DATA), "EUAGENTS_MEMOS": str(tmp_path)}
    async with ToolBox(default_servers(env), retries=0, backoff=0.0, faults=FaultInjector(fail_first=99, tools={"search_articles"})) as tb:
        llm = ScriptedLLM({"plan": [plan()], "draft": []})
        s = await graph.run(graph.build_graph(llm, tb, graph.Config(human="off")), "q", "t5")
        assert s["status"] == "no_answer" and any("search failed" in n for n in s["notes"])


# -- evaluation ---------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("result", "gold", "cause"),
    [
        ({"status": "saved", "citations": ["GDPR:17"], "evidence_keys": ["GDPR:17"]}, ("GDPR:17",), "success"),
        ({"status": "timeout", "citations": []}, ("GDPR:17",), "no_answer"),
        ({"status": "answered", "citations": ["GDPR:5"], "evidence_keys": ["GDPR:5"], "candidate_keys": ["GDPR:5", "GDPR:6"]}, ("GDPR:17",), "retrieval_miss"),
        ({"status": "answered", "citations": ["GDPR:5"], "evidence_keys": ["GDPR:5"], "candidate_keys": ["GDPR:5", "GDPR:17"]}, ("GDPR:17",), "selection_miss"),
        ({"status": "answered", "citations": ["GDPR:5"], "evidence_keys": ["GDPR:5", "GDPR:17"], "candidate_keys": ["GDPR:5", "GDPR:17"]}, ("GDPR:17",), "wrong_citation"),
        ({"status": "answered", "citations": ["GDPR:99"], "evidence_keys": ["GDPR:17"], "candidate_keys": ["GDPR:17"]}, ("GDPR:17",), "ungrounded_citation"),
        ({"status": "answered", "citations": ["GDPR:5"], "read": ["GDPR:5", "GDPR:17"], "seen": ["GDPR:5", "GDPR:17"]}, ("GDPR:17",), "wrong_citation"),  # baseline shape
    ],
)
def test_failure_classification(result, gold, cause):
    assert evaluate.classify(result, gold) == cause


def test_wilson_and_mcnemar():
    lo, hi = evaluate.wilson(8, 10)
    assert lo < 0.8 < hi and 0.4 < lo < 0.5 and hi < 0.97
    assert evaluate.wilson(0, 0) == (0.0, 0.0)
    a = [True] * 9 + [False]
    b = [False] * 6 + [True] * 4
    n10, n01, p = evaluate.mcnemar_exact(a, b)
    assert (n10, n01) == (6, 1) and 0.05 < p < 0.15
    assert evaluate.mcnemar_exact(a, a)[2] == 1.0


# -- single-agent baseline --------------------------------------------------------------------------


@needs_data
async def test_single_agent_loop_uses_tools_and_extracts_grounded_citations(tmp_path):
    from langchain_core.messages import AIMessage

    from euagents import baseline

    script = {"chat": [
        AIMessage(content="", tool_calls=[{"name": "search_articles", "args": {"query": "erasure", "act": "GDPR", "k": 3}, "id": "1"}]),
        AIMessage(content="", tool_calls=[{"name": "get_article", "args": {"act": "GDPR", "number": "17"}, "id": "2"}]),
        AIMessage(content="You can demand erasure of your personal data (GDPR:17)."),
    ]}
    async with box(tmp_path) as tb:
        r = await baseline.run_single_agent(ScriptedLLM(script), tb, "When can I demand deletion of my data?")
    assert r["status"] == "answered" and r["citations"] == ["GDPR:17"] and "GDPR:17" in r["read"] and r["steps"] == 3
    assert evaluate.classify(r, ("GDPR:17",)) == "success"


@needs_data
async def test_single_agent_that_never_calls_a_tool_is_classified_as_ungrounded(tmp_path):
    from langchain_core.messages import AIMessage

    from euagents import baseline

    script = {"chat": [AIMessage(content="Under GDPR:17 you may.")]}
    async with box(tmp_path) as tb:
        r = await baseline.run_single_agent(ScriptedLLM(script), tb, "q")
    assert r["read"] == [] and r["citations"] == ["GDPR:17"]
    assert evaluate.classify(r, ("GDPR:17",)) == "success"  # it happened to cite the right key...
    assert evaluate.classify(r, ("GDPR:5",)) == "ungrounded_citation"  # ...and when wrong, the cause is that it cited what it never read


@needs_data
async def test_single_agent_stops_at_the_step_limit(tmp_path):
    from langchain_core.messages import AIMessage

    from euagents import baseline

    call = AIMessage(content="", tool_calls=[{"name": "list_acts", "args": {}, "id": "x"}])
    async with box(tmp_path) as tb:
        r = await baseline.run_single_agent(ScriptedLLM({"chat": [call] * 10}), tb, "q", max_steps=3)
    assert r["status"] == "step_limit" and r["steps"] == 3
    assert evaluate.classify(r, ("GDPR:17",)) == "no_answer"


def test_prompt_variants_differ_and_the_directive_one_orders_the_procedure():
    from euagents import baseline

    assert baseline.SYSTEM != baseline.SYSTEM_DIRECTIVE
    assert "get_article" in baseline.SYSTEM_DIRECTIVE and "Never answer before" in baseline.SYSTEM_DIRECTIVE
