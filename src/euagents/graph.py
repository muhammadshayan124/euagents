"""The agent graph.

    plan -> research -> draft -> verify -+-> human -> save -> END
                          ^              |      |
                          +-- retry <----+      +-- rejected: back to draft with the reviewer's feedback

Design choices, each aimed at what a 1.7B model gets wrong:
  * Decisions are small and schema-constrained (a list of search queries, a short list of keys), not one
    open-ended "do the task" prompt.
  * Tools are called by the graph, not chosen freely by the model; the model chooses *which* articles to
    read from candidates it was actually shown (a key it was not shown is rejected, then falls back).
  * The verifier is deterministic code, not another model: every citation must be an article that was
    actually retrieved in this run, and the answer must overlap the cited text. This is what catches
    invented citations, the failure small models make most.
  * The only side effect (saving a memo) sits behind a human approval gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from euagents import law, tracing
from euagents.llm import LLM, LLMOutputError
from euagents.tools import ToolBox, ToolError, ToolTransient

KEY_RE = re.compile(r"^(AI|GDPR|DSA|DATA|NIS2|CRA):\d+[a-z]?$")
EVIDENCE_CHARS = 1500
ACT_GUIDE = "\n".join(f"- {code}: {name}" for code, (_, name) in law.ACTS.items())


class Plan(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=3, description="1-3 short keyword queries using legal vocabulary")
    act: Literal["AI", "GDPR", "DSA", "DATA", "NIS2", "CRA"] | None = Field(None, description="the act the question is clearly about, else null")


class Selection(BaseModel):
    keys: list[str] = Field(min_length=1, max_length=3, description="keys of the candidate articles that best answer the question")


class Answer(BaseModel):
    answer: str = Field(description="at most 100 words, using only the evidence")
    citations: list[str] = Field(min_length=1, max_length=4, description="keys of the articles relied on, exactly as shown")


class State(TypedDict, total=False):
    question: str
    plan: dict
    candidates: dict[str, dict]
    selected: list[str]
    evidence: dict[str, dict]
    answer: str
    citations: list[str]
    attempts: int
    feedback: str
    problems: list[str]
    verified: bool
    approved: bool
    saved: dict | None
    status: str
    notes: list[str]


@dataclass
class Config:
    use_planner: bool = True
    use_verifier: bool = True
    human: Literal["interrupt", "auto", "off"] = "auto"  # interrupt: wait for a person; auto: approve verified answers; off: no gate or save
    max_attempts: int = 3


CITE_RE = re.compile(r"\b(AI|GDPR|DSA|DATA|NIS2|CRA):(\d+[a-z]?)\b")


def normalise_citations(raw: list[str]) -> list[str]:
    """Reduce whatever the model wrote to bare article keys. Seen in practice: the model puts the whole heading
    ('[AI:50] AI Act, Article 50: ...') in the citation list. The key inside it is unambiguous, so extract it;
    strings with no key in them are kept as they are so the verifier still rejects them."""
    out: list[str] = []
    for c in raw:
        m = CITE_RE.search(c)
        key = f"{m.group(1)}:{m.group(2)}" if m else c
        if key not in out:
            out.append(key)
    return out


def verify_answer(answer: str, citations: list[str], evidence: dict[str, dict]) -> list[str]:
    """Deterministic checks. An empty list means the answer is grounded in what was retrieved."""
    problems = []
    if not answer.strip():
        problems.append("the answer is empty")
    if len(answer.split()) > 160:
        problems.append("the answer is longer than 160 words")
    if not citations:
        problems.append("no citations")
    for c in citations:
        if not KEY_RE.match(c):
            problems.append(f"'{c}' is not a valid article key (expected e.g. GDPR:17)")
        elif c not in evidence:
            problems.append(f"'{c}' was not retrieved in this run, so it cannot be cited; cite only: {', '.join(evidence)}")
    cited = [evidence[c] for c in citations if c in evidence]
    if cited and not problems:
        words = {w for w in law.tokenize(answer) if len(w) > 3}
        best = max(len(words & set(law.tokenize(e["text"]))) for e in cited)
        if len(words) >= 5 and best < 3:
            problems.append("the answer shares almost no vocabulary with the cited articles")
    return problems


def build_graph(llm: LLM, tools: ToolBox, config: Config | None = None):
    cfg = config or Config()
    tracer_ = tracing.current

    async def plan(state: State) -> State:
        q = state["question"]
        if not cfg.use_planner:
            return {"plan": {"queries": [q], "act": None}}
        with tracer_().span("node", "plan"):
            try:
                p = await llm.json("plan", f"You help search EU digital law. Acts:\n{ACT_GUIDE}\n"
                                   "Rewrite the user's question as 1-3 short keyword search queries using the vocabulary of the law itself "
                                   "(e.g. 'erasure', 'supervisory authority', 'conformity assessment'), not whole sentences. "
                                   "Set act to one code only if the question clearly concerns that act.", q, Plan)
                return {"plan": p.model_dump()}
            except LLMOutputError as exc:
                return {"plan": {"queries": [q], "act": None}, "notes": state.get("notes", []) + [f"planner failed, used the raw question: {exc}"]}

    async def research(state: State) -> State:
        plan_ = state["plan"]
        notes = list(state.get("notes", []))
        cands: dict[str, dict] = {}
        with tracer_().span("node", "research"):
            for query in plan_["queries"]:
                for act in ([plan_["act"], None] if plan_["act"] else [None]):
                    try:
                        hits = await tools.call("search_articles", {"query": query, "act": act, "k": 5})
                    except (ToolError, ToolTransient) as exc:
                        notes.append(f"search failed for {query!r}: {exc}")
                        continue
                    for h in hits if isinstance(hits, list) else [hits]:
                        if h["key"] not in cands or h["score"] > cands[h["key"]]["score"]:
                            cands[h["key"]] = h
                    if act and len(hits) >= 3:
                        break  # the act-restricted search was enough; skip the unrestricted fallback
            if not cands:
                return {"candidates": {}, "selected": [], "evidence": {}, "notes": notes + ["no candidates found"]}
            ranked = sorted(cands.values(), key=lambda h: -h["score"])[:8]
            listing = "\n".join(f"{h['key']} | {h['heading'].split(', ', 1)[1]} | {h['snippet'][:110]}" for h in ranked)
            try:
                sel = await llm.json("select", "Choose the 1-3 articles that most directly answer the question. Use only keys from the list, exactly as written.",
                                     f"Question: {state['question']}\n\nCandidates:\n{listing}", Selection)
                chosen = [k for k in sel.keys if k in cands]
                if len(chosen) != len(sel.keys):
                    notes.append(f"selector proposed keys that were not candidates: {[k for k in sel.keys if k not in cands]}")
            except LLMOutputError as exc:
                chosen, notes = [], notes + [f"selector failed: {exc}"]
            if not chosen:
                chosen = [h["key"] for h in ranked[:2]]
                notes.append("fell back to the top-ranked search results")
            evidence = {}
            for key in dict.fromkeys(chosen):
                act, num = key.split(":")
                try:
                    art = await tools.call("get_article", {"act": act, "number": num})
                    evidence[key] = {"heading": art["heading"], "text": art["text"][:EVIDENCE_CHARS]}
                except (ToolError, ToolTransient) as exc:
                    notes.append(f"could not read {key}: {exc}")
            return {"candidates": cands, "selected": list(evidence), "evidence": evidence, "notes": notes}

    async def draft(state: State) -> State:
        ev = state.get("evidence", {})
        attempts = state.get("attempts", 0) + 1
        with tracer_().span("node", "draft", attempt=attempts):
            if not ev:
                return {"answer": "", "citations": [], "attempts": attempts, "problems": ["no evidence was retrieved"]}
            blocks = "\n\n".join(f"[{k}] {e['heading']}\n{e['text']}" for k, e in ev.items())
            user = f"Question: {state['question']}\n\nEvidence:\n{blocks}"
            if state.get("feedback"):
                user += f"\n\nYour previous answer was rejected: {state['feedback']}\nWrite a corrected answer."
            try:
                a = await llm.json("draft", "Answer the question in at most 100 words using ONLY the evidence. "
                                   "List in citations the keys (exactly as shown in square brackets) of the articles you used.", user, Answer)
                return {"answer": a.answer, "citations": normalise_citations(a.citations), "attempts": attempts}
            except LLMOutputError as exc:
                return {"answer": "", "citations": [], "attempts": attempts, "problems": [f"drafting failed: {exc}"]}

    async def verify(state: State) -> State:
        with tracer_().span("node", "verify") as span:
            problems = verify_answer(state.get("answer", ""), state.get("citations", []), state.get("evidence", {})) if cfg.use_verifier else []
            if state.get("problems") and not state.get("answer"):
                problems = state["problems"]
            span["problems"] = problems
            return {"problems": problems, "verified": not problems, "feedback": "; ".join(problems)}

    def after_verify(state: State) -> str:
        if state["problems"] and state.get("attempts", 0) < cfg.max_attempts and cfg.use_verifier and state.get("evidence"):
            return "draft"
        return "human" if cfg.human != "off" else "finish"

    async def human(state: State) -> State:
        if cfg.human == "auto":
            ok = bool(state.get("verified") and state.get("answer"))
            tracer_().event("human_auto", approved=ok)
            return {"approved": ok, "feedback": "" if ok else "auto-reviewer: answer failed verification"}
        decision = interrupt({"question": state["question"], "answer": state.get("answer"), "citations": state.get("citations"),
                              "problems": state.get("problems"), "action": "save this answer as a memo?"})
        tracer_().event("human_decision", **decision)
        return {"approved": bool(decision.get("approved")), "feedback": decision.get("feedback", "")}

    def after_human(state: State) -> str:
        if state["approved"]:
            return "save"
        if state.get("feedback") and state.get("attempts", 0) < cfg.max_attempts and cfg.human == "interrupt":
            return "draft"
        return "finish"

    async def save(state: State) -> State:
        with tracer_().span("node", "save"):
            body = f"**Question:** {state['question']}\n\n{state['answer']}\n\n**Sources:** {', '.join(state['citations'])}"
            try:
                res = await tools.call("save_memo", {"title": state["question"][:80], "body": body})
                return {"saved": res}
            except (ToolError, ToolTransient) as exc:
                return {"saved": None, "notes": state.get("notes", []) + [f"memo not saved: {exc}"]}

    async def finish(state: State) -> State:
        if state.get("saved"):
            status = "saved"
        elif not state.get("answer"):
            status = "no_answer"
        elif state.get("approved") is False and cfg.human != "off":
            status = "rejected"
        else:
            status = "answered" if state.get("verified") or not cfg.use_verifier else "unverified"
        return {"status": status}

    g = StateGraph(State)
    for name, fn in (("plan", plan), ("research", research), ("draft", draft), ("verify", verify), ("human", human), ("save", save), ("finish", finish)):
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_edge("plan", "research")
    g.add_edge("research", "draft")
    g.add_edge("draft", "verify")
    g.add_conditional_edges("verify", after_verify, {"draft": "draft", "human": "human", "finish": "finish"})
    g.add_conditional_edges("human", after_human, {"draft": "draft", "save": "save", "finish": "finish"})
    g.add_edge("save", "finish")
    g.add_edge("finish", END)
    return g.compile(checkpointer=MemorySaver())


async def run(graph, question: str, thread: str, resume: Any | None = None) -> dict:
    """Run to completion or to the human-approval interrupt. Returns the final state, plus `__interrupt__` if paused."""
    cfg = {"configurable": {"thread_id": thread}, "recursion_limit": 40}
    inp: Any = Command(resume=resume) if resume is not None else {"question": question, "attempts": 0, "notes": []}
    return await graph.ainvoke(inp, cfg)
