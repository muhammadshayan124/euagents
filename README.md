# euagents

A multi-agent research assistant over EU digital law (AI Act, GDPR, Digital Services Act, Data Act, NIS2, Cyber Resilience Act), built to find out
**what actually breaks when you run an agent system on a small local model**, and to make each failure measurable.

It uses LangGraph for orchestration, two MCP tool servers, a human approval gate before the only side effect, bounded retries, JSONL tracing, and a
benchmark that classifies every failure into exactly one cause. The model is **Qwen3-1.7B (1.4 GB) on a CPU through Ollama**, which is small enough that
failures are frequent and informative rather than rare.

> **Scope, stated plainly.** This is an engineering and measurement project, not a legal advice tool, and the answers are not reviewed legal analysis.
> The benchmark counts a run as a success if the answer **cites the article that answers the question**. That is objective and cheap, and it is the first thing a
> legal-research assistant must get right, but it does not judge the quality of the prose.

## Architecture

```
question
   |
   v
[plan]      LLM -> {1-3 keyword queries, optional act code}          schema-constrained; falls back to the raw question
   |
   v
[research]  MCP lawsearch.search_articles x queries -> candidates     tool errors are recorded, not fatal
            LLM picks <=3 candidate keys (must be keys it was shown)  falls back to the top-ranked results
            MCP lawsearch.get_article for each pick -> evidence
   |
   v
[draft]     LLM -> {answer <= 100 words, citations}                   citations normalised to bare ACT:N keys
   |
   v
[verify]    deterministic code, no model:                             every citation must be an article retrieved IN THIS RUN,
            |                                                          and the answer must share vocabulary with the cited text
            +-- problems and attempts left --> back to [draft] with the problems as feedback (max 3 attempts)
   |
   v
[human]     LangGraph interrupt: shows the answer, citations and verifier verdict; resumes with approve / reject+feedback
            +-- rejected with feedback --> back to [draft]
   |
   v
[save]      MCP memo.save_memo   (the ONLY tool with a side effect; unreachable without approval)
```

**MCP servers** (stdio, official Python SDK): `lawsearch` (`list_acts`, `search_articles` BM25, `get_article`) is read-only; `memo` (`save_memo`, `list_memos`)
refuses to overwrite and slugifies titles so a hostile title cannot write outside the memo directory (tested).

**Retries and errors.** The tool layer separates `ToolTransient` (the call did not complete: timeout, broken pipe: retried with exponential backoff and jitter) from `ToolError`
(the tool ran and said no, e.g. unknown article: *not* retried, because repeating a bad call cannot help). A fault injector exercises the retry path on purpose in the tests.
Structured LLM outputs use schema-constrained decoding, with a bounded re-ask that feeds the validation error back to the model.

**Tracing.** Every LLM call, tool call and graph node is one JSON line (inputs/outputs summarised, latency, token counts, error). The failure analysis below is computed
*from these traces and results*, not from the console.

## Method

Tasks are 30 of the 67 hand-labelled questions from the companion `eurag` benchmark (a fixed, evenly spaced sample, identical for every configuration), each with one gold
article whose text was mechanically checked to contain the labelled evidence phrase. Every configuration runs the same 30 tasks sequentially on one CPU (Ollama capped at 3 threads),
temperature 0, thinking mode off. **Failures get exactly one cause**, assigned in pipeline order (where the gold article was first lost): retrieval miss, selection miss, wrong citation,
ungrounded citation (cited an article never retrieved), or no answer.

## Results

<!-- main:start -->
| configuration | tasks | success (95% CI) | LLM calls / task | tokens / task | seconds / task |
|---|---|---|---|---|---|
| Graph: planner + verifier + approval gate | 30 | 57% (39%-73%) | 3.1 | 1,609 | 75 |
| Graph without the planner (raw question as the query) | 30 | 47% (30%-64%) | 2.2 | 1,509 | 71 |
| Graph without the verifier | 30 | 57% (39%-73%) | 3.1 | 1,613 | 42 |
| Single agent, free tool use (permissive prompt) | 30 | 3% (1%-17%) | 1.0 | 507 | 18 |
| Single agent, directive prompt (must search, then read) | 30 | 20% (10%-37%) | 2.8 | 2,215 | 47 |
<!-- main:end -->

Where the failures come from (counts of the 30 tasks per configuration):

<!-- causes:start -->
| configuration | success | retrieval miss | selection miss | wrong citation | ungrounded citation | no answer |
|---|---|---|---|---|---|---|
| Graph: planner + verifier + approval gate | 17 | 11 | 2 | 0 | 0 | 0 |  (of 30)
| Graph without the planner (raw question as the query) | 14 | 16 | 0 | 0 | 0 | 0 |  (of 30)
| Graph without the verifier | 17 | 11 | 2 | 0 | 0 | 0 |  (of 30)
| Single agent, free tool use (permissive prompt) | 1 | 0 | 0 | 0 | 21 | 8 |  (of 30)
| Single agent, directive prompt (must search, then read) | 6 | 3 | 0 | 0 | 11 | 10 |  (of 30)

What each cause means: **retrieval miss**: the search never surfaced the gold article; **selection miss**: the gold article was a candidate but the model did not choose to read it; **wrong citation**: the model read the gold article and still cited a different one; **ungrounded citation**: the model cited an article it never retrieved (invented or remembered); **no answer**: no usable answer (step limit, timeout, error, or no citation).
<!-- causes:end -->

Paired comparisons on the same tasks (exact McNemar test; with 30 tasks only large differences are detectable):

<!-- paired:start -->
| comparison (same tasks) | A only succeeds | B only succeeds | exact McNemar p |
|---|---|---|---|
| full vs no_planner (n=30) | 6 | 3 | 0.508 |
| full vs no_verifier (n=30) | 0 | 0 | 1.000 |
| full vs single_agent (n=30) | 16 | 0 | 0.000 |
| full vs single_agent_directive (n=30) | 12 | 1 | 0.003 |
| single_agent_directive vs single_agent (n=30) | 5 | 0 | 0.062 |
<!-- paired:end -->

### What the pipeline did

<!-- behaviour:start -->
- **Verifier.** 0 of 30 runs needed a second draft (0 of those succeeded); 0 ended without passing verification (those are never saved).
- **Fallbacks.** planner failed 0x, selector proposed non-candidate keys 7x, selector failed 0x, fell back to top search results 4x; schema failures 2; tool errors 0.
- **Free tool use.** In 30 of 30 permissive-prompt runs the model made no tool call at all and answered from memory.
- **Directive prompt.** 0 of 30 runs still made no tool call; mean tool calls per run 1.8.
<!-- behaviour:end -->

### Example failures

<!-- examples:start -->
- **retrieval miss** (full, ai-01): "Is it forbidden to use AI that detects the emotional state of employees at work?" gold ['AI:5'], cited ['AI:10', 'AI:59', 'AI:3']
- **retrieval miss** (full, gdpr-05): "Can a person take back the permission they gave for their data to be used?" gold ['GDPR:7'], cited ['GDPR:17', 'GDPR:15', 'GDPR:18']
- **selection miss** (full, ai-09): "How large can the penalty be for deploying a banned AI practice?" gold ['AI:99'], cited ['AI:101', 'DSA:52', 'DSA:78']
- **selection miss** (full, ai-14): "Which procedure proves a high-risk system meets the rules before it can be sold?" gold ['AI:43'], cited ['AI:30']
- **ungrounded citation** (single_agent_directive, ai-01): "Is it forbidden to use AI that detects the emotional state of employees at work?" gold ['AI:5'], cited ['AI:10', 'AI:3', 'AI:50']
- **ungrounded citation** (single_agent_directive, ai-03): "Do people have to be told that they are talking to a chatbot rather than a human?" gold ['AI:50'], cited ['GDPR:22', 'GDPR:45', 'GDPR:52']
<!-- examples:end -->

## What the measurements say

1. **Structure beat freedom by a wide margin at this model size.** The graph succeeds on 57% of tasks; the same model as a free single agent succeeds on 3% with a permissive
   prompt and 20% with a directive one (paired exact tests, p < 0.001 and p = 0.003). Both gaps are well outside the noise of 30 tasks.
2. **Free tool use fails at the very first step.** With the permissive prompt the model made **no tool call in any of 30 runs**: it answered from memory and cited articles it never
   retrieved (21 of its 29 failures are *ungrounded citations*). A prompt that orders "search, then read, then answer" fixed the tool use (every run called tools, 1.8 calls on average)
   but not the reliability: 11 ungrounded citations and 10 runs with no usable answer remain. The improvement from the better prompt (3% to 20%) is suggestive, not resolved (p = 0.06).
3. **The graph's failures are almost all retrieval.** 11 of its 13 failures are *retrieval misses*: the BM25 search never surfaced the gold article, so nothing downstream could succeed.
   The search tool, not the agent, bounds performance here. The planner (which rewrites the question into legal-vocabulary keyword queries) helps in the expected direction
   (57% vs 47%, retrieval misses 16 to 11), but with 30 tasks that is **not statistically resolved** (p = 0.51).
4. **The verifier never fired, and removing it changed nothing.** In all 30 runs the first draft already passed (0 redrafts), and the no-verifier configuration produced identical outcomes
   (17 of 30 in both). The reason is structural: the drafter only ever sees evidence that was retrieved, so an invented citation cannot reach it. The verifier is insurance for failure modes
   this pipeline already designs out, and this benchmark cannot measure its value. What the data does show is that the same grounding rule applied to the permissive single agent would have
   flagged those 21 invented-citation answers.
5. **Smaller things the traces caught.** The selector was told to choose only from the candidates it was shown and still proposed a key it had not been shown in 7 of 30 runs (the pipeline
   rejected those and fell back to the top search results in 4); schema-constrained decoding removed malformed JSON (2 schema failures in 30 runs, both recovered by the re-ask).
6. **A format failure found in the first smoke test and fixed.** The model sometimes wrote a whole heading (`[AI:50] AI Act, Article 50: ...`) into the citation list, which the verifier read as
   an unknown key. Citations are now normalised to the bare `ACT:N` key inside them (tested); strings with no key in them still fail verification.

**What is not shown.** The seconds-per-task column is not comparable across rows: the graph run overlapped with my own test runs on the same CPU, which is why it looks slower than the
no-verifier run despite near-identical token counts (1,609 vs 1,613). The human-approval gate and the retry-with-backoff path are covered by tests with real MCP servers and a scripted
model, but they are not exercised by this benchmark (it auto-approves verified answers, and no injected faults were used).


## Run it

```bash
pip install -e ".[dev]"
ollama pull qwen3:1.7b                      # 1.4 GB
euagents fetch                              # six acts from the EU Publications Office
pytest                                      # 27 tests; the model is replaced by a scripted stand-in, MCP servers are real
euagents ask "When can I demand deletion of my data?"      # interactive: you approve or reject the answer before it is saved
euagents eval full --n 30                   # resumable; also: no_planner, no_verifier, single_agent, single_agent_directive
euagents report && euagents readme          # tables, then fill this README from the results
```

## Limitations

- **30 tasks.** Intervals are wide; differences smaller than about 25 points are not resolved. The questions were written by one author.
- **One small model, one seed (temperature 0).** Findings describe a 1.7B model; larger models fail differently and less often.
- **A citation test, not an answer-quality judgement.** A run can cite the right article and still write a poor summary.
- **Lexical search (BM25).** Retrieval misses are partly a property of the search tool, not of the agent; they bound what any agent on top of it can do.
- **Human approval is exercised in tests and in `euagents ask`, but the benchmark auto-approves verified answers**, so it measures the agent, not a reviewer.
- **CPU latency** (about a minute per task) is a property of this hardware, not of the design.

MIT licence. Source text: European Union legislation (EUR-Lex), reusable under Commission Decision 2011/833/EU.
