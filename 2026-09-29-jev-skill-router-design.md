# Jev as a Zero-Shot Skill/Tool Router — Design Spec

## Motivation

The Jev-Cascade research (see `research.md`) tested Jev as a correctness oracle (failed — "fluency trap": confidence tracks fluency, not correctness) and then as a grounded-familiarity judge over a worked-example library, on two math domains chosen to bracket the hypothesis from opposite sides:

- **AIME** (adversarial, technique-diverse): familiarity almost never fired (2.7% match rate against a 15-example, all-tier, technique-diverse library) — a **floor effect**. Open-ended retrieval against a sparse sample of an effectively infinite problem space.
- **GSM8K harder slice** (templated, homogeneous): familiarity almost always fired (0.78 average) — a **ceiling effect**. Same retrieval shape, but the sample space is small and homogeneous, so everything looks similar.

Both failures share a root cause: asking Jev to judge similarity against an **open-ended space** (arbitrary math problems) using a **sparse, arbitrary sample** (a handful of worked examples) of that space. The judge itself was proven sensitive in both cases (a reworded positive control scored 0.99 both times), so the failure is about problem *shape*, not instrument quality.

**New hypothesis under test:** skill/tool selection inside an agent's tool-use loop is a structurally different problem — a **bounded, closed-set classification** (a fixed catalogue of tools, each with a human-engineered discriminating description) rather than open-ended retrieval against an unbounded space. This spec tests whether that structural difference lets Jev avoid the floor/ceiling collapse, using a minimal from-scratch agent harness as the testbed.

**Explicit scope decision:** this spec covers **zero-shot routing only** — Jev classifies relevance from the tool's trigger description alone, with no memory of past routing decisions and no learning over time. The "does routing accuracy improve with accumulated evidence" question (the graduation mechanism from the original System-1/System-2 thesis) is an explicit follow-up, only worth pursuing if zero-shot shows real signal here.

## Architecture

A minimal ReAct-style loop with Jev inserted as a pre-filter before every LLM call in the loop, not inside tool execution:

```
user query / current turn context
        │
        ▼
  Jev skill-router (single call, one Noul score per tool)
        │
        ▼
  filtered tool set (high scorers)  ── OR ──  full catalogue (escalation, if scores ambiguous)
        │
        ▼
  main LLM call (sees ONLY the filtered tool descriptions + conversation history)
        │
        ▼
  tool call? ──yes──► execute tool ──► append observation ──► loop back to Jev router
        │no
        ▼
  final answer
```

Jev re-runs the filter on **every loop iteration**, not once per user turn, since tool relevance can shift mid-task (e.g. after reading a file, the next relevant tool differs from the first). This also means the boundary-case stress test recurs across a multi-step task, not just once at the start.

## Components

- **`toy_tools.py`** — a small, fixed catalogue (~6-8 tools: `calculator`, `read_file`, `write_file`, `web_search_stub`, `note_taker`, `list_files`, plus 1-2 more). Each tool has:
  - `name`
  - `trigger_description` — a human-written description styled like real skill descriptions ("use when X, Y, Z")
  - `execute(**kwargs)` — a stub implementation (no real file I/O or network needed for the routing experiment; can log calls and return canned results)
  - The catalogue includes **2-3 deliberately overlapping pairs** (e.g. `note_taker` vs `write_file` — both plausibly handle "save this for later") — these are intentional boundary cases, not incidental noise, since they're the actual thing this spec needs to measure.

- **`jev_router.py`**
  - `score_tools(query: str, history: list, catalogue: list[Tool]) -> dict[str, float]` — one Jev `system_one` call, one `Noul` sub-question per tool, each instructed with that tool's `trigger_description`. Returns a relevance score per tool name.
  - `select_tools(scores: dict[str, float], threshold: float, ambiguity_margin: float) -> list[str] | Literal["ESCALATE"]` — returns tool names scoring above `threshold`; returns the escalation sentinel if the top-2 scores are within `ambiguity_margin` of each other (too close to trust a clean cut) rather than guessing.

- **`agent_loop.py`** — the ReAct loop itself: builds the LLM prompt from the filtered tool descriptions (or full catalogue on escalation) plus conversation history, parses either a tool call or a final answer, executes via `toy_tools`, appends the result as an observation, loops back to the router.

- **`main.py`** — wires the pieces together for a single task run (CLI entry point).

- **`cache_utils.py`** (reused from the math-cascade work) — Jev routing scores get cached, keyed by `(query, catalogue_signature)`, so re-running the validation analysis after the first pass costs nothing.

## Validation harness

This is the actual experiment, not incidental test coverage.

**Labeled test set:** hand-authored, ~30-40 `(query, correct_tool_set)` pairs, split by design into two buckets — mirroring the tier/tag breakdowns used in the math-cascade work:
- **Obvious cases** — exactly one tool is unambiguously correct (e.g. "what's 47 * 83" → `calculator`).
- **Boundary cases** — deliberately targets the overlapping-tool pairs, or requires zero tools (a query needing no tool at all), or requires two tools together.

**Metrics** (computed via an offline, cache-only re-analysis script, same pattern as `analyze_familiarity_threshold.py`):
- Precision/recall of Jev's selected tool set vs. ground truth, **split by obvious vs. boundary** — this is the direct test of whether boundary-case accuracy flattens to noise the way the within-tier math gaps did.
- Escalation rate — how often ambiguity triggers the full-catalogue fallback, and whether escalations concentrate in the boundary bucket (they should, if the mechanism is working as intended) or are scattered randomly (which would mean the ambiguity signal itself isn't tracking real ambiguity).
- A positive-control-equivalent: at least one test query that's a near-verbatim restatement of a tool's trigger description, to confirm the judge is sensitive at all before trusting any null result (same discipline as the math work's reworded-duplicate control).

## Error handling

- Jev API failure or malformed response → fail-safe to the **full catalogue**, never to zero tools. A missing tool is a correctness bug for the agent, not a graceful degradation, so failure must never silently narrow capability.
- Ambiguous top-2 scores (within `ambiguity_margin`) → escalate to full catalogue rather than pick the higher-scoring guess.

## Testing plan

1. Unit-level: `toy_tools.py` stubs are deterministic and independently testable.
2. Routing accuracy: run `jev_router.score_tools` over the full labeled test set (cached), then the offline analysis script computes precision/recall split by obvious/boundary and the escalation-rate breakdown described above.
3. End-to-end: a few multi-step `agent_loop.py` runs (e.g. "read this file, then calculate the sum of these numbers, then save a note") to confirm the router re-filters correctly across loop iterations, not just at task start.

## Explicit non-goals (this spec)

- No learning/memory across routing decisions — every call is independent, zero-shot from the trigger description alone. (Follow-up spec if this shows signal.)
- No embedding-based retrieval or hierarchical categorization (approaches B/C considered and deferred — only worth building if the toy catalogue needs to scale past what a single batched Jev call handles cleanly, or if boundary-case accuracy is poor for reasons an embedding pre-filter could plausibly fix).
- No real tool side effects (file I/O, network calls) — `execute()` stubs are sufficient since this spec is testing the *routing* decision, not the tools' functionality.
