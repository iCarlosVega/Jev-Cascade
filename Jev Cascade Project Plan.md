# Jev Cascade: Project Plan

Sep 28, 2026 · @Carlos Vega

## Overview

The goal is to test whether Jev, a System One model, can act as a "feeling of rightness" judge: fast models answer most queries cheaply, and a frontier model is called only when Jev's confidence is low.

**Hypothesis:** a cascade of fast models judged by Jev can come close to Opus-level accuracy at a fraction of Opus's cost and latency, and beat simple majority voting as an escalation signal.

**Cognitive science mapping:**

| System component | Human analog | Source |
| --- | --- | --- |
| n fast responses | Competing System 1 intuitions | De Neys |
| Jev's confidence | Feeling of Rightness | Thompson |
| Escalation to a frontier model | System 2 intervening | Kahneman; Evans & Stanovich |
| Threshold tuned by stakes | Expected Value of Control | Shenhav et al. |

The entire system is built by hand. This doc describes what each piece does and how to know it works, never how to implement it.

## Architecture

The system is three steps; the threshold is a single check in code, not its own layer.

&#91;embedded content: cascade flow · 3 steps, 1 decision\]

Jev returns its confidence in the same call where it picks the best candidate, so no second Jev call is needed.

Optional additions, once the basic version works:

- A Jev stakes classification that tightens the threshold for high-stakes queries.
- Passing the rejected candidates to Opus as context when escalating.
- Agreement across the n candidates as a second escalation trigger.

## Build checklist

Each milestone lists what the piece must do and how you'll know it works. Implementation is up to you.

**1. Access and setup**

- [ ] Get Jev early access (the waitlist, or Vercel AI Gateway).
- [ ] Get API access to the fast model and Opus.
- [ ] Done when: one test call to each of the three returns a response.

**2. Dataset loader**

- [ ] Loads about 100 GSM8K problems with their ground-truth answers.
- [ ] Extracts a comparable final answer from any model's response.
- [ ] Done when: the ground-truth answer checked against itself scores 100%.

**3. Fast-model stage**

- [ ] Sends one query to the fast model n times in parallel and collects all n responses.
- [ ] Done when: n responses come back for a query, and running in parallel is faster than running sequentially.

**4. Jev judge stage**

- [ ] Sends the query plus the n candidates to Jev and gets back one pick and one confidence value.
- [ ] Done when: on a hand-made test where one candidate is obviously right, Jev picks it with high confidence.

**5. Escalation logic**

- [ ] Compares confidence to a configurable threshold and either returns the pick or calls Opus.
- [ ] Done when: a threshold of 0 never escalates and a threshold of 1 always does.

**6. Logging**

- [ ] Records for every query: each call's output, Jev's pick and confidence, whether it escalated, cost, latency, and correctness.
- [ ] Done when: one run produces a log you can reopen and analyze without rerunning anything.

**7. Baselines**

- [ ] Fast model only, Opus only, and majority vote with escalation on disagreement, all using the same dataset and log format.
- [ ] Done when: all four setups produce comparable logs.

**8. Threshold sweep**

- [ ] Runs the cascade at 0.6, 0.7, 0.8, and 0.9.
- [ ] Done when: each threshold has its own accuracy, cost, latency, and escalation rate.

**9. Analysis**

- [ ] Accuracy-vs-cost plot with every setup and threshold as a point.
- [ ] Calibration plot: Jev's stated confidence vs. how often it was actually right.
- [ ] Done when: you can say whether the cascade beats majority vote and whether Jev is calibrated.

## Benchmark plan

Four setups are compared on accuracy, cost, and latency; accuracy is what gives the other two meaning.

| Setup | What it tests |
| --- | --- |
| Fast model only | The cheap floor: lowest cost, lowest expected accuracy |
| Opus only | The ceiling: highest accuracy, highest cost |
| Majority vote + escalate on disagreement | Whether a simple agreement signal already does the job |
| Jev cascade at 0.6 / 0.7 / 0.8 / 0.9 | Whether Jev adds value as a judge |

**Metrics:**

- **Accuracy:** correctness against ground truth.
- **Cost:** total dollars per query across all calls.
- **Latency:** end-to-end time per query.
- **Escalation rate:** share of queries sent to Opus; this explains the cost results.
- **Calibration:** Jev's stated confidence vs. actual accuracy, as a reliability diagram.

**Datasets:** start with tasks that have checkable answers, GSM8K (math) and a coding set with unit tests (HumanEval or MBPP). Test open-ended tasks later to find where the approach breaks.

**Success criteria:**

- At some threshold, the cascade reaches roughly Opus-level accuracy at much lower cost.
- The cascade beats majority vote on the accuracy-vs-cost tradeoff. If not, Jev isn't earning its place.
- Jev's confidence is reasonably calibrated.

A negative result still counts as a finding worth writing up.

## Known risks

| Risk | What happens | What to watch |
| --- | --- | --- |
| Fluency trap | Jev favors answers that sound right over answers that are right | High-confidence wrong answers in the log — **confirmed 2026-09-28**: on 100 harder GSM8K problems (rows 300-399), 6 of 8 wrong picks scored 0.92-0.99 confidence, indistinguishable from the 0.97 average confidence on correct picks. Per-candidate `Noul` scoring (judging each candidate independently instead of a comparative `Choice`) fixed a separate wording-diversity confound but did not create confidence separation between right and wrong answers. |
| Correlated errors | Fast models agree on the same wrong answer, so nothing escalates | Wrong answers where all n candidates matched |
| Escalation overhead | Escalated queries pay for every stage | Escalation rate; cost per escalated query |
| Unproven calibration | Jev's confidence may not track correctness on this data | The calibration plot |
| Task dependence | Strong on checkable tasks, weaker on open-ended reasoning | Results when moving beyond GSM8K |

## Decisions log

Record each choice and why; this becomes the backbone of the write-up.

| Decision | Choice | Why |
| --- | --- | --- |
| Fast model | DeepSeek V4.1 Flash (tentative) |  |
| Escalation model | Opus 5.5 (tentative) |  |
| Value of n |  | Start with 3–5 |
| First dataset | GSM8K, about 100 problems | Checkable answers |
| Thresholds tested | 0.6, 0.7, 0.8, 0.9 |  |

## Experiment log

One row per run, newest first.

| Date | Setup | Threshold | Accuracy (%) | Cost ($/query) | Latency (s) | Escalation rate (%) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-28 | Fast model (deepseek-v4-pro, no reasoning, low effort, n=3) + Jev judge (per-candidate `Noul` "does this make sense") | N/A (no escalation logic run) | 92 (92/100) | N/A | N/A | N/A | Calibration probe, not a cascade run. 100 harder GSM8K problems (test.jsonl rows 300-399). Confidence when right avg 0.97; confidence when wrong avg 0.94 — the two distributions barely separate (6/8 wrong picks still scored 0.92-0.99). Confirms the fluency-trap risk: current confidence signal is not yet reliable enough to gate escalation. |

## References

The papers below are cited from memory; verify titles and details before relying on them. The Jev details are from [TypeSafe AI's launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

**Cognitive science**

- Thompson, Prowse Turner & Pennycook (2011): Feeling of Rightness
- Ackerman & Thompson (2017), *TICS*: Meta-Reasoning
- De Neys (2023), *BBS*: Advancing theorizing about fast-and-slow thinking
- Shenhav, Botvinick & Cohen (2013), *Neuron*: Expected Value of Control
- Daw, Niv & Dayan (2005): uncertainty-based arbitration between systems
- Lieder & Griffiths (2017): strategy selection as rational metareasoning

**AI systems**

- Booch, Rossi et al. (2021), AAAI: SOFAI, "Thinking Fast and Slow in AI"
- Chen, Zaharia & Zou (2023): FrugalGPT
- Madaan et al. (2023): AutoMix
- Jiang et al. (2023): LLM-Blender
- Wang et al. (2022): Self-consistency
- Brown et al. (2024): Large Language Monkeys
- Zaharia et al. (2024), BAIR blog: Compound AI Systems
