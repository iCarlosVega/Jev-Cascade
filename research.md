# Jev-Cascade Research Log

Trajectory of findings, in chronological order, with the actual numbers behind each conclusion. This documents *why* the architecture changed shape twice, not just what the current code does.

## 1. Original architecture: Jev as a correctness oracle

**Hypothesis:** Jev estimates `P(correct | query, answer)` — given a candidate answer, judge whether it's right, and escalate to Opus on low confidence.

**Early result (20 harder GSM8K problems, rows unspecified early subset):**
- Accuracy: 18/20 (90%)
- Avg confidence when correct: 0.97
- Avg confidence when wrong: 0.79
- Gap: 0.18 — looked healthy at this sample size.

**Result that broke it (100 harder GSM8K problems, rows 300-399):**
- Accuracy: 92/100 (92%)
- Avg confidence when correct: 0.97
- Avg confidence when wrong: **0.94**
- Gap: **0.03** — confidence and correctness became statistically indistinguishable.
- 6 of 8 wrong picks scored 0.92-0.99 confidence — high confidence on wrong answers, not just occasionally but as the norm.
- Per-candidate `Noul` scoring fixed a wording-diversity confound but did not fix calibration.

**Conclusion — the "fluency trap":** Jev's confidence tracks how *fluent/plausible* an answer sounds, not whether it's *correct*. Two answers can sit close together in representation space while one is right and one is wrong — semantic separability ≠ correctness separability. This killed the correctness-oracle framing.

## 2. Architectural pivot: Jev as a competence/familiarity router

New hypothesis: instead of "is this answer correct," ask "have I encountered a situation structurally like this before, and do I have a known competence for it." Reframes Jev from a correctness evaluator into a familiarity/competence classifier. This is the framing all work below tests.

## 3. Precondition test (M1): does structure even predict fast-model accuracy?

Before building any router, check whether "problem type" predicts the fast model's success at all — if not, there's nothing to route around.

**v1 — GSM8K, keyword-heuristic tags (n=60, then rerun n=100 on harder slice, `deepseek-v4-pro`):**
```
percentage             n= 11  acc= 90.9%
rate_per_unit          n= 22  acc= 95.5%
ratio_multiplier       n=  9  acc=100.0%
remaining_after_spend  n=  6  acc= 83.3%
sum_total              n=  1  acc=100.0%
unclassified           n= 11  acc= 90.9%
```
Flat, near-ceiling, inconclusive. Two suspected causes: (a) small/uneven tag buckets, (b) GSM8K may be saturated for this model.

**Confound identified:** `deepseek-v4-pro` might be cost-cheap but not capability-cheap — i.e. actually a strong/"expensive-tier" model mislabeled as the fast tier, in which case no benchmark shows a routing-worthy gap.

**v2 — MATH-500 (Hendrycks, 500 problems w/ ground-truth `subject`+`level` 1-5; filtered to 316 numeric-answer problems since ~37% of answers are LaTeX/symbolic and would need a separate grader), n=50, still `deepseek-v4-pro`:**
```
Accuracy by subject: 83.3%-100% across all 7 subjects
Accuracy by level:   85.7% (L1) to 100% (L5)
```
Still ceiling — confirms the model-choice confound, not a dataset problem. Swapped fast model to `deepseek-flash` (v4.1, non-reasoning tier). Reran MATH-500: still ~100%. MATH-500 is too easy for either tier.

**v3 — AIME (2022-2024 pooled, 90 problems, integer answers 0-999, via `AI-MO/aimo-validation-aime`), `deepseek-flash`:**
- Baseline accuracy: 70/90 (77.8%), rerun 72/90 (80.0%) — first real, non-ceiling gap. (Note: the ~2pt swing between identical reruns is model output nondeterminism — no temperature/seed pinned — flagged as a confound in its own right.)
- Bucketed by contest problem number (1-15 per contest, which AIME orders roughly by increasing difficulty — a free ground-truth proxy, not a guessed heuristic):

```
1-5 (easy)      n=30  acc=96.7%
6-10 (mid)      n=30  acc=86.7%
11-15 (hard)    n=30  acc=56.7%
```

**Conclusion — M1 confirmed:** clean monotonic accuracy drop across known-difficulty tiers. A real, structure-correlated capability gap exists. This is the first solid precondition pass and the dataset/model combo used from here on.

## 4. M2 v1: ungrounded familiarity (failed, informatively)

Asked Jev, given *only* the raw problem text (no memory, no library, no state), "does this look like a familiar/well-practiced problem type?"

```
Avg familiarity by tier:
1-5 (easy)      0.77
6-10 (mid)      0.61
11-15 (hard)    0.52

Overall: familiarity when RIGHT = 0.67 (n=69), when WRONG = 0.51 (n=21), gap = 0.16
```

Looked promising until checked **within tier** (controlling for the fact that hard tier has both lower familiarity and lower accuracy):

```
1-5 (easy)      right avg=0.78 (n=29)  wrong avg=0.70 (n= 1)  gap=0.08
6-10 (mid)      right avg=0.63 (n=23)  wrong avg=0.58 (n= 7)  gap=0.06
11-15 (hard)    right avg=0.54 (n=19)  wrong avg=0.49 (n=11)  gap=0.05
```

Gap collapsed from 0.16 to ~0.05-0.08 once tier was controlled for. **Conclusion:** the earlier gap was mostly just "Jev knows tier 11-15 is hard," not per-problem discrimination. This is the same disease as the original fluency trap, one level up: coarse-category separability without fine-grained (per-instance) separability.

**Deeper critique (user-identified):** this test is conceptually incoherent on its own terms. The thesis is "have *I* seen this before" — but there is no actual solved-problem history anywhere in the system. Asking Jev to introspect on familiarity with no library grounds the answer in generic pretrained difficulty-perception, not genuine competence recall. The test measured the wrong thing.

## 5. M2 v2: grounded familiarity (real library, real confound found)

Built an actual "competence library": 6 problems the fast model had genuinely solved correctly (3 from tier 1-5, 3 from tier 6-10, verified against ground truth — not guessed). Held out the other 84. Asked Jev to compare each held-out problem against this concrete library ("does this match one of these known solved patterns"), instead of asking an abstract vibe-check.

Added a caching layer (`cache_utils.py`) at this point — fast-model and Jev results are now stored in `results/*.jsonl`, keyed by question text, so reruns and new analyses don't re-hit the API or introduce fresh model-noise into comparisons.

**Result:**
```
1-5 (easy)      right avg=0.29 (n=25)  wrong avg=0.29 (n= 2)  gap=-0.00
6-10 (mid)      right avg=0.23 (n=19)  wrong avg=0.24 (n= 8)  gap=-0.01
11-15 (hard)    right avg=0.23 (n=11)  wrong avg=0.24 (n=19)  gap=-0.01
```
Flat at zero, and notably *lower* in absolute terms than the ungrounded version (0.23-0.29 vs 0.52-0.77). Also surfaced the model-nondeterminism confound directly: this run's hard-tier split was 11 right / 19 wrong (36.7% acc) vs the earlier 19/11 (63.3%) — same 30 problems, opposite majority outcome, underscoring that "ground truth" itself isn't stable run-to-run without a pinned temperature/seed.

**Sanity check — is the judge even sensitive?** Took one library problem, reworded it (same math, different phrasing) via the fast model, and judged it against the same library as a positive control.
```
Ordinary held-out avg familiarity: 0.25
Positive control (reworded duplicate): 0.99
Control minus ordinary avg: 0.74
```
**The judge works.** 0.99 vs 0.25 is a strong, clean signal — Jev can detect a genuine match when one exists. So the flat result above is not a broken instrument.

**Offline re-analysis (free — cached data, no new API calls):** thresholded at familiarity ≥ 0.5.
```
familiarity >= 0.5: n=1 (tier 1-5, correct=True) → 100% acc, but n=1 is statistically meaningless
familiarity <  0.5: n=83 → 65.1% acc
```

**Conclusion:** the null result in M2 v2 is a **library-coverage floor effect**, not evidence against the hypothesis. 6 exemplars spanning at most 2 technique types cannot meaningfully overlap with 84 problems drawn from geometry, number theory, combinatorics, algebra, precalculus, etc. Only 1/84 crossed even a 0.5 threshold. The judge is proven sensitive (via the control); the library is proven too sparse to exercise it.

## 6. M2 v3: curated, technique-diverse library (coverage fix, still null)

Hand-curated 15 exemplars — up from 6 — deliberately spanning distinct competition-math technique families (algebra/linear-combination, 3D geometry, combinatorics/set sums, inclusion-exclusion, coloring/symmetry, tangent circles, triangle centers, tetrahedron geometry, circular-arrangement probability, prime number theory, cyclotomic polynomials, sequences, rate word problems, game theory, lattice paths) **and, critically, all three difficulty tiers** (5 easy, 6 mid, 4 hard) — the old library was easy+mid only, so hard-tier technique families had zero representation. Selected via `select_curated_library()`, which pulls exact matches out of the existing fast-model cache by unique question prefix — no new fast-model calls, only new Jev judgments needed.

**Engineering fix required first:** 15 full AIME solutions (avg ~5,300 chars each) plus a held-out query exceeded Jev's max-token limit (`400 Bad Request: max_tokens_exceeded`). Fixed by truncating each library solution to a 400-character excerpt in `build_library()` — enough to convey the method, not the full proof.

**Result:**
```
1-5 (easy)      right avg=0.28 (n=23)  wrong avg=0.33 (n= 2)  gap=-0.06
6-10 (mid)      right avg=0.31 (n=16)  wrong avg=0.34 (n= 8)  gap=-0.03
11-15 (hard)    right avg=0.36 (n= 7)  wrong avg=0.32 (n=19)  gap= 0.03
```
Still flat — noise-level gaps in every tier, same as the 6-example library. Positive control still holds (0.99 vs 0.31 ordinary avg), so the judge is still working correctly; the null is not an instrument problem.

**Offline threshold re-analysis surfaced a second bug:** `analyze_familiarity_threshold.py` joined on question text only, without checking `library_signature` — it silently mixed familiarity scores computed against the *old* 6-example library with scores from the *new* 15-example library (any question held-out under the old library but now inside the new library still had a stale cached score). First run reported "87 held-out problems" (should be 75 = 90 − 15). Fixed by filtering to only the majority `library_signature` before joining, with a printed count of dropped stale rows.

**Corrected result:**
```
Dropped 13 stale scores from the old library.
Joined 75 held-out problems (correct).

familiarity >= 0.5: n=2 (50.0% acc)
familiarity <  0.5: n=73 (61.6% acc)
```
Only **2 of 75 (2.7%)** held-out problems crossed even a 0.5 match threshold against a 15-example, all-tier, technique-diverse library — barely more than the 1/84 (1.2%) from the 6-example library. And where it did fire, accuracy was *not* elevated (50% vs 61.6% baseline; n=2 is too small to call this a real negative, but it is definitively not a positive).

**Conclusion — this is no longer a coverage problem.** Tripling the library and spreading it across every tier and a dozen distinct techniques moved the match rate by only ~1.5 points (1.2% → 2.7%) and did not open any right/wrong gap. Three independent attempts (ungrounded, 6-example grounded, 15-example diverse grounded) all land on the same null, while the judge itself is repeatedly proven sensitive via the control. The most likely explanation: **AIME's difficulty lives in execution, not method-identification.** The benchmark is specifically constructed so that most problems, even ones built on a technique you've seen before, have their own unique twist — recognizing "I've used Vieta's before" doesn't predict whether *this instance's* specific numbers will trip up the fast model. Type-recognition and per-instance execution success may simply be different constructs in a domain adversarially designed to separate them.

## 7. Where this stands

Established:
1. **A real, structure-correlated capability gap exists** (AIME, tiered by contest problem number: 96.7% / 86.7% / 56.7%).
2. **Jev's grounded familiarity judge is a real, working similarity detector** — proven repeatedly via positive control (0.99 vs ~0.25-0.31 baseline, across two library versions).
3. **Neither an ungrounded nor a technique-diverse grounded familiarity signal predicts per-problem fast-model success on AIME.** This has now survived a coverage-based counterargument (6 → 15 exemplars, all tiers) without changing.

**Working hypothesis for why:** familiarity-with-a-technique and execution-success-on-this-instance are weakly correlated by construction in an adversarially-designed benchmark like AIME. The pivot's core claim ("have I seen this before" → "can I skip deliberation") may still hold in domains where recognizing the type *does* mostly determine success — just not in one built to defeat exactly that shortcut.

**Next step (in progress):** test the type-recognition-vs-execution-success distinction directly, likely on a less adversarial dataset — one where problems within a recognized category are more homogeneous in execution difficulty, so a genuine type-match is more likely to actually predict success if the underlying mechanism is real at all.

## 8. Option B: same method, less adversarial domain (GSM8K harder slice)

Rationale: AIME is deliberately constructed so a known technique still requires unique execution — a bad domain to validate "type-match predicts success" on. GSM8K's harder slice (rows 300-399) is the opposite: built from ~5-6 genuinely recurring templates (percentage, rate-per-unit, remaining-after-spend, ratio-multiplier, sum-total). If grounded familiarity ever predicts execution success, this is the domain where it should.

Built a library with 2 actually-solved exemplars **per tag** (so every template has real representation, unlike AIME's sparse coverage), held out the rest, ran the same grounded-familiarity + caching + positive-control methodology (`gsm8k_grounded_familiarity_test.py`, `analyze_gsm8k_familiarity_threshold.py`).

**Result:**
```
percentage             right avg=0.75 (n=11)  wrong avg=0.52 (n=2)  gap= 0.23
rate_per_unit          right avg=0.77 (n=18)  wrong avg=0.79 (n=2)  gap=-0.02
ratio_multiplier       right avg=0.82 (n=23)  wrong avg=0.89 (n=2)  gap=-0.07
remaining_after_spend  right avg=0.82 (n= 7)  wrong avg=nan (n=0)  gap= nan
sum_total              right avg=0.78 (n= 6)  wrong avg=0.87 (n=1)  gap=-0.09
unclassified           right avg=0.76 (n=15)  wrong avg=0.77 (n=1)  gap=-0.01

Ordinary held-out avg familiarity: 0.78
Positive control familiarity:      0.99  (control minus ordinary avg: 0.21)
```

**Conclusion — a different failure mode, same outcome.** Unlike AIME (2.7% match rate — floor effect), GSM8K saturated near ceiling: 0.78 average familiarity means nearly everything "looks familiar" in a domain this templated, leaving almost no dynamic range to separate real matches from superficial ones (control gain shrank from AIME's 0.68-0.74 to just 0.21). Within-tag gaps are noise-level everywhere except `percentage` (+0.23, but n=2 wrong — not reliable). Total wrong count across the whole 100-problem run is only 8, split across 6 tags — too little statistical power to detect a real effect even if one existed.

**This is the decisive composite result.** Two domains chosen to bracket the hypothesis from opposite sides — AIME (technique known, execution adversarial) and GSM8K (technique known, execution homogeneous) — both defeat grounded, LLM-introspected familiarity, for opposite structural reasons (floor vs. ceiling). The judge itself works in both cases (control consistently separates from baseline). The conclusion is no longer "wrong dataset" — it's that **bare LLM introspection over a text library, asked "does this match a known type," does not carry enough resolution to track per-instance execution risk**, regardless of domain. A different operationalization of "competence" (e.g. embedding-based nearest-neighbor retrieval with a numeric similarity score, or tracking success rate over actual repeated exposure rather than one-shot text comparison) would be needed to keep testing the pivot's core claim.

## 9. Infrastructure built this session

- `data_loader.py`: added `load_math()` (MATH-500, numeric-only filter) and `load_aime()` (zero-padding-safe ground truth normalization). `load_gsm8k()` untouched — still used by 6 other test scripts.
- `fast_stage.py`: model swapped `deepseek-v4-pro` → `deepseek-flash`.
- `cache_utils.py`: `load_cache()` / `append_result()` / `signature()` — JSONL caching keyed by question text, with library-signature invalidation for familiarity judgments.
- `Dataset/MATH-500/test.jsonl` (500 rows), `Dataset/AIME/test.jsonl` (90 rows, pooled 2022-2024).
- `test_scripts/`: `competence_precondition_test.py` (M1, MATH-500 subject/level breakdown), `aime_baseline_test.py` (M1, AIME tiered by problem number), `familiarity_test.py` (M2 v1, ungrounded), `grounded_familiarity_test.py` (M2 v2/v3, AIME grounded + cached + curated library + positive control), `analyze_familiarity_threshold.py` (AIME offline re-analysis, filters by current library signature), `gsm8k_grounded_familiarity_test.py` (Option B, same methodology on GSM8K's templated harder slice), `analyze_gsm8k_familiarity_threshold.py` (matching offline re-analysis for GSM8K).
- `results/aime_fast_model_cache.jsonl`, `results/aime_grounded_familiarity_cache.jsonl`, `results/gsm8k_fast_model_cache.jsonl`, `results/gsm8k_grounded_familiarity_cache.jsonl` — cached ground truth and familiarity judgments, reusable across analysis iterations without new API cost.
