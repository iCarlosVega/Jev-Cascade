import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_loader import load_aime, extract_answers
from fast_stage import call_fast_model
from cache_utils import load_cache, append_result, signature
from typesafe_sdk import Noul, TypeSafeClient
from dotenv import load_dotenv

load_dotenv()
jev = TypeSafeClient(api_key=os.environ["TYPESAFE_API"])

FAST_CACHE_PATH = "results/aime/aime_fast_model_cache.jsonl"
FAMILIARITY_CACHE_PATH = "results/aime/aime_grounded_familiarity_cache.jsonl"

# Hand-curated to span distinct competition-math techniques and ALL
# three tiers (the old library was only easy+mid, so it had zero
# coverage of hard-tier technique families). Each entry is a unique
# question prefix identifying one specific, actually-solved problem
# in the fast-model cache -- verified to match exactly one row before
# being used here.
CURATED_LIBRARY_PREFIXES = [
    "Quadratic polynomials $P(x)$ and $Q(x)$ have leading coefficients",
    "Three spheres with radii $11$, $13$, and $19$",
    "For any finite set $X$, let $| X |$",
    "Among the 900 residents of Aimeville",
    "Each vertex of a regular dodecagon",
    "A circle with radius $6$ is externally tangent",
    r"Let $\triangle ABC$ have circumcenter $O$",
    "Let $ABCD$ be a tetrahedron",
    "Five men and nine women stand equally spaced",
    "Let $p$ be the least prime number",
    "There is a polynomial $P(x)$ with integer coefficients",
    "Find the number of ordered pairs of integers $(a, b)$ such that the sequence",
    "A straight river that is $264$ meters wide",
    "Alice and Bob play the following game. A stack of $n$ tokens",
    "Consider the paths of length $16$",
]

# Real M2, cached: fast-model correctness and Jev's grounded familiarity
# judgments are stored on disk so re-running analysis costs nothing and
# every rerun compares against the SAME frozen ground truth (removes
# call-to-call model noise as a confound between experiment versions).

def problem_tier(url: str) -> str:
    match = re.search(r"Problem_(\d+)", url)
    num = int(match.group(1))
    if num <= 5:
        return "1-5 (easy)"
    if num <= 10:
        return "6-10 (mid)"
    return "11-15 (hard)"

def select_curated_library(fast_cache: dict) -> list[dict]:
    """Pull the 15 hand-picked, technique-diverse exemplars out of the
    fast-model cache by their unique question prefix. All must already
    be cached and correctly solved -- this only selects, it never
    triggers new API calls."""
    library_problems = []
    for prefix in CURATED_LIBRARY_PREFIXES:
        matches = [row for row in fast_cache.values() if row["question"].startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly 1 cached match for prefix {prefix!r}, got {len(matches)}")
        row = matches[0]
        if not row["correct"]:
            raise ValueError(f"Curated exemplar was not actually solved correctly: {prefix!r}")
        library_problems.append(row)
    return library_problems

def build_library(solved_problems: list[dict], max_examples: int = 6, max_solution_chars: int = 400) -> str:
    """Full AIME solutions average ~5,300 chars -- 15 of them plus a
    held-out query blows past Jev's max token limit. We only need
    enough of the solution to convey the METHOD, not a full rigorous
    proof, so each one is truncated to a short excerpt."""
    chunks = []
    for p in solved_problems[:max_examples]:
        solution = p["reasoning"][:max_solution_chars]
        if len(p["reasoning"]) > max_solution_chars:
            solution += "..."
        chunks.append(f"Problem: {p['question']}\nSolution: {solution}")
    return "\n\n---\n\n".join(chunks)

def judge_grounded_familiarity(query: str, library: str) -> float:
    response = jev.system_one(
        state=f"Known solved examples:\n\n{library}\n\nNew problem:\n{query}",
        questions={
            "matches_known": Noul(
                instructions=(
                    "Does this new problem match the type of one of the "
                    "known solved examples above, such that the same "
                    "method used there would apply here?"
                ),
            ),
        },
    )
    return response.answers["matches_known"].noul

def paraphrase_problem(question: str) -> str:
    """Reword a problem's surface phrasing while keeping its exact
    mathematical content -- used to build a positive control: a
    'new' problem that SHOULD score high familiarity, since it's
    really just a library problem in different words."""
    prompt = (
        "Rewrite the following math problem using different wording "
        "and phrasing, but keep the exact same numbers, mathematical "
        "content, and question being asked. Output ONLY the reworded "
        "problem text, nothing else.\n\n"
        f"Original problem:\n{question}"
    )
    return call_fast_model(prompt)

def get_fast_result(p: dict, tier: str, fast_cache: dict) -> dict:
    """Return cached fast-model result for this problem, or compute
    and cache it if this is the first time we've seen it."""
    cached = fast_cache.get(p["question"])
    if cached is not None:
        return cached

    response = call_fast_model(p["question"])
    predicted = extract_answers(response)
    is_correct = (predicted == p["ground_truth"])

    result = {
        "question": p["question"],
        "reasoning": p["reasoning"],
        "tier": tier,
        "predicted": predicted,
        "correct": is_correct,
    }
    append_result(FAST_CACHE_PATH, result)
    fast_cache[p["question"]] = result
    return result

def get_familiarity(question: str, library_text: str, familiarity_cache: dict) -> float:
    """Return cached familiarity score IF it was computed against this
    same library. If the library changed, the old cache entry is stale
    and must be recomputed."""
    lib_sig = signature(library_text)
    cached = familiarity_cache.get(question)
    if cached is not None and cached.get("library_signature") == lib_sig:
        return cached["familiarity"]

    familiarity = judge_grounded_familiarity(question, library_text)
    result = {
        "question": question,
        "library_signature": lib_sig,
        "familiarity": familiarity,
    }
    append_result(FAMILIARITY_CACHE_PATH, result)
    familiarity_cache[question] = result
    return familiarity

def run(n=90):
    problems = load_aime(n=n)
    fast_cache = load_cache(FAST_CACHE_PATH)
    familiarity_cache = load_cache(FAMILIARITY_CACHE_PATH)

    # Phase A: get fast-model ground truth for every problem (cached).
    all_results = []

    for i, p in enumerate(problems):
        tier = problem_tier(p["url"])
        result = get_fast_result(p, tier, fast_cache)

        all_results.append((p, tier, result["correct"]))

        print(f"[{i+1}/{len(problems)}] tier={tier} correct={result['correct']}")

    # Phase B: build the competence library from the curated, technique-
    # diverse exemplar set (all pulled from the cache -- no new fast-
    # model calls needed since we already know which ones were solved).
    library_problems = select_curated_library(fast_cache)
    library_text = build_library(library_problems, max_examples=len(library_problems))
    library_question_texts = {p["question"] for p in library_problems}

    print(f"\nBuilt competence library from {len(library_problems)} actually-solved problems.")

    # Positive control: a reworded near-duplicate of a library problem.
    # If judge_grounded_familiarity is sensitive at all, this MUST score
    # noticeably higher than ordinary held-out problems -- it's really
    # just a library problem in disguise. If it doesn't, the judge
    # itself isn't working, independent of whether real matches exist
    # elsewhere in the held-out set.
    control_source = library_problems[0]
    control_question = paraphrase_problem(control_source["question"])
    control_familiarity = get_familiarity(control_question, library_text, familiarity_cache)
    print(f"\n--- Positive control (reworded duplicate of a library problem) ---")
    print(f"control familiarity={control_familiarity:.2f}")

    # Phase C: grounded familiarity on held-out problems only (cached).
    by_tier_outcome = defaultdict(lambda: {"right": [], "wrong": []})

    for p, tier, is_correct in all_results:
        if p["question"] in library_question_texts:
            continue

        familiarity = get_familiarity(p["question"], library_text, familiarity_cache)
        key = "right" if is_correct else "wrong"
        by_tier_outcome[tier][key].append(familiarity)

        print(f"held-out: tier={tier} grounded_familiarity={familiarity:.2f} correct={is_correct}")

    # Phase D: within-tier right vs wrong.
    print("\n--- Grounded familiarity: right vs wrong, WITHIN each tier ---")
    for tier in ["1-5 (easy)", "6-10 (mid)", "11-15 (hard)"]:
        right = by_tier_outcome[tier]["right"]
        wrong = by_tier_outcome[tier]["wrong"]

        avg_right = sum(right) / len(right) if right else float("nan")
        avg_wrong = sum(wrong) / len(wrong) if wrong else float("nan")
        gap = avg_right - avg_wrong if right and wrong else float("nan")

        print(f"{tier:15s} right: avg={avg_right:.2f} (n={len(right):2d})   "
              f"wrong: avg={avg_wrong:.2f} (n={len(wrong):2d})   gap={gap:.2f}")

    all_held_out = [v for outcome in by_tier_outcome.values() for v in outcome["right"] + outcome["wrong"]]
    overall_avg = sum(all_held_out) / len(all_held_out)
    print(f"\nOrdinary held-out avg familiarity: {overall_avg:.2f}")
    print(f"Positive control familiarity:      {control_familiarity:.2f}")
    print(f"Control minus ordinary avg:        {control_familiarity - overall_avg:.2f}")


if __name__ == "__main__":
    run(n=90)
