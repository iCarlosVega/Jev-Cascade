import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_loader import load_gsm8k, extract_answers
from fast_stage import call_fast_model
from cache_utils import load_cache, append_result, signature
from typesafe_sdk import Noul, TypeSafeClient
from dotenv import load_dotenv

load_dotenv()
jev = TypeSafeClient(api_key=os.environ["TYPESAFE_API"])

FAST_CACHE_PATH = "results/gsm8k/gsm8k_fast_model_cache.jsonl"
FAMILIARITY_CACHE_PATH = "results/gsm8k/gsm8k_grounded_familiarity_cache.jsonl"

# Option B: test type-recognition vs execution-success on a LESS
# adversarial domain than AIME. GSM8K word problems are built from a
# small set of genuinely recurring templates (percentage, rate-per-unit,
# remaining-after-spend, etc.) -- unlike AIME, which is deliberately
# constructed so a known technique still requires unique execution.
# If grounded familiarity ever predicts success, this is the domain
# where it should. If it doesn't work here either, the framing itself
# (not just "wrong dataset") is in trouble.

def tag_problem(question: str) -> str:
    """Same heuristic tagger from the original M1 precondition test --
    used ONLY to pick a diverse library here, not as the measured
    signal itself."""
    q = question.lower()
    if "percent" in q or "%" in q:
        return "percentage"
    if "half" in q or "twice" in q or "double" in q or "triple" in q:
        return "ratio_multiplier"
    if "left" in q or "remaining" in q or "spent" in q:
        return "remaining_after_spend"
    if "each" in q or "per " in q:
        return "rate_per_unit"
    if "total" in q or "altogether" in q:
        return "sum_total"
    return "unclassified"

def build_library(solved_problems: list[dict], max_solution_chars: int = 400) -> str:
    chunks = []
    for p in solved_problems:
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
    prompt = (
        "Rewrite the following math problem using different wording "
        "and phrasing, but keep the exact same numbers, mathematical "
        "content, and question being asked. Output ONLY the reworded "
        "problem text, nothing else.\n\n"
        f"Original problem:\n{question}"
    )
    return call_fast_model(prompt)

def get_fast_result(p: dict, tag: str, fast_cache: dict) -> dict:
    cached = fast_cache.get(p["question"])
    if cached is not None:
        return cached

    response = call_fast_model(p["question"])
    predicted = extract_answers(response)
    is_correct = (predicted == p["ground_truth"])

    result = {
        "question": p["question"],
        "reasoning": p["reasoning"],
        "tag": tag,
        "predicted": predicted,
        "correct": is_correct,
    }
    append_result(FAST_CACHE_PATH, result)
    fast_cache[p["question"]] = result
    return result

def get_familiarity(question: str, library_text: str, familiarity_cache: dict) -> float:
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

def run(n=400, offset=300, n_exemplars_per_tag=2):
    problems = load_gsm8k(n=n)[offset:]
    fast_cache = load_cache(FAST_CACHE_PATH)
    familiarity_cache = load_cache(FAMILIARITY_CACHE_PATH)

    # Phase A: fast-model ground truth for every problem (cached).
    solved_by_tag = defaultdict(list)
    all_results = []

    for i, p in enumerate(problems):
        tag = tag_problem(p["question"])
        result = get_fast_result(p, tag, fast_cache)

        all_results.append((p, tag, result["correct"]))
        if result["correct"]:
            solved_by_tag[tag].append(result)

        print(f"[{i+1}/{len(problems)}] tag={tag} correct={result['correct']}")

    # Phase B: library = a couple of actually-solved exemplars PER TAG,
    # so every known template has real representation.
    library_problems = []
    for tag, rows in solved_by_tag.items():
        library_problems.extend(rows[:n_exemplars_per_tag])
    library_text = build_library(library_problems)
    library_question_texts = {r["question"] for r in library_problems}

    print(f"\nBuilt library from {len(library_problems)} actually-solved problems across {len(solved_by_tag)} tags.")

    # Positive control.
    control_source = library_problems[0]
    control_question = paraphrase_problem(control_source["question"])
    control_familiarity = get_familiarity(control_question, library_text, familiarity_cache)
    print(f"\n--- Positive control (reworded duplicate of a library problem) ---")
    print(f"control familiarity={control_familiarity:.2f}")

    # Phase C: grounded familiarity on held-out problems only.
    by_tag_outcome = defaultdict(lambda: {"right": [], "wrong": []})

    for p, tag, is_correct in all_results:
        if p["question"] in library_question_texts:
            continue

        familiarity = get_familiarity(p["question"], library_text, familiarity_cache)
        key = "right" if is_correct else "wrong"
        by_tag_outcome[tag][key].append(familiarity)

        print(f"held-out: tag={tag} grounded_familiarity={familiarity:.2f} correct={is_correct}")

    # Phase D: within-tag right vs wrong.
    print("\n--- Grounded familiarity: right vs wrong, WITHIN each tag ---")
    for tag in sorted(by_tag_outcome.keys()):
        right = by_tag_outcome[tag]["right"]
        wrong = by_tag_outcome[tag]["wrong"]

        avg_right = sum(right) / len(right) if right else float("nan")
        avg_wrong = sum(wrong) / len(wrong) if wrong else float("nan")
        gap = avg_right - avg_wrong if right and wrong else float("nan")

        print(f"{tag:22s} right: avg={avg_right:.2f} (n={len(right):2d})   "
              f"wrong: avg={avg_wrong:.2f} (n={len(wrong):2d})   gap={gap:.2f}")

    all_held_out = [v for outcome in by_tag_outcome.values() for v in outcome["right"] + outcome["wrong"]]
    overall_avg = sum(all_held_out) / len(all_held_out)
    print(f"\nOrdinary held-out avg familiarity: {overall_avg:.2f}")
    print(f"Positive control familiarity:      {control_familiarity:.2f}")
    print(f"Control minus ordinary avg:        {control_familiarity - overall_avg:.2f}")


if __name__ == "__main__":
    run(n=400, offset=300, n_exemplars_per_tag=2)
