import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_loader import load_aime, extract_answers
from fast_stage import call_fast_model
from typesafe_sdk import Noul, TypeSafeClient
from dotenv import load_dotenv

load_dotenv()
jev = TypeSafeClient(api_key=os.environ["TYPESAFE_API"])

# M2 continued: the tier-level gap (0.67 right vs 0.51 wrong) could just
# mean "Jev knows hard tier is hard" rather than "Jev discriminates
# individual solvable problems." This version splits right/wrong
# familiarity WITHIN each tier to check for that confound.

def problem_tier(url: str) -> str:
    match = re.search(r"Problem_(\d+)", url)
    num = int(match.group(1))
    if num <= 5:
        return "1-5 (easy)"
    if num <= 10:
        return "6-10 (mid)"
    return "11-15 (hard)"

def judge_familiarity(query: str) -> float:
    response = jev.system_one(
        state=f"Problem: {query}",
        questions={
            "familiar": Noul(
                instructions=(
                    "Does this problem match a type you have a reliable, "
                    "well-practiced method for solving, as opposed to one "
                    "requiring novel multi-step insight or an unfamiliar trick?"
                ),
            ),
        },
    )
    return response.answers["familiar"].noul

def run(n=90):
    problems = load_aime(n=n)

    # per-tier, split by right/wrong
    by_tier_outcome = defaultdict(lambda: {"right": [], "wrong": []})

    for i, p in enumerate(problems):
        tier = problem_tier(p["url"])
        familiarity = judge_familiarity(p["question"])

        response = call_fast_model(p["question"])
        predicted = extract_answers(response)
        is_correct = (predicted == p["ground_truth"])

        key = "right" if is_correct else "wrong"
        by_tier_outcome[tier][key].append(familiarity)

        print(f"[{i+1}/{len(problems)}] tier={tier} familiarity={familiarity:.2f} correct={is_correct}")

    print("\n--- Familiarity: right vs wrong, WITHIN each tier ---")
    for tier in ["1-5 (easy)", "6-10 (mid)", "11-15 (hard)"]:
        right = by_tier_outcome[tier]["right"]
        wrong = by_tier_outcome[tier]["wrong"]

        avg_right = sum(right) / len(right) if right else float("nan")
        avg_wrong = sum(wrong) / len(wrong) if wrong else float("nan")
        gap = avg_right - avg_wrong if right and wrong else float("nan")

        print(
            f"{tier:15s} right: avg={avg_right:.2f} (n={len(right):2d})   "
            f"wrong: avg={avg_wrong:.2f} (n={len(wrong):2d})   gap={gap:.2f}"
        )


if __name__ == "__main__":
    run(n=90)
