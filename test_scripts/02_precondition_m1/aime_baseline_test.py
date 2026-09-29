import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_loader import load_aime, extract_answers
from fast_stage import call_fast_model

# AIME has no subject/level metadata like MATH-500 -- but each contest
# orders its 15 problems roughly by increasing difficulty (that's the
# contest writers' own difficulty labeling, not a guess). We recover
# that number from the URL and bucket into early/mid/late tiers.

def problem_tier(url: str) -> str:
    match = re.search(r"Problem_(\d+)", url)
    num = int(match.group(1))
    if num <= 5:
        return "1-5 (easy)"
    if num <= 10:
        return "6-10 (mid)"
    return "11-15 (hard)"

def run(n=90):
    problems = load_aime(n=n)
    correct = 0
    by_tier = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, p in enumerate(problems):
        response = call_fast_model(p["question"])
        predicted = extract_answers(response)
        is_correct = (predicted == p["ground_truth"])
        correct += is_correct

        tier = problem_tier(p["url"])
        by_tier[tier]["total"] += 1
        if is_correct:
            by_tier[tier]["correct"] += 1

        print(f"[{i+1}/{len(problems)}] tier={tier} predicted={predicted!r} ground_truth={p['ground_truth']!r} correct={is_correct}")

    acc = 100 * correct / len(problems)
    print(f"\nAIME overall accuracy: {correct}/{len(problems)} ({acc:.1f}%)")

    print("\n--- Accuracy by contest problem-number tier ---")
    for tier in ["1-5 (easy)", "6-10 (mid)", "11-15 (hard)"]:
        r = by_tier[tier]
        tier_acc = 100 * r["correct"] / r["total"]
        print(f"{tier:15s} n={r['total']:3d}  acc={tier_acc:5.1f}%")


if __name__ == "__main__":
    run(n=90)
