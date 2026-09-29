import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_math, extract_answers
from fast_stage import call_fast_model

# MATH-500 ships real ground-truth structure labels (subject, level),
# so no keyword heuristic is needed here anymore.

def run(n=316):
    problems = load_math(n=n)
    by_subject = defaultdict(lambda: {"correct": 0, "total": 0})
    by_level = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, p in enumerate(problems):
        response = call_fast_model(p["question"])
        predicted = extract_answers(response)
        correct = (predicted == p["ground_truth"])

        by_subject[p["subject"]]["total"] += 1
        by_level[p["level"]]["total"] += 1
        if correct:
            by_subject[p["subject"]]["correct"] += 1
            by_level[p["level"]]["correct"] += 1

        print(f"[{i+1}/{len(problems)}] subject={p['subject']} level={p['level']} correct={correct}")

    print("\n--- Accuracy by subject ---")
    for subject, r in sorted(by_subject.items()):
        acc = 100 * r["correct"] / r["total"]
        print(f"{subject:22s} n={r['total']:3d}  acc={acc:5.1f}%")

    print("\n--- Accuracy by difficulty level ---")
    for level, r in sorted(by_level.items()):
        acc = 100 * r["correct"] / r["total"]
        print(f"level {level}                n={r['total']:3d}  acc={acc:5.1f}%")


if __name__ == "__main__":
    run(n=10)