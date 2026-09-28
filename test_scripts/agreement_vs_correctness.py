import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_gsm8k
from jev_stage import judge


def make_wrong_answer(ground_truth: str, seed: int) -> str:
    n = float(ground_truth)
    delta = max(1, int(n * 0.2)) * (seed + 1)
    wrong = n + random.choice([-1, 1]) * delta
    return str(int(wrong))


def build_sets(p):
    correct = f"{p['reasoning']} The answer is {p['ground_truth']}."
    wrong_1 = f"{p['reasoning']} The answer is {make_wrong_answer(p['ground_truth'], 0)}."
    wrong_2 = f"{p['reasoning']} The answer is {make_wrong_answer(p['ground_truth'], 1)}."

    return {
        "max_agreement": [correct, correct, correct],
        "medium_agreement": [correct, correct, wrong_1],
        "min_agreement": [correct, wrong_1, wrong_2],
    }


if __name__ == "__main__":
    problems = load_gsm8k(n=10)
    results = {"max_agreement": [], "medium_agreement": [], "min_agreement": []}

    for p in problems:
        sets = build_sets(p)
        for label, candidates in sets.items():
            shuffled = candidates[:]
            random.shuffle(shuffled)
            pick, confidence = judge(p["question"], shuffled)
            results[label].append(confidence)
            print(f"{label:16s} confidence={confidence:.2f}")

    print()
    for label, confs in results.items():
        avg = sum(confs) / len(confs)
        print(f"{label:16s} avg confidence = {avg:.2f} (n={len(confs)})")
