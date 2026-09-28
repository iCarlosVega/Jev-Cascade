import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_gsm8k, extract_answers
from fast_stage import get_n_candidates
from jev_stage import judge


if __name__ == "__main__":
    problems = load_gsm8k(n=15)
    n = 3

    correct_picks = 0
    confidences_when_right = []
    confidences_when_wrong = []

    for p in problems:
        candidates = get_n_candidates(p["question"], n=n)
        pick, confidence = judge(p["question"], candidates)

        predicted = extract_answers(pick)
        picked_correct = (predicted == p["ground_truth"])

        if picked_correct:
            correct_picks += 1
            confidences_when_right.append(confidence)
        else:
            confidences_when_wrong.append(confidence)

        print(
            f"picked_correct={picked_correct}  confidence={confidence:.2f}  "
            f"predicted={predicted!r}  ground_truth={p['ground_truth']!r}"
        )
        if not picked_correct:
            print(f"  --- raw picked text ---\n{pick}\n  -----------------------")

    print(f"\nAccuracy of Jev's pick vs ground truth: {correct_picks}/{len(problems)}")
    if confidences_when_right:
        avg_right = sum(confidences_when_right) / len(confidences_when_right)
        print(f"Avg confidence when RIGHT: {avg_right:.2f}")
    if confidences_when_wrong:
        avg_wrong = sum(confidences_when_wrong) / len(confidences_when_wrong)
        print(f"Avg confidence when WRONG: {avg_wrong:.2f}")
