import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_loader import load_gsm8k
from jev_stage import judge

def make_wrong_answer(ground_truth: str) -> str:
    """Perturb the correct number to make an obviously wrong candidate."""
    try:
        n = float(ground_truth)
        wrong = n + random.choice([-1, 1]) * max(1, int(n * 0.3))
        return str(int(wrong)) if wrong == int(wrong) else str(wrong)
    except ValueError:
        return ground_truth + "0"  # fallback if not numeric


if __name__ == "__main__":
    problems = load_gsm8k(n=15)

    correct_picks = 0
    confidences_when_right = []
    confidences_when_wrong = []

    for p in problems:
        correct = f"{p['reasoning']} The answer is {p['ground_truth']}."
        wrong_answer = make_wrong_answer(p["ground_truth"])
        wrong = f"{p['reasoning']} The answer is {wrong_answer}."

        # randomize order so position isn't a confound
        options = [correct, wrong]
        random.shuffle(options)
        correct_index = options.index(correct)

        pick, confidence = judge(p["question"], options)

        picked_correct = (pick == correct)
        if picked_correct:
            correct_picks += 1
            confidences_when_right.append(confidence)
        else:
            confidences_when_wrong.append(confidence)

        print(f"picked_correct={picked_correct}  confidence={confidence:.2f}")

    print(f"\nAccuracy picking correct candidate: {correct_picks}/{len(problems)}")
    if confidences_when_right:
        avg_right = sum(confidences_when_right) / len(confidences_when_right)
        print(f"Avg confidence when RIGHT: {avg_right:.2f}")
    if confidences_when_wrong:
        avg_wrong = sum(confidences_when_wrong) / len(confidences_when_wrong)
        print(f"Avg confidence when WRONG: {avg_wrong:.2f}")
