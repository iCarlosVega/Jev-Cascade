import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_gsm8k, extract_answers
from fast_stage import get_n_candidates
from jev_stage import judge

if __name__ == "__main__":
    problems = load_gsm8k(n=15)

    duplicate_confidences = []
    paraphrase_confidences = []
    skipped = 0

    for p in problems:
        real_candidates = get_n_candidates(p["question"], n=3)

        # only use cases where all 3 independently agree with ground truth,
        # so both conditions below hold "correct and numerically agreeing" constant
        answers = [extract_answers(c) for c in real_candidates]
        if not all(a == p["ground_truth"] for a in answers):
            skipped += 1
            continue

        # Condition A: same correct answer, duplicated verbatim (zero wording diversity)
        duplicate_set = [real_candidates[0]] * 3
        _, dup_confidence = judge(p["question"], duplicate_set)
        duplicate_confidences.append(dup_confidence)

        # Condition B: same correct final answer, but 3 independently-worded versions
        _, para_confidence = judge(p["question"], real_candidates)
        paraphrase_confidences.append(para_confidence)

        print(f"duplicate={dup_confidence:.2f}  paraphrase={para_confidence:.2f}")

    print(f"\nSkipped (candidates disagreed): {skipped}/{len(problems)}")
    if duplicate_confidences:
        avg_dup = sum(duplicate_confidences) / len(duplicate_confidences)
        avg_para = sum(paraphrase_confidences) / len(paraphrase_confidences)
        print(f"Avg confidence, verbatim duplicates: {avg_dup:.2f} (n={len(duplicate_confidences)})")
        print(f"Avg confidence, real paraphrases:     {avg_para:.2f} (n={len(paraphrase_confidences)})")
