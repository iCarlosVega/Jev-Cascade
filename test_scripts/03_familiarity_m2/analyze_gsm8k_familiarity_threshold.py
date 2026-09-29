import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cache_utils import load_cache

FAST_CACHE_PATH = "results/gsm8k/gsm8k_fast_model_cache.jsonl"
FAMILIARITY_CACHE_PATH = "results/gsm8k/gsm8k_grounded_familiarity_cache.jsonl"

# Same offline re-analysis as analyze_familiarity_threshold.py, pointed
# at the GSM8K caches instead of AIME's. Filters to the current
# library's signature to avoid mixing scores across library versions.

def run(threshold=0.5):
    fast_cache = load_cache(FAST_CACHE_PATH)
    familiarity_cache = load_cache(FAMILIARITY_CACHE_PATH)

    sig_counts = Counter(row["library_signature"] for row in familiarity_cache.values())
    current_sig, current_count = sig_counts.most_common(1)[0]
    stale = sum(sig_counts.values()) - current_count
    if stale:
        print(f"Dropping {stale} stale familiarity score(s) computed against an older library.\n")

    joined = []
    for question, fam_row in familiarity_cache.items():
        if fam_row["library_signature"] != current_sig:
            continue
        fast_row = fast_cache.get(question)
        if fast_row is None:
            continue
        joined.append({
            "question": question,
            "tag": fast_row["tag"],
            "correct": fast_row["correct"],
            "familiarity": fam_row["familiarity"],
        })

    high = [r for r in joined if r["familiarity"] >= threshold]
    low = [r for r in joined if r["familiarity"] < threshold]

    print(f"Joined {len(joined)} held-out problems with both cached results.\n")

    print(f"--- familiarity >= {threshold} (n={len(high)}) ---")
    for r in high:
        print(f"  tag={r['tag']:22s} familiarity={r['familiarity']:.2f} correct={r['correct']}")
    if high:
        acc = 100 * sum(r["correct"] for r in high) / len(high)
        print(f"  accuracy: {acc:.1f}%")

    print(f"\n--- familiarity < {threshold} (n={len(low)}) ---")
    if low:
        acc = 100 * sum(r["correct"] for r in low) / len(low)
        print(f"  accuracy: {acc:.1f}%")


if __name__ == "__main__":
    run(threshold=0.5)
