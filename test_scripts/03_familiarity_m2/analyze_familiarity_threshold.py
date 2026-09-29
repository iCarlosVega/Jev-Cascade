import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cache_utils import load_cache

FAST_CACHE_PATH = "results/aime/aime_fast_model_cache.jsonl"
FAMILIARITY_CACHE_PATH = "results/aime/aime_grounded_familiarity_cache.jsonl"

# Pure offline re-analysis: no API calls, no network. Reads the two
# cache files already on disk and asks a narrower question than before:
# not "does familiarity separate right/wrong on average" but "on the
# problems where familiarity actually fired high, is the fast model
# more likely to have gotten it right?"

def run(threshold=0.5):
    fast_cache = load_cache(FAST_CACHE_PATH)
    familiarity_cache = load_cache(FAMILIARITY_CACHE_PATH)

    # The library has changed across runs (6-example -> 15-example),
    # so the familiarity cache can hold scores computed against
    # DIFFERENT libraries for different questions. Mixing those would
    # silently contaminate the result. Only trust the majority
    # signature -- the one from the most recent full run -- and drop
    # everything else as stale.
    sig_counts = Counter(row["library_signature"] for row in familiarity_cache.values())
    current_sig, current_count = sig_counts.most_common(1)[0]
    stale = sum(sig_counts.values()) - current_count
    if stale:
        print(f"Dropping {stale} stale familiarity score(s) computed against an older library.\n")

    # Join on question text. The control's paraphrased question was
    # never run through the fast model, so it's naturally excluded here.
    joined = []
    for question, fam_row in familiarity_cache.items():
        if fam_row["library_signature"] != current_sig:
            continue
        fast_row = fast_cache.get(question)
        if fast_row is None:
            continue
        joined.append({
            "question": question,
            "tier": fast_row["tier"],
            "correct": fast_row["correct"],
            "familiarity": fam_row["familiarity"],
        })

    high = [r for r in joined if r["familiarity"] >= threshold]
    low = [r for r in joined if r["familiarity"] < threshold]

    print(f"Joined {len(joined)} held-out problems with both cached results.\n")

    print(f"--- familiarity >= {threshold} (n={len(high)}) ---")
    for r in high:
        print(f"  tier={r['tier']:15s} familiarity={r['familiarity']:.2f} correct={r['correct']}")
    if high:
        acc = 100 * sum(r["correct"] for r in high) / len(high)
        print(f"  accuracy: {acc:.1f}%")

    print(f"\n--- familiarity < {threshold} (n={len(low)}) ---")
    if low:
        acc = 100 * sum(r["correct"] for r in low) / len(low)
        print(f"  accuracy: {acc:.1f}%")


if __name__ == "__main__":
    run(threshold=0.5)
