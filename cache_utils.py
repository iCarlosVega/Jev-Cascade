import hashlib
import json
import os

def load_cache(path: str) -> dict:
    """Load cached results, keyed by problem question text."""
    if not os.path.exists(path):
        return {}
    cache = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            cache[row["question"]] = row
    return cache

def append_result(path: str, result: dict):
    """Append one result row, creating the containing dir if needed."""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(result) + "\n")

def signature(text: str) -> str:
    """Short stable hash, used to detect when a cached judgment was
    made against a DIFFERENT library/context than the current run --
    e.g. if the competence library changes, old cached familiarity
    scores are no longer valid and must be recomputed."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]
