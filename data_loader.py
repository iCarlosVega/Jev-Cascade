import json
import re

def load_gsm8k(path="Dataset/test.jsonl", n=20):
    problems = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            row = json.loads(line)
            question = row["question"]
            full_answer = row["answer"]
            ground_truth = full_answer.split("####")[-1].strip()
            problems.append({
                "question": question,
                "reasoning": full_answer.split("####")[0].strip(),
                "ground_truth": ground_truth,
            })
    return problems

def load_math(path="Dataset/MATH-500/test.jsonl", n=500, numeric_only=True):
    """
    Load MATH-500 (Hendrycks et al.) problems. Each row already carries
    real ground-truth structure labels: `subject` (7 categories) and
    `level` (1-5 difficulty) -- no keyword-heuristic tagging needed.

    numeric_only=True filters out problems whose answer is a LaTeX
    expression (fraction, symbolic, coordinate, etc.) rather than a
    bare number, since extract_answers() only knows how to compare
    plain numbers. This drops ~37% of MATH-500, mostly Geometry.
    """
    problems = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            answer = row["answer"].strip()
            if numeric_only and not re.fullmatch(r"-?\d+(\.\d+)?", answer):
                continue
            problems.append({
                "question": row["problem"],
                "reasoning": row["solution"],
                "ground_truth": answer,
                "subject": row["subject"],
                "level": row["level"],
            })
            if len(problems) >= n:
                break
    return problems

def load_aime(path="Dataset/AIME/test.jsonl", n=90):
    """
    Load AIME (2022-2024, pooled via AI-MO/aimo-validation-aime), 90
    competition-math problems. Answers are always integers 0-999, but
    stored zero-padded (e.g. "033"), which would never string-match a
    model's free-text "33" -- normalize to plain int strings here so
    grading isn't silently wrong.
    """
    problems = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            row = json.loads(line)
            problems.append({
                "question": row["problem"],
                "reasoning": row["solution"],
                "ground_truth": str(int(row["answer"])),
                "url": row["url"],
            })
    return problems

def extract_answers(text: str) -> str:
    """Pull a final numeric answer out of any model's free-text response"""
    if "####" in text:
        text = text.split("####")[-1]

    # Normalize: strip $, commas, trailing punctuation
    match = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not match:
        return ""
    answer = match[-1].replace(',', '')
    if answer.endswith("."):
        answer = answer[:-1]
    return answer

if __name__ == "__main__":
    problems = load_gsm8k(n=20)
    correct = 0
    for p in problems:
        full_text = p["reasoning"] + " #### " + p["ground_truth"]
        predicted = extract_answers(full_text)
        if predicted == p["ground_truth"]:
            correct += 1
    print(f"Self_check Accuracy: {correct}/{len(problems)}({100*correct/len(problems):.1f}%)")