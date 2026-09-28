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