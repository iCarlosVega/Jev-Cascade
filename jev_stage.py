import os
from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, TypeSafeClient

from fast_stage import get_n_candidates

load_dotenv()

jev = TypeSafeClient(api_key=os.environ["TYPESAFE_API"])

def judge(query: str, candidates: list[str]) -> tuple[str, float]:
    """
    Score each candidate independently, then pick 
    the highest-scoring one.
    """

    scores = []
    for c in candidates:
        state = f"Query: {query}\n\nAnswer: {c}"
        response = jev.system_one(
            state=state,
            questions={
                "makes_sense": Noul(
                    instructions="Does this answer correctly and sensibly solve the query?",
                ),
            },
        )

        scores.append(response.answers["makes_sense"].noul)

    best_index = max(range(len(scores)), key=lambda i:scores[i])
    return candidates[best_index], scores[best_index]


if __name__ == "__main__":
    query = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    candiates = [
        "Natalia sold 48 + 24 = 72 clips altogether.",
        "Natalia sold 48 clips in total, ignoring May.",
        "Natalia sold 100 clips because 48 + 52 = 100.",
      ]
        
    best, confidence = judge(query, candiates)
    print(f"Jev Picked: {best!r}")
    print(f"Confidence: {confidence}")

    assert best == candiates[0]
    assert confidence > 0.7, f"Expected High confidence on an obvious pick, got {confidence}"

    real_candidates = get_n_candidates(query, n=3)
    real_best, real_confidence = judge(query, real_candidates)
    print(f"\nOn real candidates, Jev picked: {real_best!r}")
    print(f"Confidence: {real_confidence}")