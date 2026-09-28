import os
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic

from fast_stage import get_n_candidates
from jev_stage import judge

load_dotenv()

sonnet = Anthropic(api_key = os.environ["ANTHROPIC_API"])

def call_sonnet(query: str) -> str:
    response = sonnet.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=512,
        messages=[{"role": "user", "content": query}]
    )

    return response.content[0].text

def cascade(query:str, n: int=3, threshold: float = 0.8) -> dict:
    """
    Run the fast stage + Jev Judge, escalate to Sonnet if confidence is below threshold.
    """
    candidates = get_n_candidates(query, n=3)
    pick, confidence = judge(query, candidates)

    if confidence >= threshold:
        return {"answer": pick, "escalated": False, "confidence": confidence}
    else:
        answer = call_sonnet(query)
        return {"answer": answer, "escalated": True, "confidence": confidence}

if __name__ == "__main__":
    query = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    result_low = cascade(query, threshold = 0.0)
    assert result_low["escalated"] is False, "threshold 0 should never escalate"

    # threshold 1 should always escalate (confidence is essentially never exactly 1.0... but even if it is, >= 1.0 must escalate on anything less)
    result_high = cascade(query, threshold=1.01)
    assert result_high["escalated"] is True, "threshold above max possible confidence should always escalate"

    print("threshold=0.0:", result_low)
    print("threshold=1.01:", result_high)
