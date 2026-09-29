import os 
import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

deepseek = OpenAI(
    api_key = os.environ["DEEPSEEK_API"],
    base_url = "https://api.deepseek.com",
)

def call_fast_model(query:str) -> str:
    response = deepseek.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "user", "content": query}],
        extra_body={
            "thinking": {"type": "disabled"},
            "reasoning_effort": "low",
        },
    )
    return response.choices[0].message.content

def get_n_candidates(query: str, n: int = 3) -> list[str]:
    """Send the same query to the fast model n times in parallel. """
    with ThreadPoolExecutor(max_workers = n) as pool:
        futures = [pool.submit(call_fast_model, query) for _ in range(n)]
        return [f.result() for f in futures]

if __name__ == "__main__":
    query = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did natalia sell altogether in April and May?"
    n = 3

    #Parallel Timing
    start = time.time()
    candidates = get_n_candidates(query, n=n)
    parallel_time = time.time() - start

    #Sequential timing, for comparison
    start = time.time()
    for _ in range(n):
        call_fast_model(query)
    sequential_time = time.time() - start

    print(f"Got {len(candidates)} candidates")
    for i, c in enumerate(candidates):
        print(f"--- candidate {i} ---\n{c}\n")

    print(f"Parallel time:   {parallel_time:.2f}s")
    print(f"Sequential time: {sequential_time:.2f}s")
    assert len(candidates) == n
    assert parallel_time < sequential_time