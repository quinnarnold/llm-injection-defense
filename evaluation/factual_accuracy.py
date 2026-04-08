# evaluation/factual_accuracy.py
"""
Evaluate factual accuracy on held-out Q&A set.

Usage: python -m evaluation.factual_accuracy --url http://localhost:8080 --test evaluation/test_questions.jsonl
"""
import argparse
import asyncio
import json
from pathlib import Path

import httpx

CONCURRENCY = 5


async def _fetch_answer(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    url: str,
    item: dict,
) -> dict:
    async with sem:
        resp = await client.post(f"{url}/chat", json={"message": item["question"]}, timeout=120)
        resp.raise_for_status()
        answer = resp.json()["response"]
    print(f"Q: {item['question'][:60]}...")
    print(f"A: {answer[:80]}...")
    print()
    return {
        "question": item["question"],
        "ground_truth": item["answer"],
        "generated": answer,
    }


async def evaluate_accuracy(url: str, test_path: Path) -> None:
    with open(test_path) as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient() as client:
        tasks = [_fetch_answer(client, sem, url, item) for item in test_data]
        results = await asyncio.gather(*tasks)

    out_path = test_path.parent / "eval_results.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Results saved to {out_path}")
    print(f"Total questions: {len(results)}")
    print("Review results manually or score with RAGAS.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--test", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(evaluate_accuracy(args.url, args.test))
