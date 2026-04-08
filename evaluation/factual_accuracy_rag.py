# evaluation/factual_accuracy_rag.py
"""
Evaluate factual accuracy through the full pipeline (with RAG).

Usage: python -m evaluation.factual_accuracy_rag --url http://localhost:8080
"""
import argparse
import json
from pathlib import Path
import httpx


def evaluate(url: str, input_path: Path, output_path: Path):
    with open(input_path) as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    done = 0
    if output_path.exists():
        with open(output_path) as f:
            done = sum(1 for line in f if line.strip())
        print(f"Resuming from {done}/{len(test_data)}")

    for i, item in enumerate(test_data):
        if i < done:
            continue
        try:
            resp = httpx.post(
                f"{url}/chat",
                json={"message": item["question"]},
                timeout=300,
            )
            resp.raise_for_status()
            answer = resp.json()["response"]
        except Exception as e:
            answer = f"ERROR: {e}"

        result = {
            "question": item["question"],
            "ground_truth": item.get("ground_truth") or item.get("answer", ""),
            "generated": answer,
        }
        with open(output_path, "a") as f:
            f.write(json.dumps(result) + "\n")

        print(f"[{i+1}/{len(test_data)}] Q: {item['question'][:60]}...")
        print(f"  A: {answer[:80]}...")

    print(f"\nResults saved to {output_path}")
    print(f"{len(test_data)} questions, done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--input", type=Path, default=Path("evaluation/test_questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/factual_results_rag.jsonl"))
    args = parser.parse_args()
    evaluate(args.url, args.input, args.output)
