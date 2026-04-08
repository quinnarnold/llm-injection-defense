# evaluation/latency.py
"""
Measure end-to-end latency p50 and p95.

Usage: python -m evaluation.latency --url http://localhost:8080 --n 50
"""
import argparse
import time
import httpx
import numpy as np

SAMPLE_QUERIES = [
    "What time does the dining hall open?",
    "Where is the registrar's office?",
    "What majors does Bryant offer?",
    "How do I apply for financial aid?",
    "What are the library hours?",
]


def measure_latency(url: str, n: int = 50):
    latencies = []
    for i in range(n):
        query = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
        start = time.perf_counter()
        resp = httpx.post(f"{url}/chat", json={"message": query}, timeout=120)
        elapsed = time.perf_counter() - start
        resp.raise_for_status()
        latencies.append(elapsed)
        print(f"  [{i+1}/{n}] {elapsed:.2f}s")

    arr = np.array(latencies)
    print(f"\np50: {np.percentile(arr, 50):.2f}s")
    print(f"p95: {np.percentile(arr, 95):.2f}s")
    print(f"mean: {np.mean(arr):.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    measure_latency(args.url, args.n)
