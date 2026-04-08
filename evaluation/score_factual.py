# evaluation/score_factual.py
"""
LLM-as-judge scoring for factual accuracy and tone using Groq API.

Scores each (ground_truth, generated) pair on two dimensions:
  - Factual correctness (0/1/2)
  - Tone appropriateness (0/1/2)

Usage:
    export GROQ_API_KEY=...
    python evaluation/score_factual.py
    python evaluation/score_factual.py --input evaluation/factual_results.jsonl
"""
import argparse
import json
import re
import time
from pathlib import Path

from openai import OpenAI, RateLimitError

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "llama-3.3-70b-versatile"

FACTUAL_RUBRIC = """Score the generated answer against the ground truth on a 0-2 scale:
2 = All key facts correct and present
1 = Directionally correct but missing one or more specific facts, or has a minor error
0 = Factually wrong, contradicts ground truth, or hallucinates specific details

Ground truth: {ground_truth}
Generated answer: {generated}

Reply with ONLY a JSON object: {{"score": <0|1|2>, "reason": "<one sentence>"}}"""

TONE_RUBRIC = """You are evaluating a university campus chatbot called "Ask Tupper" for Bryant University.
Score the tone of the generated response on a 0-2 scale:
2 = Professional, helpful, concise; appropriate for a university campus assistant
1 = Acceptable but verbose, too casual, or slightly off-brand
0 = Robotic, evasive, inappropriate, or off-topic

Generated answer: {generated}

Reply with ONLY a JSON object: {{"score": <0|1|2>, "reason": "<one sentence>"}}"""


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def judge(client: OpenAI, prompt: str) -> dict:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=150,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content.strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
            score_match = re.search(r'"score"\s*:\s*(\d)', text)
            if score_match:
                reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
                return {
                    "score": int(score_match.group(1)),
                    "reason": reason_match.group(1) if reason_match else text[:100],
                }
            digit_match = re.search(r'\b([012])\b', text)
            if digit_match:
                return {"score": int(digit_match.group(1)), "reason": text[:100]}
            return {"score": -1, "reason": f"Parse error: {text[:100]}"}
        except RateLimitError:
            time.sleep(5 * (attempt + 1))
    return {"score": -1, "reason": "Rate limited after 3 retries"}


def score_results(input_path: Path, output_path: Path):
    import os
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise SystemExit("Set GROQ_API_KEY before running.")
    client = OpenAI(api_key=groq_key, base_url=GROQ_BASE_URL)

    with open(input_path) as f:
        results = [json.loads(line) for line in f if line.strip()]

    scored = []
    if output_path.exists():
        with open(output_path) as f:
            scored = [json.loads(line) for line in f if line.strip()]
        print(f"Resuming from {len(scored)}/{len(results)}")

    for i, item in enumerate(results):
        if i < len(scored):
            continue
        generated = strip_think_tags(item["generated"])
        ground_truth = item["ground_truth"]

        factual = judge(client, FACTUAL_RUBRIC.format(
            ground_truth=ground_truth, generated=generated,
        ))
        time.sleep(1.0)

        tone = judge(client, TONE_RUBRIC.format(generated=generated))
        time.sleep(1.0)

        scored.append({
            "question": item["question"],
            "factual_score": factual.get("score", -1),
            "factual_reason": factual.get("reason", ""),
            "tone_score": tone.get("score", -1),
            "tone_reason": tone.get("reason", ""),
        })
        print(f"[{i+1}/{len(results)}] factual={factual.get('score')} tone={tone.get('score')} | {item['question'][:60]}")

        with open(output_path, "a") as f:
            f.write(json.dumps(scored[-1]) + "\n")

    factual_scores = [s["factual_score"] for s in scored if s["factual_score"] >= 0]
    tone_scores = [s["tone_score"] for s in scored if s["tone_score"] >= 0]

    print("\n--- Factual Accuracy ---")
    print(f"  2 (correct):  {factual_scores.count(2)}/{len(factual_scores)}")
    print(f"  1 (partial):  {factual_scores.count(1)}/{len(factual_scores)}")
    print(f"  0 (wrong):    {factual_scores.count(0)}/{len(factual_scores)}")
    print(f"  Mean: {sum(factual_scores)/len(factual_scores):.2f}")

    print("\n--- Tone ---")
    print(f"  2 (good):     {tone_scores.count(2)}/{len(tone_scores)}")
    print(f"  1 (okay):     {tone_scores.count(1)}/{len(tone_scores)}")
    print(f"  0 (bad):      {tone_scores.count(0)}/{len(tone_scores)}")
    print(f"  Mean: {sum(tone_scores)/len(tone_scores):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("evaluation/factual_results.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/factual_scores.jsonl"))
    args = parser.parse_args()
    score_results(args.input, args.output)
