"""
Build the injection classifier dataset from multiple sources.

Sources (in priority order):
  1. data/injection/audit_payloads.jsonl  — hand-crafted Bryant-specific payloads
  2. deepset/prompt-injections (HuggingFace) — ~550 general injection examples
  3. data/scraped/*.json                  — Bryant corpus for domain-specific benign examples

Outputs:
  data/injection/train.jsonl
  data/injection/test.jsonl

Usage:
    python -m src.classifier.build_dataset
    python -m src.classifier.build_dataset --max-hf 400 --max-bryant 200
"""
import argparse
import json
import re
import random
from pathlib import Path

from datasets import load_dataset

from src.classifier.dataset import load_injection_dataset, create_train_test_split


def load_hf_examples(max_per_class: int) -> list[dict]:
    ds = load_dataset("deepset/prompt-injections", split="train")
    attacks, benign = [], []
    for row in ds:
        text = row["text"].strip()
        if not text:
            continue
        if int(row["label"]) == 1:
            attacks.append({"text": text, "label": 1})
        else:
            benign.append({"text": text, "label": 0})
    random.shuffle(attacks)
    random.shuffle(benign)
    return attacks[:max_per_class] + benign[:max_per_class]


def load_bryant_benign(scraped_dir: Path, max_examples: int) -> list[dict]:
    sentences = []
    for path in scraped_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            text = data.get("text", "")
            for part in re.split(r'(?<=[.?!])\s+', text):
                part = part.strip()
                if 20 < len(part) < 200 and part[0].isupper():
                    sentences.append(part)
        except Exception:
            continue
    random.shuffle(sentences)
    return [{"text": s, "label": 0} for s in sentences[:max_examples]]


def build(
    audit_path: Path,
    scraped_dir: Path,
    output_dir: Path,
    max_hf: int,
    max_bryant: int,
    seed: int = 42,
):
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading audit payloads...")
    combined = load_injection_dataset([audit_path])

    print("Loading deepset/prompt-injections...")
    hf = load_hf_examples(max_hf)

    print(f"Sampling {max_bryant} benign sentences from Bryant corpus...")
    bryant = load_bryant_benign(scraped_dir, max_bryant)

    # Merge manually since load_injection_dataset takes paths, not lists
    seen = {item["text"].strip() for item in combined}
    for item in hf + bryant:
        text = item["text"].strip()
        if text not in seen:
            seen.add(text)
            combined.append(item)

    attacks = [r for r in combined if r["label"] == 1]
    benign  = [r for r in combined if r["label"] == 0]

    # Balance to smaller class
    n = min(len(attacks), len(benign))
    random.shuffle(attacks)
    random.shuffle(benign)
    combined = attacks[:n] + benign[:n]

    train, test = create_train_test_split(combined, seed=seed)

    def write(path, rows):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    write(output_dir / "train.jsonl", train)
    write(output_dir / "test.jsonl", test)

    print(f"train: {len(train)} ({sum(r['label']==1 for r in train)} attack, {sum(r['label']==0 for r in train)} benign)")
    print(f"test:  {len(test)}  ({sum(r['label']==1 for r in test)} attack, {sum(r['label']==0 for r in test)} benign)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=Path("data/injection/audit_payloads.jsonl"))
    parser.add_argument("--scraped", type=Path, default=Path("data/scraped"))
    parser.add_argument("--output", type=Path, default=Path("data/injection"))
    parser.add_argument("--max-hf", type=int, default=400)
    parser.add_argument("--max-bryant", type=int, default=200)
    args = parser.parse_args()
    build(args.audit, args.scraped, args.output, args.max_hf, args.max_bryant)
