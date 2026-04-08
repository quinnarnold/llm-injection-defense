"""
Dataset loading and preparation for the injection classifier.

Loads JSONL files with {text, label} format, deduplicates, and creates
stratified train/test splits.
"""
import json
import random
from pathlib import Path


def load_injection_dataset(paths: list[Path]) -> list[dict]:
    """Load and deduplicate injection dataset from one or more JSONL files."""
    seen_texts: set[str] = set()
    dataset: list[dict] = []

    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                text = item["text"].strip()
                if text not in seen_texts:
                    seen_texts.add(text)
                    dataset.append({"text": text, "label": item["label"]})

    return dataset


def create_train_test_split(
    dataset: list[dict],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Stratified train/test split preserving label balance."""
    rng = random.Random(seed)
    positives = [item for item in dataset if item["label"] == 1]
    negatives = [item for item in dataset if item["label"] == 0]

    rng.shuffle(positives)
    rng.shuffle(negatives)

    n_pos_test = max(1, int(len(positives) * test_ratio))
    n_neg_test = max(1, int(len(negatives) * test_ratio))

    test = positives[:n_pos_test] + negatives[:n_neg_test]
    train = positives[n_pos_test:] + negatives[n_neg_test:]

    rng.shuffle(train)
    rng.shuffle(test)

    return train, test
