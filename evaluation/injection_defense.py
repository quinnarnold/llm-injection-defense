# evaluation/injection_defense.py
"""
Evaluate injection classifier precision, recall, F1 on held-out test set.

Usage:
    python -m evaluation.injection_defense --model models/classifier --test data/injection/test.jsonl
"""
import argparse
import json
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, recall_score
from src.classifier.model import InjectionClassifier


def evaluate_classifier(model_path: Path, test_path: Path, threshold: float = 0.85):
    classifier = InjectionClassifier(model_path, block_threshold=threshold)

    with open(test_path) as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    # Collect raw attack probabilities for threshold-independent analysis
    attack_probs = []
    y_true = []
    for item in test_data:
        result = classifier.predict(item["text"])
        y_true.append(item["label"])
        attack_probs.append(result.attack_prob)

    y_pred = [1 if p > threshold else 0 for p in attack_probs]

    print(classification_report(y_true, y_pred, target_names=["benign", "attack"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    attack_recall = recall_score(y_true, y_pred)
    benign_indices = [i for i, y in enumerate(y_true) if y == 0]
    false_positives = sum(1 for i in benign_indices if y_pred[i] == 1)
    fpr = false_positives / len(benign_indices) if benign_indices else 0

    print(f"\nAt threshold {threshold}:")
    print(f"  Attack recall: {attack_recall:.3f} (target: >=0.95)")
    print(f"  False positive rate: {fpr:.3f} (target: <=0.05)")
    passed = attack_recall >= 0.95 and fpr <= 0.05
    print(f"  PASS: {passed}")

    # ROC sweep over attack_prob directly — valid across the full [0,1] range
    print("\nROC threshold sweep:")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        y_sweep = [1 if p > t else 0 for p in attack_probs]
        r = recall_score(y_true, y_sweep)
        fp = sum(1 for i in benign_indices if y_sweep[i] == 1)
        fp_rate = fp / len(benign_indices) if benign_indices else 0
        marker = " <-- current" if t == threshold else ""
        print(f"  t={t:.2f}: recall={r:.3f}, FPR={fp_rate:.3f}{marker}")

    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()
    evaluate_classifier(args.model, args.test, args.threshold)
