"""
RoBERTa-based injection classifier.

Wraps a fine-tuned RoBERTa-base model with a binary classification head.
"""
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification


@dataclass
class ClassifierResult:
    label: int
    confidence: float
    attack_prob: float
    blocked: bool


class InjectionClassifier:
    def __init__(
        self,
        model_path: str | Path,
        block_threshold: float = 0.85,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.block_threshold = block_threshold
        self.device = device
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_path, num_labels=2
        ).to(device)
        self.model.eval()

    def predict(self, text: str) -> ClassifierResult:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            label = int(probs.argmax(dim=-1).item())
            confidence = float(probs[0, label].item())
            attack_prob = float(probs[0, 1].item())

        return ClassifierResult(
            label=label,
            confidence=confidence,
            attack_prob=attack_prob,
            blocked=(attack_prob > self.block_threshold),
        )
