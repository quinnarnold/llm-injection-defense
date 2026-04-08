"""
Fine-tune RoBERTa-base on the injection detection dataset.

Usage:
    python -m src.classifier.train \
        --data data/injection/train.jsonl \
        --output models/classifier \
        --epochs 5 \
        --batch-size 16 \
        --lr 2e-5
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)


class InjectionDataset(Dataset):
    def __init__(self, data: list[dict], tokenizer: RobertaTokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(item["label"], dtype=torch.long),
        }


def train(
    train_path: Path,
    output_path: Path,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(train_path) as f:
        train_data = [json.loads(line) for line in f if line.strip()]

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2
    ).to(device)

    dataset = InjectionDataset(train_data, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        print(f"Epoch {epoch + 1}/{epochs} -- loss: {avg_loss:.4f}, accuracy: {accuracy:.4f}")

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/classifier"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    train(args.data, args.output, args.epochs, args.batch_size, args.lr)
