# src/finetune/train_qlora.py
"""
QLoRA fine-tuning of Qwen3-32B using Unsloth for 2x speed and 70% less VRAM.

Usage:
    python -m src.finetune.train_qlora \
        --data data/instruction/train.jsonl \
        --output models/qwen3-32b-bryant-lora \
        --base-model Qwen/Qwen3-32B
"""
import argparse
import json
from pathlib import Path

import unsloth  # noqa: F401
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig


def load_and_format(path: Path, tokenizer) -> Dataset:
    """Load ChatML JSONL and apply chat template to produce a 'text' field."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                text = tokenizer.apply_chat_template(
                    item["messages"], tokenize=False, add_generation_prompt=False,
                )
                records.append({"text": text})
    return Dataset.from_list(records)


def train(
    data_path: Path,
    output_path: Path,
    base_model: str = "Qwen/Qwen3-32B",
    epochs: int = 3,
    batch_size: int = 1,
    grad_accum: int = 16,
    lr: float = 2e-4,
    max_seq_length: int = 1024,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
    )

    dataset = load_and_format(data_path, tokenizer)
    print(f"Loaded {len(dataset)} training examples")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            output_dir=str(output_path),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            optim="adamw_8bit",
            weight_decay=0.001,
            max_grad_norm=1.0,
            seed=3407,
            report_to="none",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train()
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"LoRA adapter saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/qwen3-32b-bryant-lora"))
    parser.add_argument("--base-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    args = parser.parse_args()
    train(args.data, args.output, args.base_model, args.epochs, lr=args.lr,
          lora_r=args.lora_r, lora_alpha=args.lora_alpha, max_seq_length=args.max_seq_length)
