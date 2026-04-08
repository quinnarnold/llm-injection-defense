"""
DEPRECATED: AutoAWQ is incompatible with Blackwell (RTX 5090, sm_120).
Serve the LoRA adapter directly with vLLM instead:
    vllm serve Qwen/Qwen3-32B --enable-lora --lora-modules bryant=models/qwen3-32b-bryant-lora \
        --quantization bitsandbytes --load-format bitsandbytes

Original purpose: Merge LoRA adapters into base model and export AWQ-quantized weights.
"""
import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq import AutoAWQForCausalLM


def merge_and_quantize(
    base_model: str,
    adapter_path: Path,
    output_path: Path,
    awq_bits: int = 4,
    awq_group_size: int = 128,
):
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print("Loading and merging LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()

    merged_path = output_path.parent / "merged_tmp"
    merged_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))
    del model
    torch.cuda.empty_cache()

    print("Quantizing to AWQ INT4...")
    quant_model = AutoAWQForCausalLM.from_pretrained(str(merged_path))
    quant_config = {
        "zero_point": True,
        "q_group_size": awq_group_size,
        "w_bit": awq_bits,
        "version": "GEMM",
    }
    quant_model.quantize(tokenizer, quant_config=quant_config)

    output_path.mkdir(parents=True, exist_ok=True)
    quant_model.save_quantized(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"AWQ model saved to {output_path}")

    shutil.rmtree(merged_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/qwen3-32b-bryant"))
    args = parser.parse_args()
    merge_and_quantize(args.base_model, args.adapter, args.output)
