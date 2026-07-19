#!/usr/bin/env python3
"""Unsloth QLoRA SFT for sonec on Linux CUDA.

Usage (via sonec train --backend unsloth, or directly):
  python -m sonec.training.unsloth_train \\
    --model Qwen/Qwen3.5-2B \\
    --data artifacts/train/checkpoints/sonec-sft-unsloth/train_messages.jsonl \\
    --out artifacts/train/checkpoints/sonec-sft-unsloth \\
    --steps 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"empty dataset: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="sonec Unsloth QLoRA trainer")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    args = parser.parse_args()

    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit(
            "Unsloth not installed. On Linux CUDA: pip install 'sonec[train-cuda]'"
        ) from exc

    try:
        import torch
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Missing train deps (torch/datasets/trl). pip install 'sonec[train-cuda]'"
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Unsloth training")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(args.data)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = Dataset.from_list(rows)

    def formatting_func(examples: dict) -> list[str]:
        texts: list[str] = []
        messages_col = examples["messages"]
        for messages in messages_col:
            texts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return texts

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_func,
        args=SFTConfig(
            output_dir=str(args.out / "hf_trainer"),
            max_steps=max(1, args.steps),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            logging_steps=10,
            save_strategy="no",
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            seed=3407,
            report_to=[],
            max_seq_length=args.max_seq_length,
            dataset_text_field=None,
            packing=False,
        ),
    )
    trainer.train()
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    # Ensure adapter_config.json exists for sonec weights readiness.
    cfg = args.out / "adapter_config.json"
    if not cfg.exists():
        # PEFT save_pretrained should write it; if only adapter_model exists, synthesize minimal.
        peft = args.out / "adapter_model.safetensors"
        if peft.exists():
            cfg.write_text(
                json.dumps(
                    {
                        "peft_type": "LORA",
                        "base_model_name_or_path": args.model,
                        "product": "sonec",
                        "author": "Suryanshu Nabheet",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    print(f"saved_adapter={args.out}")


if __name__ == "__main__":
    main()
