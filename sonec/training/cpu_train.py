#!/usr/bin/env python3
"""CPU LoRA SFT for sonec — zero-GPU environments.

Trains a real PEFT adapter with transformers+peft (no CUDA required).
Default base is a small Qwen that fits ~16GB RAM; override with --model.
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
    parser = argparse.ArgumentParser(description="sonec CPU PEFT LoRA trainer")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    args = parser.parse_args()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Missing CPU train deps. pip install torch transformers peft datasets accelerate trl"
        ) from exc

    args.out.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(args.data)
    print(f"rows={len(rows)} model={args.model} device=cpu steps={args.steps}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.gradient_checkpointing_enable()
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = Dataset.from_list(rows)

    def formatting_func(example: dict) -> str:
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    # TRL API differs across versions — support both.
    train_args = TrainingArguments(
        output_dir=str(args.out / "hf_trainer"),
        max_steps=max(1, args.steps),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="no",
        bf16=False,
        fp16=False,
        optim="adamw_torch",
        seed=3407,
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    try:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            formatting_func=formatting_func,
            args=train_args,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            formatting_func=formatting_func,
            args=train_args,
            max_seq_length=args.max_seq_length,
        )

    trainer.train()
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    cfg = args.out / "adapter_config.json"
    if not cfg.exists():
        cfg.write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "base_model_name_or_path": args.model,
                    "product": "sonec",
                    "author": "Suryanshu Nabheet",
                    "backend": "cpu",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    meta = {
        "product": "sonec",
        "author": "Suryanshu Nabheet",
        "backend": "cpu",
        "base_model": args.model,
        "steps": args.steps,
        "rows": len(rows),
        "note": (
            "CPU LoRA adapter produced without GPU. "
            "Promote 2B product claims only after Cap200 on MLX/Unsloth weights."
        ),
    }
    (args.out / "SONEC_TRAIN.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved_adapter={args.out}")


if __name__ == "__main__":
    main()
