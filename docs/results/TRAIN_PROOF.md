# Training proof — sonec is LoRA weights, not a prompt

This file is the **committed** record of specialization work. Raw `*.safetensors`
stay local/gitignored (large binaries). Anyone can reproduce with `sonec train --step`.

## What counts as sonec

| Claim | Valid? |
| --- | --- |
| Modelfile `SYSTEM` on `qwen3.5:2b` | No — prompt wrapper |
| Ollama tag named `sonec` from Modelfile only | No |
| MLX LoRA `*.safetensors` under `artifacts/train/checkpoints/sonec-sft-mlx` | **Yes** |
| Served via `sonec serve-llm` (base + adapter) | **Yes** |

Check locally: `sonec weights` must print `ready=True`.

## Specialization step (recorded)

- **Base:** `mlx-community/Qwen3.5-2B-4bit` (Qwen 3.5 2B lineage, Apache-2.0)
- **Method:** MLX LoRA SFT + rejection RL on TrainBench / gold curriculum
- **SFT:** 80 iterations, batch 1, 8 LoRA layers, lr `1e-5`, max seq 2048
- **Artifacts (local):** `adapters.safetensors`, mid-checkpoints `0000040_*`, `0000080_*`
- **RL rejection:** 8/8 winner groups on mock-graded TrainBench (see `artifacts/train/rl/rl_stats.json`)
- **Contamination:** sealed SonecBench / WorldBench ids excluded from fuel

## Reproduce

```bash
pip install -e ".[dev,train]"
sonec train --step --sft-iters 80 --gold-n 40 --train-n 16
sonec weights
sonec serve-llm --port 8080
```

## Live A/B

See [COMPARE_REPORT.md](COMPARE_REPORT.md) — same frozen harness, LoRA endpoint vs base-only endpoint.
