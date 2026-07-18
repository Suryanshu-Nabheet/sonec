# Training proof — sonec is LoRA weights, not a prompt

Raw `*.safetensors` are gitignored (binaries). Reproduce with `sonec train --step`.

## What counts as sonec

| Claim | Valid? |
| --- | --- |
| Modelfile `SYSTEM` on `qwen3.5:2b` | No — prompt wrapper |
| Ollama/local tag from Modelfile only | No |
| MLX LoRA `*.safetensors` under `artifacts/train/checkpoints/sonec-sft-mlx` | **Yes** |
| Served via `sonec serve-llm` (base + adapter) | **Yes** |

Check: `sonec weights` → `ready=True`.

## Specialization recorded here

| Field | Value |
| --- | --- |
| Base | `mlx-community/Qwen3.5-2B-4bit` (Qwen 3.5 2B, Apache-2.0) |
| Method | MLX LoRA SFT |
| Iters | 160 |
| Val loss | 1.770 → **0.016** |
| Train loss (final) | **0.019** |
| Curriculum | Gold agent trajectories with native Qwen `<tool_call>` markup (not `"Calling tool"` text) |
| Adapter files | `adapters.safetensors`, `0000080_*`, `0000160_*` |

## Reproduce

```bash
pip install -e ".[dev,train]"
sonec train --step --sft-iters 160 --gold-n 96 --train-n 12
sonec weights
sonec serve-llm --port 8080
```

## Live A/B

See [COMPARE_REPORT.md](COMPARE_REPORT.md) — same frozen harness, LoRA `:8080` vs base-only `:8081`.
