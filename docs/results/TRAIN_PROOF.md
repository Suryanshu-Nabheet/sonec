# Training proof — specialized LoRA weights

Raw `*.safetensors` are gitignored. Reproduce with `sonec train --step`.

## What counts as sonec

| Claim | Status |
| --- | --- |
| Chat Modelfile only | Incomplete — runner without specialized weights |
| MLX LoRA under `artifacts/train/checkpoints/sonec-sft-mlx` | Product |
| Served via `sonec serve-llm` | Product |

Check: `sonec weights` → `ready=True`.

## Specialization recorded here

| Field | Value |
| --- | --- |
| Product | sonec by Suryanshu Nabheet — coding model |
| Method | MLX LoRA SFT |
| Iters | 160 |
| Val loss | 1.770 → **0.016** |
| Train loss (final) | **0.019** |
| Adapter files | `adapters.safetensors`, `0000080_*`, `0000160_*` |

Base weight lineage for redistribution: [NOTICE](../../NOTICE).

## Reproduce

```bash
pip install -e ".[dev,train]"
sonec train --step --live-fuel --sft-iters 300 --gold-n 0
sonec weights
sonec serve-llm --port 8080
```

## Live A/B

See [COMPARE_REPORT.md](COMPARE_REPORT.md) — same frozen harness, LoRA vs base endpoint.
