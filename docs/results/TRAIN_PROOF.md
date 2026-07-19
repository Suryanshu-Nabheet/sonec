# Training proof — specialized LoRA weights

Raw `*.safetensors` are gitignored. Reproduce with `sonec train --step` or `./scripts/overnight_specialize.sh`.

## What counts as sonec

| Claim | Status |
| --- | --- |
| Chat Modelfile only | Incomplete — runner without specialized weights |
| MLX LoRA under `artifacts/train/checkpoints/sonec-sft-mlx` | Product |
| Served via `sonec serve-llm` | Product |

Check: `sonec weights` → `ready=True`.

## Latest specialization (2026-07-19)

| Field | Value |
| --- | --- |
| Product | sonec by Suryanshu Nabheet — coding model |
| Method | MLX LoRA SFT + rejection RFT (+ optional GRPO-lite) |
| Corpus | TrainBench curriculum (incl. py-util / pkg-greet) + oracle-graded gold |
| Checkpoints | `adapters.safetensors` |
| Manifest | `artifacts/train/PRODUCT.json` |

Base weight lineage: [NOTICE](../../NOTICE) (Apache-2.0 · Qwen 3.5).

## Weight-level proof

Gold-trajectory mean token NLL (n=8 probe) — lower is better:

| Model | NLL |
| --- | ---: |
| Base `Qwen3.5-2B-4bit` | 2.159 |
| sonec LoRA | **0.022** |

Source: [SFT_METRICS.json](SFT_METRICS.json). Adapter fit ≠ sealed agent ranking.

## Live A/B + multi-model board

- A/B: [COMPARE_REPORT.md](COMPARE_REPORT.md) — `ab_agent_v1` saturated at **100%** for sonec and base after write-first harness fix.
- Multi-model: [leaderboard_2b/LEADERBOARD.md](leaderboard_2b/LEADERBOARD.md) — sonec preferred on ties (`kind=lora`).

Promote further adapters only when pass rate holds or improves with no restraint regression. Use sealed SonecBench / WorldBench for differentiation once `ab_agent_v1` is saturated.

## Reproduce

```bash
pip install -e ".[dev,train]"
./scripts/overnight_specialize.sh
# multi-model board (resume-safe pulls + eval):
./scripts/world_rl_leaderboard.sh
# board only:
SKIP_GRPO=1 ./scripts/world_rl_leaderboard.sh
```
