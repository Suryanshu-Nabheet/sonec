# Training proof — specialized LoRA weights

Raw `*.safetensors` are gitignored. Reproduce with `sonec train --step`.

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
| Method | MLX LoRA SFT + rejection filtering |
| Corpus | Live harness rollouts + oracle-graded gold · OpenAI structured `tool_calls` (XML / “Calling tool” text dumps rejected) |
| Checkpoints | `adapters.safetensors` through `0000480_*` |
| Manifest | `artifacts/train/PRODUCT.json` |

Base weight lineage: [NOTICE](../../NOTICE) (Apache-2.0 · Qwen 3.5).

## Weight-level proof

Gold-trajectory mean token NLL (n=8 probe) — lower is better:

| Model | NLL |
| --- | ---: |
| Base `Qwen3.5-2B-4bit` | 2.159 |
| sonec LoRA | **0.022** |

Source: [SFT_METRICS.json](SFT_METRICS.json). This shows adapter fit to agent trajectories; it is not a substitute for live A/B.

## Live A/B

See [COMPARE_REPORT.md](COMPARE_REPORT.md).

**Latest sealed result (`ab_agent_v1`):** sonec LoRA **4/6 (67%)** vs base **3/6 (50%)** — **+17%** pass rate. Decisive delta: `verify-script`. Still failing both arms: `py-util-main`, `pkg-greet`.

Promote further adapters only when pass rate still exceeds base with no restraint regression.

## Reproduce

```bash
pip install -e ".[dev,train]"
sonec train --step --live-fuel --sft-iters 300 --gold-n 0 --train-n 40
sonec weights
sonec serve-llm --port 8080
# optional A/B base:
python -m mlx_lm server --model mlx-community/Qwen3.5-2B-4bit --port 8081
sonec compare --suite examples/benchmarks/ab_agent_v1.json --out docs/results
```
