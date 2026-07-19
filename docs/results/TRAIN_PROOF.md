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
| Method | MLX LoRA SFT |
| Corpus | 508 examples, OpenAI structured `tool_calls` (oracle-graded + gold); XML text dumps rejected |
| Iters | 500 |
| SFT wall time | ~1082 s |
| Adapter | `adapters.safetensors` + checkpoints through `0000480_*` |

Base weight lineage for redistribution: [NOTICE](../../NOTICE).

## Live A/B (after tool-argument wire fix)

See [COMPARE_REPORT.md](COMPARE_REPORT.md). Protocol fix required `arguments` as a JSON string on the OpenAI wire.

**Latest sealed result (`ab_agent_v1`):** sonec LoRA **4/6 (67%)** vs base **3/6 (50%)** — **+17%** pass rate. Decisive delta: `verify-script`. Promote further adapters only when pass rate still exceeds base.

## Reproduce

```bash
pip install -e ".[dev,train]"
./scripts/overnight_specialize.sh
# or:
sonec train --step --mock-fuel --sft-iters 500 --gold-n 240 --train-n 64 --rollout-group 4
sonec weights
sonec serve-llm --port 8080
```
