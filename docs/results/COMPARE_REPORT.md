# sonec vs base — evidence

## 1. Weight specialization (primary ML proof)

Gold-trajectory probe NLL (lower is better) — same Qwen 3.5 2B base, LoRA adapter only:

| Model | Mean token NLL |
| --- | --- |
| Base `Qwen3.5-2B-4bit` | **2.159** |
| **sonec LoRA** | **0.022** |
| Improvement | **−2.137 NLL** |

Source: [`SFT_METRICS.json`](SFT_METRICS.json). Training: 160 MLX LoRA iters, val loss 1.77 → 0.016. See [`TRAIN_PROOF.md`](TRAIN_PROOF.md).

A Modelfile / system prompt cannot produce this NLL gap.

## 2. Live agent A/B (same harness)

Suite: `examples/benchmarks/ab_agent_v1.json`  
Endpoints: LoRA `:8080` vs base-only `:8081` · focused tool allowlist · fresh workspace per task.

| Arm | Kind | Pass rate | Passed | Mean score | Mean duration |
| --- | --- | --- | --- | --- | --- |
| sonec_lora | lora | 50% | 3/6 | 0.50 | 9.9s |
| qwen35_2b_base | base | 50% | 3/6 | 0.50 | 8.8s |

**Pass-rate winner:** tie on this 6-task slice (2 tasks hit intermittent mlx_lm 404s on both arms).  
Specialize further with `sonec train --step` and re-run `sonec compare`.

## Commands

```bash
sonec weights
sonec serve-llm --port 8080
python -m mlx_lm server --model mlx-community/Qwen3.5-2B-4bit --port 8081
sonec compare --out docs/results
```
