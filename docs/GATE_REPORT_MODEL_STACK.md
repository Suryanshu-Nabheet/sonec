# Gate — sonec on Qwen 3.5 (2B)

**Product:** LoRA adapter `sonec` (not a Modelfile SYSTEM string)  
**Base:** `Qwen/Qwen3.5-2B` / `mlx-community/Qwen3.5-2B-4bit` (Apache-2.0)  
**Code:** MIT — LICENSE + NOTICE

## Pipeline

1. TrainBench graded rollouts (training-only ids)
2. Gold agentic curriculum (localize → patch → verify)
3. MLX LoRA SFT
4. RL rejection / group winners
5. Serve with `sonec serve-llm` (base + adapter)

## Commands

```bash
sonec train --step --sft-iters 80 --gold-n 40 --train-n 16
sonec weights
sonec serve-llm --port 8080
# base-only on :8081 for A/B
python -m mlx_lm server --model mlx-community/Qwen3.5-2B-4bit --port 8081
sonec compare --out docs/results
```

## Proof artifacts (committed)

- `docs/results/TRAIN_PROOF.md` — what counts as specialization
- `docs/results/COMPARE_REPORT.md` — live LoRA vs base, same harness

## Contamination rule

Never train on sealed SonecBench / WorldBench task ids.
