# Gate — sonec specialization

**Product:** LoRA adapter `sonec` by Suryanshu Nabheet — coding model  
**Weights:** `artifacts/train/checkpoints/sonec-sft-mlx`  
**License:** Apache-2.0 — LICENSE · lineage — NOTICE

## Pipeline

1. TrainBench graded rollouts (training-only ids) — includes py-util + pkg/greet curriculum
2. Verified trajectories (path-correct writes, verify, restraint)
3. MLX LoRA SFT
4. Rejection sampling → **second SFT (RFT)**
5. Optional **GRPO-lite** (`sonec grpo`)
6. Serve with `sonec serve-llm`
7. Promote via `sonec compare` + multi-model `sonec leaderboard`

## Commands

```bash
# Canonical overnight specialize + A/B
./scripts/overnight_specialize.sh

# Multi-model 2B board (+ optional GRPO)
./scripts/world_rl_leaderboard.sh
# SKIP_GRPO=1 ./scripts/world_rl_leaderboard.sh   # board only

# Manual
sonec train --step --mock-fuel --sft-iters 500 --gold-n 240 --train-n 100
sonec weights && sonec serve-llm --port 8080
sonec compare --out docs/results
sonec grpo --live --group-size 8 --train-n 24
sonec leaderboard -a configs/leaderboard/arms_2b.json -o docs/results/leaderboard_2b
```

Promote an adapter only when pass rate holds or improves without restraint regression.

## Strict 2B board

- Catalog: **only** ~2B models (`configs/leaderboard/arms_2b.json`)
- Decision suite: `examples/benchmarks/ab_agent_2b_hard.json`
- Results: `docs/results/leaderboard_2b/LEADERBOARD.md`
- GRPO-lite: `configs/rl/grpo_recipe.md` · `sonec grpo`
