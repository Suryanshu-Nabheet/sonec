# sonec RL — GRPO-lite (Apple Silicon / MLX)

## Laptop-safe defaults (important)

Large live GRPO (G=8 × n=24 × live agents + 12-layer LoRA) will thrash / OOM Macs.
**Defaults are intentionally light and mock.**

```bash
# Safe densify (default) — oracle trajectories, G=2, n=8, 80 iters
sonec grpo --mock

# Explicit light live (only if serve-llm is up and you accept the cost)
sonec grpo --live --group-size 2 --train-n 8 --sft-iters 80

# Refused automatically: --live with G>4 or train_n>16
```

World board script keeps GRPO **off** unless you opt in:

```bash
SKIP_GRPO=1 ./scripts/world_rl_leaderboard.sh   # default
SKIP_GRPO=0 ./scripts/world_rl_leaderboard.sh   # runs light --mock GRPO only
LIVE_GRPO=1 SKIP_GRPO=0 ./scripts/world_rl_leaderboard.sh  # live (still capped)
```

## What it does

Implementation: `sonec/training/grpo_lite.py`

- Sample G rollouts per TrainBench prompt (same harness / graders)
- Advantage = reward − group mean (Dr.GRPO-style)
- Densify positive-advantage trajectories → continue MLX LoRA (8 layers by default)

This is **not** CUDA TRL/verl policy-gradient GRPO. Same *relative* signal on MLX.

## Decision eval (not GRPO)

Use sealed **CapabilityBench** (200 tasks) to see where sonec actually stands:

```bash
sonec capabilitybench   # writes examples/benchmarks/capabilitybench_v1.json
sonec leaderboard -s examples/benchmarks/capabilitybench_v1.json -o docs/results/leaderboard_2b
```

## Pinning

`artifacts/train/grpo_lite/grpo_stats.json` records G, n, live/mock, corpus size, SFT report.
