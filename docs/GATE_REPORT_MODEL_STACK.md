# Gate — sonec specialization

**Product:** LoRA adapter `sonec` by Suryanshu Nabheet — coding model  
**Weights:** `artifacts/train/checkpoints/sonec-sft-mlx`  
**License:** Apache-2.0 — LICENSE · lineage — NOTICE

## Latest smoke gate (2026-07-19)

| Check | Result |
| --- | --- |
| MLX compare (`ab_agent_2b_hard`) | sonec 8/8 @ 8.6s · base 8/8 @ 16.5s (pass tie; LoRA ~1.9× faster) |
| 2B board | **Winner sonec** 8/8 @ 8.5s · qwen3.5:2b 8/8 @ 11.5s · gemma2/codegemma 0/8 |
| Peers | Strict ~2B only (`arms_2b.json`) |
| Next gate | CapabilityBench 200 (`capabilitybench_v1.json`) |

Reports: [COMPARE_REPORT.md](results/COMPARE_REPORT.md) · [LEADERBOARD.md](results/leaderboard_2b/LEADERBOARD.md) · [TRAIN_PROOF.md](results/TRAIN_PROOF.md)

## Pipeline

1. TrainBench graded rollouts (training-only ids)
2. Verified trajectories (path-correct writes, verify, restraint)
3. MLX LoRA SFT
4. Rejection sampling → **second SFT (RFT)**
5. Optional **light GRPO-lite** (`sonec grpo --mock`) — never heavy live on a laptop
6. Serve with `sonec serve-llm`
7. Promote via `sonec compare` + `sonec leaderboard` (CapabilityBench primary)

## Commands

```bash
./scripts/overnight_specialize.sh
SKIP_GRPO=1 ./scripts/world_rl_leaderboard.sh

sonec capabilitybench
sonec leaderboard -s examples/benchmarks/capabilitybench_v1.json \
  -a configs/leaderboard/arms_2b.json -o docs/results/leaderboard_2b
sonec grpo --mock
```

Promote only when CapabilityBench holds or improves without restraint regression.

## Strict 2B board

- Catalog: `configs/leaderboard/arms_2b.json`
- Decision suite: `examples/benchmarks/capabilitybench_v1.json` (200 sealed)
- Smoke: `examples/benchmarks/ab_agent_2b_hard.json`
- GRPO-lite: `configs/rl/grpo_recipe.md` · `sonec grpo --mock`
