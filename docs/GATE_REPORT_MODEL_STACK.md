# Gate — sonec specialization

**Product:** LoRA adapter `sonec` by Suryanshu Nabheet  
**Weights:** `artifacts/train/checkpoints/sonec-sft-mlx`  
**License:** Apache-2.0 — LICENSE · NOTICE

## Published smoke gate (2026-07-19)

| Check | Result |
| --- | --- |
| MLX compare (`ab_agent_2b_hard`) | sonec **8/8 @ 8.6s** · base 8/8 @ 16.5s (~1.9× faster) |
| 2B board | **Winner sonec** 8/8 @ 8.5s · qwen3.5:2b 8/8 @ 11.5s · gemma2/codegemma 0/8 |
| Peers | Strict ~2B only (`configs/leaderboard/arms_2b.json`) |
| Primary decision suite | CapabilityBench 200 (`capabilitybench_v1.json`) — run when you have hours |

Reports: [COMPARE_REPORT.md](results/COMPARE_REPORT.md) · [LEADERBOARD.md](results/leaderboard_2b/LEADERBOARD.md) · [TRAIN_PROOF.md](results/TRAIN_PROOF.md)

## Pipeline

1. TrainBench graded rollouts (training-only ids; never sealed Cap/Sonec/World ids)
2. Verified trajectories (writes, verify, restraint)
3. MLX LoRA SFT
4. Rejection sampling → second SFT (RFT)
5. Optional light GRPO-lite (`sonec grpo --mock`) — never heavy live on a laptop
6. `sonec serve-llm`
7. Promote via smoke compare, then CapabilityBench when discriminating scores are needed

## Commands

```bash
# Specialize
./scripts/overnight_specialize.sh

# Smoke board (minutes)
SKIP_GRPO=1 SUITE=examples/benchmarks/ab_agent_2b_hard.json \
  ./scripts/world_rl_leaderboard.sh

# CapabilityBench 200 (hours)
SKIP_SFT=1 ./scripts/capabilitybench_e2e.sh
```

Promote only when sealed pass rate holds or improves without restraint regression.

## Harness notes (production)

- Small-file `fs_read` returns **raw** text (no `N|` line prefixes) so `fs_edit` / `fs_write` work.
- Restraint tasks use `only_files` grading — extra files fail the task.
- Live GRPO with `G>4` or `train_n>16` is refused.
