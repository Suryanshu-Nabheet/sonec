# sonec RL — GRPO-lite (Apple Silicon / MLX)

## What ships

```bash
# Group-relative rollouts → advantage-weighted LoRA SFT
sonec grpo --group-size 8 --train-n 24 --sft-iters 200 --live
# Needs: sonec serve-llm on :8080 and ready adapters (`sonec weights`)

# Offline densify (oracle tool_calls) when live inference is unavailable:
sonec grpo --mock --group-size 6 --train-n 16 --sft-iters 150
```

Implementation: `sonec/training/grpo_lite.py`

- Sample G rollouts per TrainBench prompt (same harness / graders)
- Advantage = reward − group mean (Dr.GRPO-style; no length-std)
- Densify positive-advantage trajectories → continue MLX LoRA

This is **not** CUDA TRL/verl policy-gradient GRPO. It is the same *relative* training signal on MLX. For full GRPO on GPU clusters, see the external stacks below — keep sonec as the rollout env.

## Data export (optional)

```bash
sonec rollout --live -g 8 --limit 20 --out artifacts/rollouts/live
sonec train --export -r artifacts/rollouts/live/rollouts.jsonl -o artifacts/train
# artifacts/train/grpo_prompts.jsonl
```

## External stacks (CUDA)

| Stack | Notes |
| --- | --- |
| [verl](https://github.com/volcengine/verl) | Production GRPO; wire sonec rollout worker as env |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | Ray + vLLM rollouts |
| [TRL GRPO](https://huggingface.co/docs/trl) | Smaller local experiments |

## Pinning

Every RL run records in `artifacts/train/grpo_lite/grpo_stats.json`:

- group size, prompt count, passers, corpus lines
- base mlx model id, live vs mock
- SFT report

Harness or tool-schema change requires a migration eval before new training.

## Collapse watches

- Terminal-only / list-only loops
- Entropy death (identical rollouts)
- pass@1 up but pass@k flat
