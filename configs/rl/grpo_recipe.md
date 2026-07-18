# sonec RL — GRPO-family recipe (external trainer)

## Goal

Train a coding-specialist policy **inside the frozen sonec harness** so rewards
come from environment graders (WorldBench / private tasks), not model self-report.

## Algorithm

- Prefer GRPO / Dr.GRPO-style group relative policy gradient
- Group size G = 8–16 independent rollouts per prompt (`sonec rollout -g 8`)
- Single-epoch prompts where possible
- Avoid length-std / group-std pathologies that collapse tool use

## Data

```bash
# Live rollouts with the product model (never sealed eval ids)
sonec rollout --live -m sonec \
  --suite examples/benchmarks/smoke.json -g 8 --limit 20 \
  --out artifacts/rollouts/live

sonec train --export -r artifacts/rollouts/live/rollouts.jsonl -o artifacts/train
# Use artifacts/train/grpo_prompts.jsonl as prompt pool
```

## Reward

Primary: `reward = 1.0 if graded.passed else 0.0` (file/command checks)
Auxiliary (optional weights):
- skipped_verify penalty
- catastrophic tool misuse penalty
- nonlinear length penalty (lighter on easy tasks, allow tools on hard)

Credit assignment: final reward applies to all model tokens including self-summaries.

## External stacks

| Stack | Notes |
| --- | --- |
| [verl](https://github.com/volcengine/verl) | Production GRPO; wire sonec rollout worker as env |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | Ray + vLLM rollouts |
| [TRL GRPO](https://huggingface.co/docs/trl) | Smaller local experiments |

sonec stays source of truth for **harness + graders + trajectory format**.
Do not fork a second agent loop inside the trainer.

## Pinning

Every RL run must record:
- `harness_version`
- `tool_schema_hash`
- base model id
- WorldBench / SonecBench suite versions

Harness or tool-schema change ⇒ migration eval before new training.

## Collapse watches

- Terminal-only loops
- Comment-CoT without tools
- Entropy death (identical rollouts)
- pass@1 up but pass@k flat (coverage not improving)
