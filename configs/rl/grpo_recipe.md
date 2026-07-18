# sonec RL — GRPO-family recipe

## Goal

Train a coding-model policy inside the frozen sonec harness. Rewards come from
environment graders (WorldBench / private tasks), not model self-report.

## Algorithm

- Prefer GRPO / Dr.GRPO-style group relative policy gradient
- Group size G = 8–16 independent rollouts per prompt (`sonec rollout -g 8`)
- Prefer single-epoch prompts
- Avoid length-std / group-std pathologies that collapse tool use

## Data

```bash
sonec rollout --live -m sonec \
  --suite examples/benchmarks/smoke.json -g 8 --limit 20 \
  --out artifacts/rollouts/live

sonec train --export -r artifacts/rollouts/live/rollouts.jsonl -o artifacts/train
# Use artifacts/train/grpo_prompts.jsonl as the prompt pool
```

## Reward

Primary: `reward = 1.0 if graded.passed else 0.0` (file and command checks)

Optional penalties:
- skipped verification
- catastrophic tool misuse
- nonlinear length (lighter on easy tasks; allow tools on hard ones)

Final reward applies to all model tokens, including self-summaries.

## External stacks

| Stack | Notes |
| --- | --- |
| [verl](https://github.com/volcengine/verl) | Production GRPO; wire sonec rollout worker as env |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | Ray + vLLM rollouts |
| [TRL GRPO](https://huggingface.co/docs/trl) | Smaller local experiments |

sonec remains the source of truth for harness, graders, and trajectory format.
Do not fork a second agent loop inside the trainer.

## Pinning

Every RL run must record:
- `harness_version`
- `tool_schema_hash`
- base model id
- WorldBench / SonecBench suite versions

Harness or tool-schema change requires a migration eval before new training.

## Collapse watches

- Terminal-only loops
- Comment-only reasoning without tools
- Entropy death (identical rollouts)
- pass@1 up but pass@k flat (coverage not improving)
