# Gate — sonec on Qwen 3.5 (2B)

**Product:** `sonec`  
**Base:** `qwen3.5:2b` / `Qwen/Qwen3.5-2B` (Apache-2.0)  
**Code:** MIT — see LICENSE + NOTICE

## Pipeline

1. TrainBench graded rollouts (training-only ids)
2. Gold agentic curriculum (localize → patch → verify)
3. MLX LoRA SFT (`mlx-community/Qwen3.5-2B-4bit` or `Qwen/Qwen3.5-2B`)
4. RL rejection / group winners (RFT); GRPO recipe for larger GPU runs later
5. Serve product name `sonec` via CLI / HTTP / MCP

## Commands (start small)

```bash
sonec train --step --sft-iters 80 --gold-n 40 --train-n 16
sonec doctor
sonec worldbench --run --live -m sonec --limit 5
```

Raise `--sft-iters` / `--train-n` only after a step looks healthy.

## Contamination rule

Never train on sealed SonecBench / WorldBench task ids.
