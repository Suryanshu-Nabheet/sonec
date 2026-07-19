# sonec architecture

**sonec** is a coding model. This repository also ships the training and serving stack around it: thin identity, tools, a frozen harness, and graded evidence. Training starts from an Apache-2.0 base (see NOTICE); the product name is the specialized adapter `sonec`.

```
CLI / MCP / HTTP
  → thin identity + tools
  → AgentRuntime (frozen)
    → evidence graders
  → TrainBench → SFT (MLX) → RFT → optional GRPO-lite → product LoRA (sonec)
  → compare (LoRA vs base) · leaderboard (multi-model, resume-safe)
```

| Layer | Canonical entry |
| --- | --- |
| Specialize overnight | `./scripts/overnight_specialize.sh` |
| Multi-model board | `./scripts/world_rl_leaderboard.sh` |
| Decision metrics | sealed SonecBench / WorldBench (never training fuel) |
| Training fuel | TrainBench + verified live trajectories |
| Inference | OpenAI-compatible `/v1` via `sonec serve-llm` |

Scripts under `scripts/` are the only long-running entrypoints. One-off helpers were removed.
