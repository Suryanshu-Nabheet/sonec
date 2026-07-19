# sonec architecture

**sonec** is a coding model. This repository also ships the training and serving stack around it: thin identity, tools, a frozen harness, and graded evidence. Training starts from an Apache-2.0 base (see NOTICE); the product name is the specialized adapter `sonec`.

```
CLI / MCP / HTTP
  → thin identity + tools
  → AgentRuntime (frozen)
    → evidence graders
  → TrainBench → SFT (MLX) → RFT → optional light GRPO-lite → product LoRA (sonec)
  → compare (LoRA vs base) · leaderboard (CapabilityBench 200 / smoke ab_agent_2b_hard)
```

| Layer | Canonical entry |
| --- | --- |
| Specialize overnight | `./scripts/overnight_specialize.sh` |
| Multi-model board | `SKIP_GRPO=1 ./scripts/world_rl_leaderboard.sh` |
| Decision metric | CapabilityBench 200 (sealed; never training fuel) |
| Smoke | `ab_agent_2b_hard` (8 tasks) |
| Training fuel | TrainBench + verified live trajectories |
| Inference | OpenAI-compatible `/v1` via `sonec serve-llm` |

Scripts under `scripts/` are the only long-running entrypoints. Large live GRPO is refused on laptops.
