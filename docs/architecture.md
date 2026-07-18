# sonec architecture

**sonec** is a coding model. This repository also ships the training and serving stack around it: thin identity, tools, a frozen harness, and graded evidence. Training starts from an Apache-2.0 base (see NOTICE); the product name is the specialized adapter `sonec`.

```
CLI / MCP / HTTP
  → thin identity + tools
  → AgentRuntime (frozen)
    → evidence graders
  → TrainBench rollouts → SFT (MLX) → RL rejection → product weights (sonec)
```

- **Decision metrics:** sealed SonecBench and WorldBench (never training fuel)
- **Training fuel:** TrainBench and verified live trajectories
- **Inference:** OpenAI-compatible `/v1` (`SONEC_BASE_URL`)
