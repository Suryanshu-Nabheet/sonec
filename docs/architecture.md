# sonec architecture

Coding-agent model specialized from **Qwen 3.5 (2B)** for tool use in IDEs and CLIs.

```
IDE / CLI / MCP / HTTP
  → thin identity + core tools
  → AgentRuntime (frozen)
    → evidence graders
  → TrainBench rollouts → SFT (MLX) → RL rejection → product name sonec
```

- **Decision metrics:** sealed SonecBench + WorldBench (never in training fuel)
- **Training fuel:** TrainBench + gold agent curriculum
- **Inference:** any OpenAI-compatible `/v1` endpoint (`SONEC_BASE_URL`)
