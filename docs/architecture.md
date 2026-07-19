# sonec architecture

**sonec** is a coding model plus the training/serving stack: thin identity, tools, frozen harness, graded evidence. Base weights are Apache-2.0 (NOTICE); the product name is the specialized LoRA adapter `sonec`.

```
CLI / MCP / HTTP
  → thin identity + tools
  → AgentRuntime (frozen)
    → evidence graders (files, parses, only_files restraint, …)
  → TrainBench → SFT (MLX) → RFT → optional light GRPO-lite → product LoRA
  → compare / leaderboard
      smoke: ab_agent_2b_hard (minutes)
      decision: CapabilityBench 200 (hours)
```

| Layer | Entry |
| --- | --- |
| Specialize | `./scripts/overnight_specialize.sh` or `sonec train --step --backend auto` |
| Linux CUDA | `sonec train --step --backend unsloth` (or `axolotl`) |
| Smoke board | `SUITE=…/ab_agent_2b_hard.json SKIP_GRPO=1 ./scripts/world_rl_leaderboard.sh` |
| Cap200 board | `SKIP_SFT=1 ./scripts/capabilitybench_e2e.sh` |
| Decision metric | CapabilityBench 200 (sealed; never training fuel) |
| Training fuel | TrainBench + verified live trajectories |
| Inference | `sonec serve-llm` → OpenAI-compatible `/v1` (MLX or PEFT) |

## Harness details that matter

- **Write-first / patch-after-read** in the system prompt — agents must not stop after `fs_read` alone.
- **Small-file `fs_read`** returns raw text (no `N|` prefixes) so edits paste correctly.
- **`only_files` checks** — restraint tasks fail if unexpected files appear; harness noise (`.trajectories`, `.sonec`) is ignored.
- **Executable verify** — Cap200 verify tasks use `python_exec` / `command` (exit 0) via TerminalService.
- **Sealed exclusion** — Cap / Sonec / World / `ab_agent_*` ids never enter SFT or RFT fuel (`sonec.eval.sealed`).
- **Author** — product identity is sonec by Suryanshu Nabheet.

Scripts under `scripts/` are the long-running entrypoints. Heavy live GRPO (`G>4` or `n>16`) is refused on laptops.

CI: `.github/workflows/ci.yml` (push/PR) · Daily: `.github/workflows/daily.yml` (06:00 UTC) uploads `DAILY_STATUS.*` as workflow artifacts (no auto-commits).
