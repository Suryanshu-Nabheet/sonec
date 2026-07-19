# Training proof — specialized LoRA weights

Raw `*.safetensors` are gitignored. Reproduce with `./scripts/overnight_specialize.sh`.

## Product

| Claim | Status |
| --- | --- |
| Chat Modelfile only | Incomplete (runner, not product) |
| MLX LoRA `artifacts/train/checkpoints/sonec-sft-mlx` | **Product** |
| `sonec serve-llm` | **Product** |

## Published live results (2026-07-19) — smoke

### MLX A/B (`ab_agent_2b_hard`)

Source: [COMPARE_REPORT.md](COMPARE_REPORT.md) · `2026-07-19T07:36:47Z`

| Arm | Pass | Mean duration |
| --- | --- | ---: |
| **sonec LoRA** (`:8080`) | **8/8 (100%)** | **8.6s** |
| Qwen3.5-2B base (`:8081`) | 8/8 (100%) | 16.5s |

Pass-rate tie; sonec ≈ **1.9× faster**.

### Strict 2B multi-model board

Source: [leaderboard_2b/LEADERBOARD.md](leaderboard_2b/LEADERBOARD.md) · `2026-07-19T07:39:54Z`

| Rank | Model | Pass | Mean duration |
| ---: | --- | --- | ---: |
| 1 | **sonec** (LoRA) | **8/8 (100%)** | **8.5s** |
| 2 | qwen3.5:2b | 8/8 (100%) | 11.5s |
| 3 | gemma2:2b | 0/8 | — |
| 4 | codegemma:2b | 0/8 | — |

**Winner:** sonec (pass-rate tie → specialized LoRA + speed).

Honest note: the 8-task smoke is **saturated** for tool-capable 2B models. CapabilityBench 200 is the discriminating gate; full live Cap200 scores are **not published yet**.

## Decision suite — CapabilityBench 200

Sealed; never training fuel. Generate: `sonec capabilitybench`. Suite shipped; live Cap200 A/B pending a long local run.

| Category | n |
| --- | ---: |
| Read-only · Edit · Refactor · Bugfix · Verify | 20 each |
| Docs · CLI · Git · Restraint · Horizon | 20 each |
| **Total** | **200** |

Difficulty: 70 easy / 70 medium / 60 hard.

```bash
# Compare only (hours; board skipped by default)
SKIP_SFT=1 ./scripts/capabilitybench_e2e.sh

# Also run multi-model board
SKIP_SFT=1 SKIP_BOARD=0 ./scripts/capabilitybench_e2e.sh

# Fast probe
sonec leaderboard -s examples/benchmarks/capabilitybench_v1.json \
  -o docs/results/leaderboard_cap --limit 40 --fresh
```

## Reproduce

```bash
./scripts/overnight_specialize.sh
sonec capabilitybench
SKIP_GRPO=1 ./scripts/world_rl_leaderboard.sh   # defaults to Cap200; long
# Smoke instead:
# SUITE=examples/benchmarks/ab_agent_2b_hard.json SKIP_GRPO=1 ./scripts/world_rl_leaderboard.sh
```

GRPO: `sonec grpo --mock` only on laptops. Heavy live GRPO is refused (`G>4` or `n>16`).
