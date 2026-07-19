# Training proof — specialized LoRA weights

Raw `*.safetensors` are gitignored. Reproduce with `./scripts/overnight_specialize.sh`.

## Product

| Claim | Status |
| --- | --- |
| Chat Modelfile only | Incomplete |
| MLX LoRA `artifacts/train/checkpoints/sonec-sft-mlx` | Product |
| `sonec serve-llm` | Product |

## Latest live results (2026-07-19)

### MLX A/B compare (`ab_agent_2b_hard`)

Source: [COMPARE_REPORT.md](COMPARE_REPORT.md) · generated `2026-07-19T07:36:47Z`

| Arm | Pass | Mean duration |
| --- | --- | ---: |
| **sonec LoRA** (`:8080`) | **8/8 (100%)** | **8.6s** |
| Qwen3.5-2B base (`:8081`) | 8/8 (100%) | 16.5s |

Pass-rate tie; sonec ≈ **1.9× faster**.

### Strict 2B multi-model board

Source: [leaderboard_2b/LEADERBOARD.md](leaderboard_2b/LEADERBOARD.md) · generated `2026-07-19T07:39:54Z`

Peers **only** ~2B: `qwen3.5:2b`, `gemma2:2b`, `codegemma:2b`.

| Rank | Model | Pass | Mean duration |
| ---: | --- | --- | ---: |
| 1 | **sonec** (LoRA) | **8/8 (100%)** | **8.5s** |
| 2 | qwen3.5:2b (Ollama) | 8/8 (100%) | 11.5s |
| 3 | gemma2:2b | 0/8 | — |
| 4 | codegemma:2b | 0/8 | — |

**Winner:** sonec (pass-rate tie with qwen3.5:2b → specialized LoRA + speed).

## Decision metric going forward

Primary: **CapabilityBench** — 200 sealed tasks (`sonec capabilitybench` → `examples/benchmarks/capabilitybench_v1.json`).

| Category | Count |
| --- | ---: |
| Read-only repo understanding | 20 |
| File editing | 20 |
| Multi-file refactoring | 20 |
| Bug fixing | 20 |
| Verification (tests/build) | 20 |
| Documentation updates | 20 |
| Terminal/CLI tasks | 20 |
| Git operations | 20 |
| Tool restraint | 20 |
| Long-horizon tasks | 20 |
| **Total** | **200** |

Difficulty: 70 easy / 70 medium / 60 hard. Tags include filesystem, python, typescript, react, rust, go, docs, verify, architecture, patch, search, git.

`ab_agent_2b_hard` remains the fast smoke (saturated for capable 2B tooling).

## Reproduce

```bash
./scripts/overnight_specialize.sh
sonec capabilitybench
SKIP_GRPO=1 ./scripts/world_rl_leaderboard.sh
```

GRPO: `sonec grpo --mock` only on laptops. Heavy live GRPO is refused (`G>4` or `n>16`).
