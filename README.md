# sonec

**sonec** by [Suryanshu Nabheet](https://github.com/Suryanshu-Nabheet)

A LoRA-specialized coding agent on **Qwen 3.5 2B**. Same frozen harness and tools as the base model — only the weights change.

[![CI](https://github.com/Suryanshu-Nabheet/sonec/actions/workflows/ci.yml/badge.svg)](https://github.com/Suryanshu-Nabheet/sonec/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

| | |
|:--|:--|
| Product | LoRA adapter under `artifacts/train/checkpoints/` |
| Serve | `sonec serve-llm` → OpenAI-compatible `/v1` |
| Base | `mlx-community/Qwen3.5-2B-4bit` · `Qwen/Qwen3.5-2B` |
| Train | MLX · Unsloth · Axolotl · CPU PEFT |
| License | Apache-2.0 ([NOTICE](NOTICE)) |

```bash
sonec weights   # ready=True means product adapters are present
```

---

## Results

### Live A/B — smoke (2026-07-19)

Suite: [`ab_agent_2b_hard.json`](examples/benchmarks/ab_agent_2b_hard.json)

| Model | Pass | Mean duration |
|:--|--:|--:|
| **sonec LoRA** | **8/8** | **8.6s** |
| Qwen 3.5 2B base | 8/8 | 16.5s |

| 2B board | Pass | Mean duration |
|:--|--:|--:|
| **sonec** | **8/8** | **8.5s** |
| qwen3.5:2b | 8/8 | 11.5s |
| gemma2:2b | 0/8 | — |
| codegemma:2b | 0/8 | — |

sonec ties base on pass rate and wins on speed (~1.9×). Full reports: [COMPARE_REPORT.md](docs/results/COMPARE_REPORT.md) · [LEADERBOARD.md](docs/results/leaderboard_2b/LEADERBOARD.md).

### CapabilityBench 200

Primary sealed decision suite: 200 tasks, never used as training fuel.

| | |
|:--|:--|
| Suite | [`capabilitybench_v1.json`](examples/benchmarks/capabilitybench_v1.json) |
| Live scores | Not published yet |
| Gate | Promote adapters only on Cap200 pass-rate lift |

```bash
SKIP_SFT=1 ./scripts/capabilitybench_e2e.sh
```

### Specialization probe (NLL)

| Model | Mean token NLL (n=8) |
|:--|--:|
| Qwen 3.5 2B base | 2.159 |
| **sonec LoRA** | **0.022** |

Source: [SFT_METRICS.json](docs/results/SFT_METRICS.json). Lower NLL = tighter fit to graded agent trajectories. Skill claims still go through Cap200.

---

## How it works

```text
CLI / MCP / HTTP
       │
       ▼
┌──────────────────────────────┐
│  Frozen AgentRuntime         │
│  identity · tools · graders  │
└──────────────┬───────────────┘
               │
               ▼
     TrainBench fuel (live / mock)
               │
               ▼
     SFT corpus (real tool_calls)
               │
               ▼
     LoRA SFT → rejection winners → product adapter
               │
               ▼
     sonec serve-llm  →  /v1  →  compare / Cap200
```

| Layer | Role |
|:--|:--|
| Harness | Frozen tools; success from workspace evidence |
| Fuel | Live / TrainBench only — sealed benches excluded |
| Train | Corpus → LoRA → rejection filtering |
| Serve | Base + adapter on OpenAI `/v1` |
| Eval | Fair A/B; promote on pass-rate lift |

More detail: [docs/architecture.md](docs/architecture.md)

---

## Install

```bash
git clone https://github.com/Suryanshu-Nabheet/sonec.git
cd sonec
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Pick a train extra for your machine:

| Extra | When |
|:--|:--|
| `.[train]` | Apple Silicon — MLX (product path) |
| `.[train-cuda]` | Linux + NVIDIA — Unsloth (preferred CUDA) |
| `.[train-axolotl]` | Linux + NVIDIA — Axolotl |
| `.[train-cpu]` | No GPU — PEFT pipeline proof on a small Qwen |

```bash
pip install -e ".[train]"         # or train-cuda / train-axolotl / train-cpu
sonec doctor
```

Product Qwen 3.5 2B adapters need MLX or CUDA. CPU mode proves the pipeline when you have no GPU.

---

## Workflow

### 1. Specialize

```bash
# Product path — live graded trajectories
sonec train --step --live-fuel --sft-iters 300 --gold-n 0 --train-n 40

# Apple Silicon / CUDA
sonec train --step --backend auto --live-fuel --sft-iters 300

# No GPU (pipeline proof)
sonec train --step --backend cpu --mock-fuel --sft-iters 40 --gold-n 32

sonec weights
```

| Artifact | Meaning |
|:--|:--|
| `artifacts/train/fuel/rollouts.jsonl` | Graded rollouts |
| `artifacts/train/sft_corpus/mlx_data/` | Chat JSONL with structured `tool_calls` |
| `artifacts/train/checkpoints/sonec-sft-*` | LoRA adapters |
| `artifacts/train/PRODUCT.json` | Product manifest |

### 2. Serve

```bash
sonec serve-llm --port 8080                    # MLX
sonec serve-llm --backend peft --port 8080     # Unsloth / Axolotl / CPU

export SONEC_BASE_URL=http://127.0.0.1:8080/v1
export SONEC_MODEL=<id advertised by the server>
```

### 3. Run

```bash
SONEC_BASE_URL=http://127.0.0.1:8080/v1 \
  sonec run "Fix the failing test" -w .
```

### 4. Evaluate

```bash
# Smoke (minutes)
sonec compare -s examples/benchmarks/ab_agent_2b_hard.json -o docs/results

# Cap200 (hours)
SKIP_SFT=1 ./scripts/capabilitybench_e2e.sh
```

Keep an adapter when Cap200 pass rate beats peers (or ties with clear speed + specialization), with no restraint regression.

---

## Train backends

| Backend | Command | Adapter |
|:--|:--|:--|
| MLX | `sonec train --step --backend mlx` | `…/sonec-sft-mlx` |
| Unsloth | `sonec train --step --backend unsloth` | `…/sonec-sft-unsloth` |
| Axolotl | `sonec train --step --backend axolotl` | `…/sonec-sft-axolotl` |
| CPU | `sonec train --step --backend cpu` | `…/sonec-sft-cpu` |

`auto` picks MLX on Apple Silicon, Unsloth on CUDA, then CPU if neither is available.

Sealed suites (`capabilitybench`, `sonecbench`, `worldbench`, `ab_agent_*`) must never become training fuel.

---

## CLI

| Command | Purpose |
|:--|:--|
| `sonec doctor` | Environment + weight readiness |
| `sonec weights` | Product adapter check |
| `sonec train` | Specialize LoRA |
| `sonec serve-llm` | Product inference |
| `sonec serve` / `sonec mcp` | Harness HTTP / MCP |
| `sonec run` | Single agent goal |
| `sonec compare` | Fair A/B vs base |
| `sonec leaderboard` | Multi-model board |
| `sonec capabilitybench` | Build Cap200 suite |
| `sonec rollout` | Collect graded trajectories |

```bash
sonec --help
```

---

## Layout

```text
sonec/
├── sonec/                 # agent · harness · train · eval · CLI
├── examples/benchmarks/   # smoke · Cap200 · TrainBench
├── artifacts/train/       # local fuel · corpus · checkpoints
├── docs/                  # architecture · gates · results
├── scripts/               # overnight · Cap200 e2e
├── configs/               # SFT · RL · leaderboard arms
├── Modelfile              # optional chat runner (not the product)
├── NOTICE
└── LICENSE
```

Raw `*.safetensors` are gitignored. Reproduce with `sonec train --step`.

---

## Docs

| Doc | |
|:--|:--|
| [Getting started](docs/getting-started.md) | Install → serve → eval |
| [Architecture](docs/architecture.md) | Harness + train |
| [Training gate](docs/GATE_REPORT_MODEL_STACK.md) | Promotion rules |
| [Training proof](docs/results/TRAIN_PROOF.md) | Published numbers |
| [Compare report](docs/results/COMPARE_REPORT.md) | Latest live A/B |
| [2B leaderboard](docs/results/leaderboard_2b/LEADERBOARD.md) | Smoke ranking |
| [SFT metrics](docs/results/SFT_METRICS.json) | NLL probe |

---

## License

Apache-2.0 © Suryanshu Nabheet.

Covers source, docs, and derived LoRA adapters, consistent with Qwen 3.5 (Apache-2.0). Redistribute with [LICENSE](LICENSE), [NOTICE](NOTICE), and Qwen’s Apache text if you ship base weights.
