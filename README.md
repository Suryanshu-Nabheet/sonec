# Sonec

**Sonec** by [Suryanshu Nabheet](https://github.com/Suryanshu-Nabheet)

LoRA specialization of **Qwen 3.5 2B** for tool-using software engineering.
Frozen harness. Evidence graders. Same tools as the base — **the weights are what change**.

[![CI](https://github.com/Suryanshu-Nabheet/sonec/actions/workflows/ci.yml/badge.svg)](https://github.com/Suryanshu-Nabheet/sonec/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Base: Qwen 3.5 2B](https://img.shields.io/badge/base-Qwen%203.5%202B-1a7f4c.svg)](NOTICE)

| Field | Value |
| --- | --- |
| **Product** | Specialized LoRA adapter (`artifacts/train/checkpoints/`) |
| **CLI** | `sonec` |
| **Serve** | `sonec serve-llm` → OpenAI-compatible `/v1` |
| **Base** | `mlx-community/Qwen3.5-2B-4bit` · `Qwen/Qwen3.5-2B` |
| **Train** | MLX · Unsloth · Axolotl · CPU PEFT |
| **License** | Apache-2.0 — code, adapters, Qwen lineage ([NOTICE](NOTICE)) |

```bash
sonec weights    # ready=True  →  product adapters present
sonec doctor     # environment + backend readiness
```

---

## Results

### Live agent A/B — published smoke (2026-07-19)

Suite: [`ab_agent_2b_hard.json`](examples/benchmarks/ab_agent_2b_hard.json) · 8 hard tasks · same frozen harness

| Arm | Kind | Pass | Mean duration | vs base |
| --- | --- | ---: | ---: | --- |
| **Sonec LoRA** | product | **8/8** | **8.6s** | **~1.9× faster** |
| Qwen 3.5 2B | base | 8/8 | 16.5s | — |

### 2B leaderboard

| Rank | Model | Pass | Mean duration |
| ---: | --- | ---: | ---: |
| **1** | **Sonec** | **8/8** | **8.5s** |
| 2 | qwen3.5:2b | 8/8 | 11.5s |
| 3 | gemma2:2b | 0/8 | — |
| 4 | codegemma:2b | 0/8 | — |

Reports: [COMPARE_REPORT.md](docs/results/COMPARE_REPORT.md) · [LEADERBOARD.md](docs/results/leaderboard_2b/LEADERBOARD.md)

### CapabilityBench 200

Primary sealed decision suite — **200** tasks (10 categories × 20). Never training fuel.

| Item | Status |
| --- | --- |
| Suite file | [`capabilitybench_v1.json`](examples/benchmarks/capabilitybench_v1.json) |
| Live multi-model scores | **Not published yet** |
| Promotion gate | Cap200 pass-rate lift (smoke is saturated) |

```bash
sonec capabilitybench
SKIP_SFT=1 ./scripts/capabilitybench_e2e.sh
```

### Weight specialization (NLL probe)

| Model | Mean token NLL (gold probe, n=8) | Δ |
| --- | ---: | ---: |
| Qwen 3.5 2B base | 2.159 | — |
| **Sonec LoRA** | **0.022** | **−2.137** |

Source: [SFT_METRICS.json](docs/results/SFT_METRICS.json).
NLL proves probe fit. **Live skill** is gated by Cap200.

---

## Architecture

```mermaid
flowchart TB
  subgraph Surfaces
    CLI["CLI · sonec run"]
    MCP[MCP]
    HTTP[HTTP gateway]
  end

  subgraph Runtime["Frozen AgentRuntime"]
    ID[Identity · Sonec]
    TOOLS["Tools · fs · terminal · git · index"]
    GRADE[Evidence graders]
  end

  subgraph Specialize
    FUEL["TrainBench fuel<br/>live / mock"]
    CORPUS["SFT corpus<br/>structured tool_calls"]
    LORA[LoRA SFT]
    RFT[Rejection winners]
    ADAPTER["Product adapter · Sonec"]
  end

  subgraph Backends
    MLX[MLX]
    UNS[Unsloth]
    AXO[Axolotl]
    CPU[CPU PEFT]
  end

  subgraph Out
    SERVE["sonec serve-llm · /v1"]
    EVAL["compare · Cap200"]
  end

  CLI --> Runtime
  MCP --> Runtime
  HTTP --> Runtime
  Runtime --> FUEL --> CORPUS --> LORA --> RFT --> ADAPTER
  LORA --> Backends
  ADAPTER --> SERVE --> EVAL
```

| Layer | Responsibility |
| --- | --- |
| **Harness** | Frozen tool surface; graders decide success from workspace evidence |
| **Fuel** | Live / TrainBench trajectories only — sealed benches excluded |
| **Train** | Corpus → LoRA SFT → rejection group winners |
| **Serve** | Base + adapter on OpenAI-compatible `/v1` |
| **Eval** | Fair A/B; promote only on pass-rate lift |

Design notes: [docs/architecture.md](docs/architecture.md) · gate: [docs/GATE_REPORT_MODEL_STACK.md](docs/GATE_REPORT_MODEL_STACK.md)

---

## Install

```bash
git clone https://github.com/Suryanshu-Nabheet/sonec.git
cd sonec
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

| Extra | Platform | Backend |
| --- | --- | --- |
| `.[train]` | Apple Silicon | **MLX** — product path |
| `.[train-cuda]` | Linux + NVIDIA | **Unsloth** — preferred CUDA |
| `.[train-axolotl]` | Linux + NVIDIA | **Axolotl** |
| `.[train-cpu]` | Any CPU | **CPU PEFT** — pipeline proof |

```bash
pip install -e ".[train]"          # or train-cuda / train-axolotl / train-cpu
sonec doctor
```

Product **Qwen 3.5 2B** adapters need MLX or CUDA. CPU mode runs the full pipeline without a GPU on a smaller Qwen.

---

## End-to-end workflow

```mermaid
flowchart LR
  A[1 · Specialize] --> B[2 · Serve]
  B --> C[3 · Run]
  C --> D[4 · Evaluate]
  D --> E{Cap200 lift?}
  E -->|yes| F[Keep adapter]
  E -->|no| A
```

### 1. Specialize

```bash
# Product — live graded trajectories
sonec train --step --live-fuel --sft-iters 300 --gold-n 0 --train-n 40

# Auto backend (MLX / Unsloth / CPU)
sonec train --step --backend auto --live-fuel --sft-iters 300

# Zero-GPU proof
sonec train --step --backend cpu --mock-fuel --sft-iters 40 --gold-n 32

sonec weights
```

| Artifact | Meaning |
| --- | --- |
| `artifacts/train/fuel/rollouts.jsonl` | Graded rollouts |
| `artifacts/train/sft_corpus/mlx_data/` | Chat JSONL with real `tool_calls` |
| `artifacts/train/checkpoints/sonec-sft-*` | LoRA adapters |
| `artifacts/train/PRODUCT.json` | Product manifest |
| `artifacts/train/TRAIN_REPORT.json` | Phase-by-phase report |

### 2. Serve

```bash
sonec serve-llm --port 8080                 # MLX product
sonec serve-llm --backend peft --port 8080  # Unsloth / Axolotl / CPU

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
# Smoke (minutes) — published claim
sonec compare -s examples/benchmarks/ab_agent_2b_hard.json -o docs/results

# Cap200 (hours) — decision suite
SKIP_SFT=1 ./scripts/capabilitybench_e2e.sh
```

**Promotion rule:** keep an adapter when Cap200 pass rate **exceeds** peers (or ties with clear speed + specialization), with no restraint regression.

---

## Training backends

```mermaid
flowchart LR
  AUTO["sonec train --backend auto"] --> DET{detect}
  DET -->|Apple Silicon + mlx-lm| MLX[mlx]
  DET -->|CUDA + unsloth| UNS[unsloth]
  DET -->|CUDA + axolotl| AXO[axolotl]
  DET -->|torch + peft| CPU[cpu]
  DET -->|nothing ready| HINT[install hint · no fake success]
```

| Backend | Command | Adapter directory |
| --- | --- | --- |
| MLX | `sonec train --step --backend mlx` | `…/sonec-sft-mlx` |
| Unsloth | `sonec train --step --backend unsloth` | `…/sonec-sft-unsloth` |
| Axolotl | `sonec train --step --backend axolotl` | `…/sonec-sft-axolotl` |
| CPU | `sonec train --step --backend cpu` | `…/sonec-sft-cpu` |

H2O LLM Studio: import `artifacts/train/sft_corpus/mlx_data/train.jsonl`.

Sealed suites (`capabilitybench`, `sonecbench`, `worldbench`, `ab_agent_*`) must **never** become training fuel.

---

## CLI

| Command | Purpose |
| --- | --- |
| `sonec version` | Package version |
| `sonec doctor` | Environment + weight readiness |
| `sonec weights` | Product adapter check |
| `sonec train` | Specialize LoRA (`--step`) |
| `sonec serve-llm` | Product inference (base + adapter) |
| `sonec serve` | Harness HTTP gateway |
| `sonec mcp` | MCP bridge for IDE hosts |
| `sonec run` | Single agent goal in a workspace |
| `sonec compare` | Fair A/B vs unmodified base |
| `sonec leaderboard` | Multi-model 2B board |
| `sonec capabilitybench` | Generate Cap200 suite |
| `sonec rollout` | Collect graded trajectories |

```bash
sonec --help && sonec train --help
```

---

## Repository layout

```text
sonec/
├── sonec/                 # Package — agent, harness, train, eval, CLI
├── examples/benchmarks/   # Smoke · Cap200 · TrainBench
├── artifacts/train/       # Fuel · corpus · checkpoints (local)
├── docs/                  # Architecture · gates · results
├── scripts/               # overnight · Cap200 e2e · leaderboard
├── configs/               # SFT · RL · leaderboard arms
├── Modelfile              # Optional chat runner (not the product)
├── NOTICE                 # Qwen attribution
└── LICENSE                # Apache-2.0
```

Raw `*.safetensors` are gitignored. Reproduce with `sonec train --step`.

---

## Roadmap

**Landed**

- Live smoke A/B: Sonec ties Qwen on pass, wins on speed
- Structured OpenAI `tool_calls` SFT
- Weight-level NLL proof on gold trajectories
- Live fuel + rejection winners in the train step
- Multi-backend train: MLX · Unsloth · Axolotl · CPU PEFT
- Production harness crash-safety + optional serve auth

**Next**

- Publish CapabilityBench 200 multi-model scores
- Promote adapters only on Cap200 pass-rate gates
- Scale live verified trajectories; keep sealed ids out of fuel

---

## Documentation

| Doc | Purpose |
| --- | --- |
| [Getting started](docs/getting-started.md) | Install → serve → smoke / Cap200 |
| [Architecture](docs/architecture.md) | Harness + train layout |
| [Training gate](docs/GATE_REPORT_MODEL_STACK.md) | Promote rules |
| [Training proof](docs/results/TRAIN_PROOF.md) | Published numbers |
| [Compare report](docs/results/COMPARE_REPORT.md) | Latest live A/B |
| [2B leaderboard](docs/results/leaderboard_2b/LEADERBOARD.md) | Multi-model smoke ranking |
| [SFT metrics](docs/results/SFT_METRICS.json) | NLL probe |
| [NOTICE](NOTICE) | Base weight lineage |

---

## License

**Apache License 2.0** © Suryanshu Nabheet.

Applies to Sonec **source**, **documentation**, and **derived LoRA adapters**, consistent with **Qwen 3.5** (Apache-2.0).

When redistributing adapters or checkpoints, include [LICENSE](LICENSE), [NOTICE](NOTICE), and Apache-2.0 text from the Qwen release if you also redistribute base weights.
