# sonec

**sonec** by [Suryanshu Nabheet](https://github.com/Suryanshu-Nabheet) — a specialized coding agent model.

LoRA specialization of **Qwen 3.5 2B** for tool-using software engineering. Trained and graded inside a frozen harness. Same tools and prompts as the base; the weights are what change.

[![CI](https://github.com/Suryanshu-Nabheet/sonec/actions/workflows/ci.yml/badge.svg)](https://github.com/Suryanshu-Nabheet/sonec/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

| | |
| --- | --- |
| **Product** | MLX LoRA adapter · `artifacts/train/checkpoints/sonec-sft-mlx` |
| **Serve** | `sonec serve-llm` → OpenAI-compatible `/v1` |
| **Base** | `mlx-community/Qwen3.5-2B-4bit` (Qwen 3.5 2B) |
| **License** | Apache-2.0 — code, adapters, and Qwen weight lineage ([NOTICE](NOTICE)) |
| **Latest A/B** | Published smoke: sonec **8/8 @ 8.5s** (2B board #1). Cap200 suite shipped; full live Cap200 scores not published yet |

---

## Table of contents

1. [What sonec is](#what-sonec-is)
2. [Results](#results)
3. [Architecture](#architecture)
4. [Requirements](#requirements)
5. [Install](#install)
6. [End-to-end workflow](#end-to-end-workflow)
7. [Training in depth](#training-in-depth)
8. [Evaluation and promotion](#evaluation-and-promotion)
9. [Configuration](#configuration)
10. [CLI reference](#cli-reference)
11. [Repository layout](#repository-layout)
12. [Roadmap](#roadmap)
13. [Documentation](#documentation)
14. [License](#license)

---

## What sonec is

sonec is **not** a larger general LLM and **not** a prompt wrapper. It is a small open-weight coding model specialized for agentic software engineering: call tools, write exact paths, edit with evidence, verify, and stay quiet on question-only asks.

| This is the product | This is not the product |
| --- | --- |
| LoRA under `artifacts/train/checkpoints/sonec-sft-mlx` (`adapters.safetensors`) | A Modelfile `SYSTEM` string alone |
| Served via `sonec serve-llm` (base + adapter) | `ollama run sonec` without the adapter |
| Frozen harness + evidence graders + trajectories | Unverified chat demos |
| Measurable win on a sealed live A/B suite | “Feels smarter” anecdotes |

Confirm readiness:

```bash
sonec weights
# ready=True  →  product adapters present
```

Optional Ollama / Modelfile tags are **chat runners**. They do not load the specialized LoRA. The product path is always `sonec serve-llm`.

---

## Results

### Live agent A/B (published smoke — 2026-07-19)

Suite: [`ab_agent_2b_hard.json`](examples/benchmarks/ab_agent_2b_hard.json) (8 tasks; fast, saturated for tool-capable 2B).

| Compare (MLX) | Pass | Mean duration |
| --- | --- | ---: |
| **sonec LoRA** | **8/8** | **8.6s** |
| Qwen3.5-2B base | 8/8 | 16.5s |

| 2B board | Pass | Mean duration |
| --- | --- | ---: |
| **sonec** | **8/8** | **8.5s** |
| qwen3.5:2b | 8/8 | 11.5s |
| gemma2:2b | 0/8 | — |
| codegemma:2b | 0/8 | — |

**Board winner:** sonec (pass-rate tie → LoRA + speed).  
Reports: [COMPARE_REPORT.md](docs/results/COMPARE_REPORT.md) · [LEADERBOARD.md](docs/results/leaderboard_2b/LEADERBOARD.md).

### CapabilityBench 200 (decision suite)

Sealed **200** tasks (10 categories × 20, easy/medium/hard, tagged). Not training fuel.
**Status:** suite + harness are in-repo; a full live Cap200 A/B takes hours on Apple Silicon and is **not** in the published results yet (smoke is).

```bash
sonec capabilitybench
# Compare only (hours):     SKIP_SFT=1 ./scripts/capabilitybench_e2e.sh
# Compare + multi-model:    SKIP_SFT=1 SKIP_BOARD=0 ./scripts/capabilitybench_e2e.sh
# Fast probe:               sonec leaderboard -s examples/benchmarks/capabilitybench_v1.json --limit 40 --fresh
```

Suite file: [`capabilitybench_v1.json`](examples/benchmarks/capabilitybench_v1.json).

### Strict 2B-only peers

Peers are **exactly ~2B** (`qwen3.5:2b`, `gemma2:2b`, `codegemma:2b`). No 1B / 1.5B / 3B+.

```bash
SKIP_GRPO=1 ./scripts/world_rl_leaderboard.sh
```

### Weight-level proof (not a prompt)

A Modelfile cannot change token likelihoods on gold agent trajectories. The LoRA does.

| Model | Mean token NLL (gold probe, n=8) |
| --- | ---: |
| Base `Qwen3.5-2B-4bit` | 2.159 |
| **sonec LoRA** | **0.022** |
| Improvement | **−2.137 NLL** |

Source: [docs/results/SFT_METRICS.json](docs/results/SFT_METRICS.json). NLL proves specialization on the probe set; **live A/B** is the agent-skill gate.

### Specialization snapshot

| Field | Value |
| --- | --- |
| Method | MLX LoRA SFT + rejection filtering |
| Corpus | Live harness rollouts + oracle-graded gold · OpenAI structured `tool_calls` |
| Scale | Checkpoints through `0000480_*` |
| Manifest | `artifacts/train/PRODUCT.json` |
| Training record | [docs/results/TRAIN_PROOF.md](docs/results/TRAIN_PROOF.md) |

---

## Architecture

```text
┌──────────────────────────────────────────────────────────┐
│  Surfaces:  CLI  ·  MCP  ·  HTTP gateway                 │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│  AgentRuntime (frozen)                                   │
│  · thin identity                                         │
│  · hashed core tools (fs / terminal / git / index)       │
│  · evidence graders (files, parses, restraint)           │
└────────────────────────────┬─────────────────────────────┘
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    TrainBench fuel     CapabilityBench     SonecBench / WorldBench
    (live rollouts)     (200 sealed / primary)  (legacy sealed)
           │            never train fuel    never train fuel
           ▼
    SFT corpus (structured tool_calls)
           │
           ▼
    MLX LoRA  →  rejection winners  →  sonec adapters
           │
           ▼
    sonec serve-llm  →  OpenAI /v1  →  sonec run / compare
```

| Layer | Responsibility |
| --- | --- |
| **Harness** | Frozen tool surface; graders decide success from workspace evidence |
| **Fuel** | Live / TrainBench trajectories only (sealed benches excluded) |
| **Train** | Corpus → MLX LoRA SFT → rejection group winners |
| **Serve** | Base + adapter on OpenAI-compatible `/v1` |
| **Eval** | Fair A/B (`sonec compare`); promote only on pass-rate lift |

Design notes: [docs/architecture.md](docs/architecture.md) · stack gate: [docs/GATE_REPORT_MODEL_STACK.md](docs/GATE_REPORT_MODEL_STACK.md)

---

## Requirements

| Requirement | Detail |
| --- | --- |
| OS | macOS recommended for MLX train/serve (Apple Silicon) |
| Python | 3.11+ |
| Disk | Room for Qwen 3.5 2B MLX weights + adapters |
| Network | First run downloads the MLX base model |
| Optional | Ollama — chat runner only, not the product path |

---

## Install

```bash
git clone https://github.com/Suryanshu-Nabheet/sonec.git
cd sonec

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev,train]"
cp .env.example .env

sonec doctor                       # environment
sonec weights                      # adapter readiness (after train)
```

| Extra | Use |
| --- | --- |
| `.[dev]` | Tests, lint, local development |
| `.[train]` | MLX LoRA specialization |

Keep the venv activated whenever you run `sonec` — the CLI entry point lives there.

---

## End-to-end workflow

One complete path from empty checkout to a fair A/B report.

### 1. Specialize

Preferred: live harness fuel (real tool trajectories).

```bash
source .venv/bin/activate

sonec train --step \
  --live-fuel \
  --sft-iters 300 \
  --gold-n 0 \
  --train-n 40

sonec weights
```

Reuse an existing corpus (skip fuel collection):

```bash
sonec train --step \
  --corpus artifacts/train/sft_corpus/mlx_data \
  --sft-iters 300
```

Overnight / longer unattended run:

```bash
./scripts/overnight_specialize.sh
```

Artifacts under `artifacts/train/`:

| Path | Meaning |
| --- | --- |
| `fuel/rollouts.jsonl` | Live / mock rollouts |
| `sft_corpus/mlx_data/` | MLX train JSONL |
| `checkpoints/sonec-sft-mlx/` | LoRA adapters (`adapters.safetensors`, step checkpoints) |
| `rl/winners.jsonl` | Rejection winners |
| `PRODUCT.json` | Product manifest |
| `TRAIN_REPORT.json` | Phase-by-phase train report |

### 2. Serve the product

```bash
# Terminal A — sonec (base + LoRA)
sonec serve-llm --port 8080
```

For A/B, also serve unmodified base:

```bash
# Terminal B — base only (no adapter)
python -m mlx_lm server \
  --model mlx-community/Qwen3.5-2B-4bit \
  --port 8081
```

Point the agent at the product endpoint:

```bash
export SONEC_BASE_URL=http://127.0.0.1:8080/v1
export SONEC_MODEL=mlx-community/Qwen3.5-2B-4bit
```

`SONEC_MODEL` must match the id advertised by `mlx_lm` / `serve-llm`, not the product name “sonec”.

### 3. Run as an agent

```bash
SONEC_BASE_URL=http://127.0.0.1:8080/v1 \
  sonec run "Fix the failing test" -w .
```

Useful flags:

```bash
sonec run "Add a unit test and verify" -w . --max-iterations 24
sonec run "Explain this module" -w ./src        # restraint-friendly goals
sonec run "…" -w . --mock                       # offline scripted provider
```

Other surfaces:

```bash
sonec serve          # harness HTTP gateway
sonec mcp            # MCP bridge for IDE hosts
```

### 4. Evaluate

**Smoke (minutes)** — published claim:

```bash
sonec compare -s examples/benchmarks/ab_agent_2b_hard.json -o docs/results
```

**CapabilityBench 200 (hours)** — discriminating decision suite:

```bash
SKIP_SFT=1 ./scripts/capabilitybench_e2e.sh
# or: sonec leaderboard -s examples/benchmarks/capabilitybench_v1.json \
#        -a configs/leaderboard/arms_2b.json -o docs/results/leaderboard_cap --fresh
```

| Flag | Default |
| --- | --- |
| `--lora-url` | `http://127.0.0.1:8080/v1` |
| `--base-url` | `http://127.0.0.1:8081/v1` |
| `--out` | `docs/results` |

| Output | Contents |
| --- | --- |
| `COMPARE_REPORT.md` | Human summary + winner |
| `COMPARE_REPORT.json` | Machine summary + per-task pass flags |
| `leaderboard_2b/LEADERBOARD.md` | Multi-model ranking |

**Promotion rule:** keep an adapter when sealed pass rate **exceeds** peers (or ties with clear speed + specialization), with no restraint regression. Prefer CapabilityBench over saturated smoke for pass-rate claims.

---

## Training in depth

Pipeline (one `sonec train --step`):

1. **Fuel** — graded rollouts (`--live-fuel` preferred; `--mock-fuel` offline)
2. **Corpus** — convert winners / gold into MLX chat JSONL with real OpenAI `tool_calls` (text-shaped “Calling tool …” dumps are rejected)
3. **SFT** — MLX LoRA on `mlx-community/Qwen3.5-2B-4bit`
4. **Rejection** — keep group winners into `rl/winners.jsonl`
5. **Product** — write `PRODUCT.json` + adapter dir readiness
6. **Modelfile** — optional chat runner update (not a substitute for LoRA)

| Flag | Role |
| --- | --- |
| `--step` | One specialize cycle (recommended) |
| `--live-fuel` | Collect live graded trajectories |
| `--mock-fuel` | Offline fuel for plumbing tests |
| `--sft-iters` | LoRA iterations (default `300`) |
| `--gold-n` | Optional gold seeds (`0` = live-only) |
| `--train-n` | TrainBench tasks for fuel |
| `--corpus PATH` | Reuse existing MLX data dir |
| `--skip-fuel` / `--skip-sft` | Resume mid-pipeline |
| `--exclude-sealed` | Keep sealed benches out of fuel (default on) |

Collect rollouts independently:

```bash
sonec rollout --live --group-size 8 --limit 40 \
  --suite examples/benchmarks/smoke.json \
  --out artifacts/rollouts
```

Sealed eval suites (`capabilitybench`, `sonecbench`, `worldbench`, `ab_agent_*`) must **never** become training fuel.

---

## Evaluation and promotion

| Suite | Role |
| --- | --- |
| **CapabilityBench** (`capabilitybench_v1.json`) | **Primary** sealed decision metric (200) |
| `ab_agent_2b_hard.json` | Fast smoke / published 2B board |
| `ab_agent_v1.json` | Legacy A/B |
| SonecBench / WorldBench | Legacy sealed |
| TrainBench / smoke | Training fuel only |

```bash
sonec capabilitybench
sonec leaderboard -s examples/benchmarks/capabilitybench_v1.json -o docs/results/leaderboard_cap
# smoke: sonec leaderboard -s examples/benchmarks/ab_agent_2b_hard.json -o docs/results/leaderboard_2b
```

Gate checklist before calling a run “better than base”:

1. `sonec weights` → `ready=True`
2. Smoke compare or CapabilityBench leaderboard with LoRA `:8080` vs peers
3. Pass rate **>** peers on CapabilityBench when claiming skill; smoke may tie on pass
4. Inspect failing tasks in `arm_*.json` (infra / connection errors do not count as model skill)

---

## Configuration

Copy [`.env.example`](.env.example) → `.env`:

```bash
SONEC_PROVIDER=local
SONEC_MODEL=mlx-community/Qwen3.5-2B-4bit
SONEC_BASE_URL=http://127.0.0.1:8080/v1
```

| Variable | Meaning |
| --- | --- |
| `SONEC_PROVIDER` | `local` · `openai` · `openai_compatible` · mock via CLI |
| `SONEC_BASE_URL` | OpenAI-compatible root including `/v1` |
| `SONEC_MODEL` | Model id returned by the server |
| `SONEC_API_KEY` | For remote OpenAI-compatible providers |

```bash
SONEC_PROVIDER=openai_compatible
SONEC_BASE_URL=http://127.0.0.1:8000/v1
SONEC_API_KEY=sk-local
SONEC_MODEL=your-served-id
```

---

## CLI reference

| Command | Purpose |
| --- | --- |
| `sonec version` | Package version |
| `sonec doctor` | Environment + weight readiness |
| `sonec weights` | Product adapter manifest check |
| `sonec train` | Specialize LoRA (`--step` recommended) |
| `sonec serve-llm` | Product inference (base + adapter) |
| `sonec serve` | Harness HTTP gateway |
| `sonec mcp` | MCP bridge |
| `sonec run` | Single agent goal in a workspace |
| `sonec compare` | Fair A/B vs unmodified base |
| `sonec leaderboard` | Multi-model 2B board (`--limit` for probes) |
| `sonec capabilitybench` | Generate sealed 200-task decision suite |
| `sonec grpo` | Light densify (`--mock` default; heavy live refused) |
| `sonec rollout` | Collect graded trajectories |
| `sonec bench` / `sonec eval` | Suite evaluation |
| `sonec sonecbench` / `worldbench` | Legacy sealed helpers |
| `sonec index` / `review` / `refactor` | Workspace helpers |
| `sonec skills` / `rules` | Skill and rule surfaces |

```bash
sonec --help
sonec train --help
sonec compare --help
```

---

## Repository layout

```text
sonec/
├── sonec/                 # Package — agent, harness, train, eval, CLI
├── examples/benchmarks/   # Suites (ab_agent_v1, smoke, sealed generators)
├── artifacts/train/       # Fuel, corpus, checkpoints, PRODUCT.json (local)
├── docs/
│   ├── getting-started.md
│   ├── architecture.md
│   ├── GATE_REPORT_MODEL_STACK.md
│   └── results/           # COMPARE_REPORT, TRAIN_PROOF, SFT_METRICS
├── scripts/               # overnight_specialize · world_rl_leaderboard · capabilitybench_e2e
├── configs/leaderboard/   # arms_2b.json catalog
├── configs/rl/            # grpo_recipe.md
├── Modelfile              # Optional chat runner (not the product)
├── NOTICE                 # Attribution + Qwen lineage
└── LICENSE                # Apache-2.0
```

Raw `*.safetensors` are gitignored. Reproduce with `sonec train --step`.

---

## Roadmap

**Landed**

- Live A/B win vs base on `ab_agent_v1` (+17%)
- Structured OpenAI `tool_calls` SFT (XML / “Calling tool” text dumps rejected)
- Weight-level NLL proof on gold trajectories
- mlx_lm multi-turn tool-argument wire fix
- Live fuel + rejection winners in the train step
- Fair compare CLI and sealed-suite separation
- Apache-2.0 licensing aligned with Qwen weight lineage

**Next**

- Publish CapabilityBench 200 multi-model scores on Apple Silicon (`SKIP_SFT=1 ./scripts/capabilitybench_e2e.sh`)
- Promote adapters only on Cap200 pass-rate gates (smoke is saturated / health-check)
- Scale live verified trajectories; keep sealed ids out of fuel (central `sonec.eval.sealed`)
- Identity: sonec by Suryanshu Nabheet — not Cursor

---

## Documentation

| Doc | Purpose |
| --- | --- |
| [Getting started](docs/getting-started.md) | Install → serve → smoke / Cap200 |
| [Architecture](docs/architecture.md) | Harness + train layout |
| [Training gate](docs/GATE_REPORT_MODEL_STACK.md) | Promote rules + harness notes |
| [Training proof](docs/results/TRAIN_PROOF.md) | Published numbers + Cap200 map |
| [Compare report](docs/results/COMPARE_REPORT.md) | Latest live A/B (smoke) |
| [2B leaderboard](docs/results/leaderboard_2b/LEADERBOARD.md) | Multi-model smoke ranking |
| [GRPO recipe](configs/rl/grpo_recipe.md) | Laptop-safe densify |
| [Daily status](docs/results/DAILY_STATUS.md) | Last committed snapshot; nightly runs upload fresh artifacts |
| [SFT metrics](docs/results/SFT_METRICS.json) | NLL specialization proof |
| [NOTICE](NOTICE) | Base weight lineage |

---

## License

**Apache License 2.0** © Suryanshu Nabheet.

Applies to sonec **source**, **documentation**, and **derived LoRA adapters**, consistent with the **Qwen 3.5** base (also Apache-2.0).

When redistributing adapters or checkpoints, include:

1. [LICENSE](LICENSE)
2. [NOTICE](NOTICE)
3. Apache-2.0 text from the Qwen release if you also redistribute base weights
