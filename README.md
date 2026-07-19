# sonec

**sonec** by [Suryanshu Nabheet](https://github.com/Suryanshu-Nabheet) — a specialized coding agent model.

LoRA specialization of **Qwen 3.5 2B** for tool-using software engineering: write, edit, verify, restrain. Same frozen harness as the base; weights differ.

[![CI](https://github.com/Suryanshu-Nabheet/sonec/actions/workflows/ci.yml/badge.svg)](https://github.com/Suryanshu-Nabheet/sonec/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Results

Live A/B on a sealed agent suite. Identical harness, tools, and prompts — only the endpoint weights change.

| Arm | Pass rate | Passed | Mean score | Mean duration |
| --- | --- | --- | --- | --- |
| **sonec LoRA** | **67%** | **4/6** | **0.67** | 19.8s |
| Qwen 3.5 2B base | 50% | 3/6 | 0.50 | 21.6s |

**Winner: sonec** — **+17%** absolute pass rate vs unmodified base.

| Task | sonec | Base |
| --- | --- | --- |
| `fs-nested-note` | Pass | Pass |
| `fix-seeded-bug` | Pass | Pass |
| `restraint-q` | Pass | Pass |
| `verify-script` | **Pass** | Fail |
| `py-util-main` | Fail | Fail |
| `pkg-greet` | Fail | Fail |

Suite: [`examples/benchmarks/ab_agent_v1.json`](examples/benchmarks/ab_agent_v1.json) · Full report: [`docs/results/COMPARE_REPORT.md`](docs/results/COMPARE_REPORT.md)

### Weight-level proof (not a prompt)

A Modelfile `SYSTEM` string does not change token likelihoods. Sonec does:

| Metric | Base | sonec LoRA |
| --- | --- | --- |
| Mean token NLL on gold agent probe | 2.159 | **0.022** |
| Improvement | — | **−2.137 NLL** |

Source: [`docs/results/SFT_METRICS.json`](docs/results/SFT_METRICS.json)

### Specialization snapshot

| Field | Value |
| --- | --- |
| Base | `mlx-community/Qwen3.5-2B-4bit` (Qwen 3.5 2B, Apache-2.0) |
| Method | MLX LoRA SFT + rejection filtering |
| Corpus | Live harness rollouts + oracle-graded gold; OpenAI structured `tool_calls` |
| Adapter | `artifacts/train/checkpoints/sonec-sft-mlx` (`adapters.safetensors`) |
| Readiness | `sonec weights` → `ready=True` |

Training record: [`docs/results/TRAIN_PROOF.md`](docs/results/TRAIN_PROOF.md) · Manifest: `artifacts/train/PRODUCT.json`

---

## What sonec is

| This is the product | This is not |
| --- | --- |
| MLX LoRA under `artifacts/train/checkpoints/sonec-sft-mlx` | A chat Modelfile alone |
| Served via `sonec serve-llm` (base + adapter) | Prompt-only wrapping of `qwen3.5:2b` |
| Agent loop: frozen tools, graders, trajectories | A larger general LLM |

Optional Ollama/Modelfile tags are **runners**. Specialized weights are the product.

---

## Stack

| Piece | Detail |
| --- | --- |
| Product | LoRA adapter on Qwen 3.5 2B |
| Harness | Frozen tool surface (filesystem, terminal, git, index) + evidence graders |
| Train | Live fuel → SFT corpus → MLX LoRA → rejection winners |
| Serve | OpenAI-compatible `/v1` (`sonec serve-llm`) |
| Code | MIT — [LICENSE](LICENSE) · Weight lineage — [NOTICE](NOTICE) |

---

## Install

```bash
git clone https://github.com/Suryanshu-Nabheet/sonec.git
cd sonec
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"
cp .env.example .env
```

Confirm environment and adapters:

```bash
sonec doctor
sonec weights
```

---

## End-to-end

### 1. Specialize

Live harness fuel (preferred):

```bash
sonec train --step --live-fuel --sft-iters 300 --gold-n 0 --train-n 40
sonec weights
```

Reuse an existing corpus:

```bash
sonec train --step --corpus artifacts/train/sft_corpus/mlx_data --sft-iters 300
```

Overnight path (if present):

```bash
./scripts/overnight_specialize.sh
```

### 2. Serve

```bash
# Specialized sonec (base + LoRA)
sonec serve-llm --port 8080

# Unmodified base (for A/B)
python -m mlx_lm server --model mlx-community/Qwen3.5-2B-4bit --port 8081
```

### 3. Run as an agent

```bash
SONEC_BASE_URL=http://127.0.0.1:8080/v1 sonec run "Fix the failing test" -w .
```

### 4. Evaluate (fair A/B)

```bash
sonec compare --suite examples/benchmarks/ab_agent_v1.json --out docs/results
```

Outputs:

- `docs/results/COMPARE_REPORT.md` — human summary
- `docs/results/COMPARE_REPORT.json` — machine summary
- `docs/results/arm_*.json` — per-task traces (gitignored; regenerate with compare)

Promote an adapter only when pass rate **exceeds** the base arm on the sealed suite.

**Product path:** `sonec serve-llm` on `:8080` (base + LoRA). `ollama run sonec` is an optional chat runner only — it does not load the specialized adapter.

---

## Surfaces

```bash
sonec serve      # harness gateway
sonec mcp        # MCP bridge
sonec doctor     # environment + weight readiness
sonec rollout    # trajectory collection
sonec bench      # harness benches
```

---

## Roadmap (honest)

Already landed:

- Live A/B win vs base on `ab_agent_v1` (+17%)
- Structured tool-call SFT (not text-shaped “Calling tool …”)
- Weight-level NLL proof
- Tool-argument wire fix for mlx_lm multi-turn

Next:

- Flip `py-util-main` and `pkg-greet` (multi-file create; stop explore-only loops)
- Scale live verified trajectories + rejection / RL
- Widen sealed eval before claiming broader superiority

---

## Documentation

| Doc | Purpose |
| --- | --- |
| [Getting started](docs/getting-started.md) | First run |
| [Architecture](docs/architecture.md) | Harness + training layout |
| [Training gate](docs/GATE_REPORT_MODEL_STACK.md) | Model stack decisions |
| [Training proof](docs/results/TRAIN_PROOF.md) | What counts as the product |
| [Compare report](docs/results/COMPARE_REPORT.md) | Latest live A/B |
| [NOTICE](NOTICE) | Base weight lineage (Apache-2.0) |

---

## License

MIT © Suryanshu Nabheet. Third-party weight lineage: see [NOTICE](NOTICE).
