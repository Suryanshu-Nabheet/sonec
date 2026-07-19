<div align="center">

# SONEC

**A specialized coding agent model**

LoRA specialization of Qwen 3.5 2B — tool-using software engineering,  
trained and graded in a frozen harness. Weights, not prompts.

<br/>

[![CI](https://github.com/Suryanshu-Nabheet/sonec/actions/workflows/ci.yml/badge.svg)](https://github.com/Suryanshu-Nabheet/sonec/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-111111?style=flat-square)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-111111?style=flat-square)](pyproject.toml)
[![Base](https://img.shields.io/badge/base-Qwen%203.5%202B-111111?style=flat-square)](NOTICE)
[![Product](https://img.shields.io/badge/product-MLX%20LoRA-111111?style=flat-square)](artifacts/train/PRODUCT.json)

<br/>

```text
sonec weights   →   sonec serve-llm   →   sonec run   →   sonec compare
```

[Install](#install) · [Quickstart](#quickstart) · [Evidence](#evidence) · [Architecture](#architecture) · [Docs](#documentation)

<br/>

by [Suryanshu Nabheet](https://github.com/Suryanshu-Nabheet)

</div>

---

<div align="center">

### Live A/B — sealed agent suite

| | **SONEC** | Base Qwen 3.5 2B | Delta |
| :---: | :---: | :---: | :---: |
| **Pass rate** | **67%** | 50% | **+17%** |
| **Passed** | **4 / 6** | 3 / 6 | +1 |
| **Mean score** | **0.67** | 0.50 | +0.17 |
| **Mean duration** | 19.8s | 21.6s | −1.8s |

Same harness · same tools · same prompts · **weights only**

</div>

| Task | SONEC | Base |
| --- | :---: | :---: |
| Nested path write | Pass | Pass |
| Seeded bug fix | Pass | Pass |
| Restraint (no tools) | Pass | Pass |
| Verify script + docs | **Pass** | Fail |
| Multi-file Python util | Fail | Fail |
| Package scaffold | Fail | Fail |

<details>
<summary><strong>Weight-level proof</strong> — NLL on gold agent trajectories (not a Modelfile)</summary>

<br/>

| Model | Mean token NLL |
| --- | ---: |
| Base `Qwen3.5-2B-4bit` | 2.159 |
| **SONEC LoRA** | **0.022** |
| Improvement | **−2.137** |

A system prompt cannot produce this gap. Source: [`docs/results/SFT_METRICS.json`](docs/results/SFT_METRICS.json)

</details>

Full report → [`docs/results/COMPARE_REPORT.md`](docs/results/COMPARE_REPORT.md)

---

## What is SONEC?

<div align="center">

| Product | Not the product |
| --- | --- |
| MLX LoRA at `artifacts/train/checkpoints/sonec-sft-mlx` | Chat Modelfile / `SYSTEM` alone |
| `sonec serve-llm` (base + adapter on `/v1`) | Prompt wrapper around `qwen3.5:2b` |
| Agent loop: tools · graders · trajectories | A larger general-purpose LLM |

</div>

**SONEC** is a coding model: specialize small open weights for verified agent behavior — write, edit, verify, restrain — then serve them through an OpenAI-compatible endpoint.

Optional Ollama tags are **runners**. The adapter is the product. Confirm with `sonec weights` → `ready=True`.

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│  CLI  ·  MCP  ·  HTTP                                       │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  AgentRuntime (frozen)                                      │
│  thin identity · hashed tools · evidence graders            │
└───────────────────────────┬─────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
   TrainBench fuel    Sealed SonecBench   WorldBench
   (live rollouts)    (never train fuel)  (never train fuel)
          │
          ▼
   SFT corpus  →  MLX LoRA  →  rejection winners  →  SONEC adapters
```

| Layer | Role |
| --- | --- |
| **Harness** | Frozen tool surface — filesystem, terminal, git, index |
| **Train** | Live fuel → structured `tool_calls` corpus → MLX LoRA → rejection filter |
| **Serve** | OpenAI-compatible `/v1` via `sonec serve-llm` |
| **Eval** | Fair A/B: LoRA `:8080` vs base-only `:8081` |

---

## Install

```bash
git clone https://github.com/Suryanshu-Nabheet/sonec.git
cd sonec
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"
cp .env.example .env
```

```bash
sonec doctor     # environment
sonec weights    # adapter readiness
```

---

## Quickstart

End-to-end path from specialization to a fair compare.

<table>
<tr>
<td width="50%" valign="top">

### 01 — Specialize

```bash
sonec train --step \
  --live-fuel \
  --sft-iters 300 \
  --gold-n 0 \
  --train-n 40

sonec weights
```

Reuse a corpus:

```bash
sonec train --step \
  --corpus artifacts/train/sft_corpus/mlx_data \
  --sft-iters 300
```

</td>
<td width="50%" valign="top">

### 02 — Serve

```bash
# SONEC — base + LoRA
sonec serve-llm --port 8080

# Unmodified base (A/B)
python -m mlx_lm server \
  --model mlx-community/Qwen3.5-2B-4bit \
  --port 8081
```

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 03 — Run

```bash
SONEC_BASE_URL=http://127.0.0.1:8080/v1 \
  sonec run "Fix the failing test" -w .
```

</td>
<td width="50%" valign="top">

### 04 — Compare

```bash
sonec compare \
  --suite examples/benchmarks/ab_agent_v1.json \
  --out docs/results
```

Promote only when pass rate **beats** base.

</td>
</tr>
</table>

**Outputs:** `COMPARE_REPORT.md` · `COMPARE_REPORT.json` · `arm_sonec_lora.json` · `arm_qwen35_2b_base.json`

> **Product path:** `sonec serve-llm` on `:8080`.  
> `ollama run sonec` does not load the specialized adapter.

---

## Specialization

| Field | Value |
| --- | --- |
| Base | `mlx-community/Qwen3.5-2B-4bit` · Qwen 3.5 2B (Apache-2.0) |
| Method | MLX LoRA SFT + rejection filtering |
| Corpus | Live harness rollouts + oracle-graded gold · OpenAI `tool_calls` |
| Adapter | `artifacts/train/checkpoints/sonec-sft-mlx` |
| Manifest | `artifacts/train/PRODUCT.json` |

Training record → [`docs/results/TRAIN_PROOF.md`](docs/results/TRAIN_PROOF.md)

Overnight:

```bash
./scripts/overnight_specialize.sh
```

---

## Surfaces

```bash
sonec serve       # harness gateway
sonec serve-llm   # OpenAI-compatible LLM (product)
sonec mcp         # MCP bridge
sonec doctor      # environment + weight readiness
sonec rollout     # trajectory collection
sonec bench       # harness benches
sonec compare     # fair A/B vs base
sonec train       # specialize LoRA
sonec weights     # product readiness
```

---

## Status

| Landed | Next |
| --- | --- |
| Live A/B win vs base (`ab_agent_v1`, +17%) | Flip `py-util-main` / `pkg-greet` |
| Structured tool-call SFT | Scale live verified trajectories + RL |
| Weight-level NLL proof | Widen sealed eval |
| mlx_lm multi-turn tool-arg wire fix | Promote only on pass-rate gates |

---

## Documentation

| | |
| --- | --- |
| [Getting started](docs/getting-started.md) | First run |
| [Architecture](docs/architecture.md) | Harness + training layout |
| [Training gate](docs/GATE_REPORT_MODEL_STACK.md) | Model stack decisions |
| [Training proof](docs/results/TRAIN_PROOF.md) | What counts as the product |
| [Compare report](docs/results/COMPARE_REPORT.md) | Latest live A/B |
| [NOTICE](NOTICE) | Base weight lineage |

---

<div align="center">

**SONEC**

MIT © [Suryanshu Nabheet](https://github.com/Suryanshu-Nabheet)  
Third-party weight lineage — [NOTICE](NOTICE)

<br/>

<sub>Specialize · Serve · Verify</sub>

</div>
