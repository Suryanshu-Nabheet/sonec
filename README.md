# SONEC

**Senior Open-source Neural Engineering Companion**

[![CI](https://github.com/Suryanshu-Nabheet/sonec/actions/workflows/ci.yml/badge.svg)](https://github.com/Suryanshu-Nabheet/sonec/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

SONEC is an **advanced agentic software-engineering system**.  
[Kimi K3](https://www.moonshot.ai/) is the reasoning engine — **SONEC is the harness** that converts it into a production coding agent: prebuilt rules, skills, multi-phase orchestration, verification gates, and critique.

> Not a thin API wrapper. The model reasons; the harness wins.

**Repository:** [github.com/Suryanshu-Nabheet/sonec](https://github.com/Suryanshu-Nabheet/sonec)

## Package layout

```
sonec/                 ← installable package (flat — no empty src/)
  harness/             ← multi-phase orchestrator (product core)
  rules/
    prebuilt/          ← shipped operating rules (constitution, design, security, …)
    engine.py
  skills/              ← progressive expertise packs
  agent/ tools/ …      ← runtime primitives
tests/ docs/ examples/
```

## Harness (what turns Kimi into SONEC)

| Layer | Role |
| --- | --- |
| **Prebuilt rules** | Always-on SE protocol + production operator standards |
| **Skills** | SE, debug, TDD, review, security, design-eng, benchmark-SWE, … |
| **Context assembler** | Goal-conditioned system prompt |
| **Orchestrator** | `RECON → PLAN → EXECUTE → VERIFY → CRITIQUE → DELIVER` |
| **Critic** | Blocks hollow “done” without verification evidence |
| **Tools** | Sandboxed fs / terminal / git / index / memory + skills/rules meta tools |

## Install

```bash
git clone https://github.com/Suryanshu-Nabheet/sonec.git
cd sonec
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```bash
sonec skills
sonec rules
sonec run "Summarize this repository" --mock

export MOONSHOT_API_KEY=sk-...
sonec run "Fix the failing tests and verify" -w .
```

## Prebuilt rules

Shipped under `sonec/rules/prebuilt/` as first-class product assets (ids like `prebuilt/engineering-constitution`).  
Always-on: engineering constitution, Suryanshu guidelines, git safety.  
Conditional: design, animation, enterprise web, security — activated by goal.

Agents load full bodies via `rules_load`.

## Docs

- [Architecture](docs/architecture.md)
- [Getting started](docs/getting-started.md)
- [Constitution](target.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)

## License

MIT © Suryanshu Nabheet
