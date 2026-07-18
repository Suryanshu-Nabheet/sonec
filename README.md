# SONEC

**Senior Open-source Neural Engineering Companion**

[![CI](https://github.com/Suryanshu-Nabheet/sonec/actions/workflows/ci.yml/badge.svg)](https://github.com/Suryanshu-Nabheet/sonec/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

SONEC is the **apex open-source agentic software-engineering system** — the operating layer that turns frontier models into staff-level engineering partners.

Built to lead the agentic era: prebuilt operator rules, progressive skills, multi-phase orchestration, verification gates, critique, sandboxed tools, and benchmark posture — end to end.

Powered by [Kimi K3](https://www.moonshot.ai/) as the default reasoning engine. Extensible to any OpenAI-compatible provider.

**Repository:** [github.com/Suryanshu-Nabheet/sonec](https://github.com/Suryanshu-Nabheet/sonec)

## What SONEC is

| Layer | Capability |
| --- | --- |
| **Prebuilt rules** | Production operator standards — constitution, process, git, design, security |
| **Skills** | Progressive expertise — SE, debug, TDD, review, security, design-eng, SWE benchmarks |
| **Context assembler** | Goal-conditioned intelligence assembly for every run |
| **Orchestrator** | `RECON → PLAN → EXECUTE → VERIFY → CRITIQUE → DELIVER` |
| **Critic** | Evidence-gated completion — verified work only |
| **Tools** | Sandboxed filesystem, terminal, git, index, memory, skills/rules meta tools |
| **Eval & training** | Task suites, grading, trajectory datasets for continuous ascent |

## Package layout

```
sonec/
  harness/             ← multi-phase orchestrator (core)
  rules/prebuilt/      ← shipped operating rules
  skills/              ← progressive expertise packs
  agent/ tools/ …      ← runtime primitives
tests/ docs/ examples/
```

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

First-class product assets under `sonec/rules/prebuilt/` (`prebuilt/<id>`).

- **Always-on:** engineering constitution, Suryanshu guidelines, git safety  
- **Conditional:** design, animation, enterprise web, security — activated by goal  

Full bodies via `rules_load`.

## Docs

- [Architecture](docs/architecture.md)
- [Getting started](docs/getting-started.md)
- [Constitution](target.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)

## License

MIT © Suryanshu Nabheet
