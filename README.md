# sonec

**Coding-agent model** on **Qwen 3.5 (2B)** — specialize it for tool use,
minimal diffs, and verify-before-done inside a frozen harness. Embed via CLI, HTTP, or MCP.

[![CI](https://github.com/Suryanshu-Nabheet/sonec/actions/workflows/ci.yml/badge.svg)](https://github.com/Suryanshu-Nabheet/sonec/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Stack

| Piece | Detail |
| --- | --- |
| Base | Qwen 3.5 2B (`Qwen/Qwen3.5-2B` / `qwen3.5:2b`) — Apache-2.0 |
| Product | `sonec` (specialized checkpoint / served name) |
| Code | MIT — see [LICENSE](LICENSE) + [NOTICE](NOTICE) |
| Inference | Any OpenAI-compatible endpoint (`SONEC_BASE_URL`) |

## Install

```bash
git clone https://github.com/Suryanshu-Nabheet/sonec.git
cd sonec
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"
cp .env.example .env   # set SONEC_BASE_URL to your local/remote OpenAI-compatible server
```

Serve `qwen3.5:2b` (or `sonec` after specialization) on that endpoint, then:

```bash
sonec doctor
sonec run "Fix the failing test and verify" -w .
```

## Small training steps (iterate)

Do not start with a giant run. One step, review, repeat:

```bash
sonec train --step --sft-iters 80 --gold-n 40 --train-n 16
# later: raise iters / train-n; keep sealed SonecBench / WorldBench out of training
```

## Surfaces

```bash
sonec serve    # HTTP agent gateway
sonec mcp      # IDE MCP stdio
```

## Docs

- [Getting started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Training gate](docs/GATE_REPORT_MODEL_STACK.md)
- [NOTICE](NOTICE) — Qwen Apache-2.0 + MIT obligations

## License

MIT © Suryanshu Nabheet — tooling. Base weights: Qwen Apache-2.0 (see NOTICE).
