# sonec

**sonec** by [Suryanshu Nabheet](https://github.com/Suryanshu-Nabheet) — a coding model.

Specialize with LoRA and serve through an OpenAI-compatible endpoint.

[![CI](https://github.com/Suryanshu-Nabheet/sonec/actions/workflows/ci.yml/badge.svg)](https://github.com/Suryanshu-Nabheet/sonec/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Stack

| Piece | Detail |
| --- | --- |
| Product | LoRA adapter at `artifacts/train/checkpoints/sonec-sft-mlx` |
| Code | MIT — [LICENSE](LICENSE) · weight lineage — [NOTICE](NOTICE) |
| Inference | `sonec serve-llm` → OpenAI-compatible `/v1` |

Confirm specialized weights: `sonec weights`.

## Install

```bash
git clone https://github.com/Suryanshu-Nabheet/sonec.git
cd sonec
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"
cp .env.example .env
```

## Specialize

```bash
sonec train --step --live-fuel --sft-iters 300 --gold-n 0 --train-n 40
sonec weights
sonec serve-llm
SONEC_BASE_URL=http://127.0.0.1:8080/v1 sonec run "Fix the failing test" -w .
```

Reuse an existing corpus:

```bash
sonec train --step --corpus artifacts/train/sft_corpus/mlx_data --sft-iters 300
```

## Evaluate

```bash
sonec serve-llm --port 8080
python -m mlx_lm server --model mlx-community/Qwen3.5-2B-4bit --port 8081
sonec compare --suite examples/benchmarks/ab_agent_v1.json --out docs/results
```

Reports: `docs/results/COMPARE_REPORT.md`.

## Surfaces

```bash
sonec serve      # harness gateway
sonec mcp        # MCP bridge
sonec doctor     # environment and weight readiness
```

## Documentation

- [Getting started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Training gate](docs/GATE_REPORT_MODEL_STACK.md)
- [Results](docs/results/TRAIN_PROOF.md)
- [NOTICE](NOTICE)

## License

MIT © Suryanshu Nabheet. Third-party weight lineage: see NOTICE.
