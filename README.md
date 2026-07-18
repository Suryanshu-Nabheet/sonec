# sonec

**Coding-agent model** specialized from **Qwen 3.5 (2B)** via real LoRA weight updates —
not a Modelfile / system-prompt wrapper. Harness + train + serve end-to-end.

[![CI](https://github.com/Suryanshu-Nabheet/sonec/actions/workflows/ci.yml/badge.svg)](https://github.com/Suryanshu-Nabheet/sonec/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Stack

| Piece | Detail |
| --- | --- |
| Base | Qwen 3.5 2B (`Qwen/Qwen3.5-2B`) — Apache-2.0 |
| Product | LoRA adapter at `artifacts/train/checkpoints/sonec-sft-mlx` |
| Code | MIT — [LICENSE](LICENSE) + [NOTICE](NOTICE) |
| Inference | `sonec serve-llm` → OpenAI-compatible `/v1` with base + adapter |

A `Modelfile` SYSTEM string alone is **not** sonec. Check: `sonec weights`.

## Install

```bash
git clone https://github.com/Suryanshu-Nabheet/sonec.git
cd sonec
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"
cp .env.example .env
```

## Specialize (real weights)

```bash
sonec train --step --sft-iters 80 --gold-n 40 --train-n 16
sonec weights                    # must show READY (*.safetensors)
sonec serve-llm                  # :8080 with LoRA loaded
SONEC_BASE_URL=http://127.0.0.1:8080/v1 sonec run "Fix the failing test" -w .
```

Reuse an existing mlx corpus:

```bash
sonec train --step --corpus artifacts/train/sft_corpus/mlx_data --sft-iters 80
```

## Prove LoRA > base

```bash
# terminal A
sonec serve-llm --port 8080
# terminal B — base weights only (no adapter)
python -m mlx_lm server --model mlx-community/Qwen3.5-2B-4bit --port 8081
# terminal C
sonec compare --suite examples/benchmarks/ab_agent_v1.json --out docs/results
```

Reports land in `docs/results/COMPARE_REPORT.md` (committed proof, not raw weights).

## Surfaces

```bash
sonec serve      # agent harness gateway
sonec mcp        # IDE MCP
sonec doctor     # fails until LoRA weights exist
```

## Docs

- [Getting started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Training gate](docs/GATE_REPORT_MODEL_STACK.md)
- [Results / proof](docs/results/TRAIN_PROOF.md) — LoRA NLL + live A/B
- [NOTICE](NOTICE) — Qwen Apache-2.0 + MIT

## License

MIT © Suryanshu Nabheet — tooling. Base weights: Qwen Apache-2.0 (see NOTICE).
