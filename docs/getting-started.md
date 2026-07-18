# Getting started — sonec

Coding-agent stack on **Qwen 3.5 (2B)**. Specialize in small steps; serve via any OpenAI-compatible endpoint.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"
cp .env.example .env
# Point SONEC_BASE_URL at a server that hosts qwen3.5:2b (or sonec after train)
sonec doctor
```

## Agent

```bash
sonec run "Add a unit test and verify" -w .
```

## IDE

```bash
sonec serve
sonec mcp
```

## Specialize (small steps)

```bash
sonec train --step --sft-iters 80 --gold-n 40 --train-n 16
# later: raise iters / train-n; keep sealed benches out of fuel
```

Sealed evals (`sonecbench`, `worldbench`) stay held out. Training uses TrainBench + gold curriculum.

## Licensing

MIT for sonec code ([LICENSE](../LICENSE)). Qwen base weights are Apache-2.0 — see [NOTICE](../NOTICE).
