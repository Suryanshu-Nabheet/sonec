# Getting started — sonec

Product = **trained LoRA on Qwen 3.5 2B**, not a prompt wrapper.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"
cp .env.example .env
```

## Specialize then serve

```bash
sonec train --step --sft-iters 160 --gold-n 96
sonec weights          # READY only when *.safetensors exist
sonec serve-llm        # base + adapter on :8080
export SONEC_BASE_URL=http://127.0.0.1:8080/v1
sonec run "Add a unit test and verify" -w .
```

## Prove specialization

```bash
# terminal B — base only
python -m mlx_lm server --model mlx-community/Qwen3.5-2B-4bit --port 8081
sonec compare --out docs/results
```

See [docs/results/TRAIN_PROOF.md](results/TRAIN_PROOF.md) and [COMPARE_REPORT.md](results/COMPARE_REPORT.md).

## Agent / IDE

```bash
sonec serve
sonec mcp
```

## Licensing

MIT for sonec code. Qwen base = Apache-2.0 — see [NOTICE](../NOTICE).
