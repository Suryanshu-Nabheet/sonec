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
sonec train --step --sft-iters 80
sonec weights          # READY only when *.safetensors exist
sonec serve-llm        # base + adapter on :8080
export SONEC_BASE_URL=http://127.0.0.1:8080/v1
sonec run "Add a unit test and verify" -w .
```

## Agent / IDE

```bash
sonec serve
sonec mcp
```

## Licensing

MIT for sonec code. Qwen base = Apache-2.0 — see [NOTICE](../NOTICE).
