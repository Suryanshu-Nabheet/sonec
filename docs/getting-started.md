# Getting started — sonec

**sonec** by Suryanshu Nabheet is a coding model. Specialized behavior comes from trained LoRA weights.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"
cp .env.example .env
```

## Specialize and serve

```bash
source .venv/bin/activate   # required — `sonec` is the venv entry point
sonec train --step --live-fuel --sft-iters 300 --gold-n 0
sonec weights
sonec serve-llm
export SONEC_BASE_URL=http://127.0.0.1:8080/v1
sonec run "Add a unit test and verify" -w .
```

Product sonec = base Qwen 3.5 2B + LoRA via `sonec serve-llm`.  
`ollama run sonec` is a chat Modelfile runner only (no LoRA).

## Evaluate

```bash
# Needs LoRA on :8080 and unmodified base on :8081
python -m mlx_lm server --model mlx-community/Qwen3.5-2B-4bit --port 8081
sonec compare --out docs/results
```

See [TRAIN_PROOF.md](results/TRAIN_PROOF.md) and [COMPARE_REPORT.md](results/COMPARE_REPORT.md).

## Surfaces

```bash
sonec serve     # harness gateway (:8787) — point SONEC_BASE_URL at serve-llm
sonec mcp
```

## Licensing

Apache-2.0 for sonec code, adapters, and Qwen weight lineage. See [LICENSE](../LICENSE) and [NOTICE](../NOTICE).
