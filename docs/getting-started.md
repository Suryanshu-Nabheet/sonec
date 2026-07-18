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
sonec train --step --live-fuel --sft-iters 300 --gold-n 0
sonec weights
sonec serve-llm
export SONEC_BASE_URL=http://127.0.0.1:8080/v1
sonec run "Add a unit test and verify" -w .
```

## Evaluate

```bash
python -m mlx_lm server --model mlx-community/Qwen3.5-2B-4bit --port 8081
sonec compare --out docs/results
```

See [TRAIN_PROOF.md](results/TRAIN_PROOF.md) and [COMPARE_REPORT.md](results/COMPARE_REPORT.md).

## Surfaces

```bash
sonec serve
sonec mcp
```

## Licensing

MIT for sonec code. Weight lineage: [NOTICE](../NOTICE).
