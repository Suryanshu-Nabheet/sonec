# Getting started — sonec

**sonec** by Suryanshu Nabheet is a coding model. Specialized behavior comes from trained LoRA weights (`artifacts/train/checkpoints/sonec-sft-mlx`), not from a Modelfile alone.

## Install (once)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"
cp .env.example .env
```

## Fast path — serve and run

```bash
source .venv/bin/activate
sonec weights                    # ready=True required
sonec serve-llm --port 8080      # product = base + LoRA
export SONEC_BASE_URL=http://127.0.0.1:8080/v1
sonec run "Add a unit test and verify" -w .
```

`ollama run sonec` is a **chat runner only** (no LoRA). Always use `sonec serve-llm` for the product.

## Specialize (train)

```bash
# Canonical overnight specialize + smoke A/B
./scripts/overnight_specialize.sh

# Or one step (live fuel preferred when serve is up)
sonec train --step --live-fuel --sft-iters 300 --gold-n 0 --train-n 40
sonec weights
```

Laptop-safe densify (no heavy GRPO):

```bash
sonec grpo --mock
```

## Evaluate

### Smoke (minutes) — published board

```bash
# Terminal A: sonec serve-llm --port 8080
# Terminal B:
python -m mlx_lm server --model mlx-community/Qwen3.5-2B-4bit --port 8081

sonec compare -s examples/benchmarks/ab_agent_2b_hard.json -o docs/results
sonec leaderboard -s examples/benchmarks/ab_agent_2b_hard.json \
  -a configs/leaderboard/arms_2b.json -o docs/results/leaderboard_2b --fresh
```

Latest published smoke (2026-07-19): **sonec 8/8 @ 8.5s**, board #1 among strict 2B peers.

### CapabilityBench 200 (hours) — decision suite

```bash
sonec capabilitybench            # writes examples/benchmarks/capabilitybench_v1.json

# Full densify + Cap200 compare + multi-model board (long)
./scripts/capabilitybench_e2e.sh

# Eval only (adapters already trained)
SKIP_SFT=1 ./scripts/capabilitybench_e2e.sh

# Or smoke-sized probe
sonec leaderboard -s examples/benchmarks/capabilitybench_v1.json \
  -a configs/leaderboard/arms_2b.json -o docs/results/leaderboard_cap --limit 40 --fresh
```

Reports: [TRAIN_PROOF.md](results/TRAIN_PROOF.md) · [COMPARE_REPORT.md](results/COMPARE_REPORT.md) · [leaderboard_2b/LEADERBOARD.md](results/leaderboard_2b/LEADERBOARD.md)

## Surfaces

```bash
sonec serve          # harness HTTP gateway (:8787)
sonec mcp            # MCP bridge
sonec doctor         # readiness checklist
sonec grpo --mock    # light densify only
```

## Licensing

Apache-2.0 — [LICENSE](../LICENSE) · [NOTICE](../NOTICE).
