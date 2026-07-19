# Getting started — sonec

**sonec** by Suryanshu Nabheet is a coding model. Specialized behavior comes from trained LoRA weights under `artifacts/train/checkpoints/`, not from a Modelfile alone.

## Install (once)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# Apple Silicon:   pip install -e ".[train]"
# Linux CUDA:      pip install -e ".[train-cuda]"
# Zero-GPU proof:  pip install -e ".[train-cpu]"
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

# Zero-GPU pipeline proof (small Qwen, not product 2B claims)
sonec train --step --backend cpu --mock-fuel --sft-iters 40 --gold-n 32

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

Suite is sealed in-repo. Full live scores are optional and slow; published claim above is smoke.

```bash
sonec capabilitybench            # writes examples/benchmarks/capabilitybench_v1.json

# Compare only (default skips multi-model board)
SKIP_SFT=1 ./scripts/capabilitybench_e2e.sh

# Compare + multi-model board (much longer)
SKIP_SFT=1 SKIP_BOARD=0 ./scripts/capabilitybench_e2e.sh

# Time-boxed probe
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
