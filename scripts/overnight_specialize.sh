#!/usr/bin/env bash
# Canonical specialization: fuel → LoRA SFT → rejection RFT → A/B vs base.
# Oracle-graded tool_calls corpus (mock fuel) for reliable overnight runs.
# For live fuel: sonec train --step --live-fuel …
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

mkdir -p artifacts/logs
LOG="artifacts/logs/overnight_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== sonec overnight specialize ==="
echo "log=$LOG"
echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "=== Phase 1: corpus + LoRA SFT + RFT ==="
rm -rf artifacts/train
sonec train --step \
  --mock-fuel --mock-rl \
  --sft-iters 500 \
  --gold-n 240 \
  --train-n 100 \
  --rollout-group 8 \
  --rl-group 6 \
  --rl-limit 40

sonec weights

echo "=== Phase 2: inference servers ==="
pkill -f "mlx_lm server.*8080" 2>/dev/null || true
pkill -f "mlx_lm server.*8081" 2>/dev/null || true
sleep 2

python -m mlx_lm server \
  --model mlx-community/Qwen3.5-2B-4bit \
  --adapter-path artifacts/train/checkpoints/sonec-sft-mlx \
  --host 127.0.0.1 --port 8080 \
  >artifacts/logs/serve_lora.log 2>&1 &
echo $! >artifacts/logs/serve_lora.pid

python -m mlx_lm server \
  --model mlx-community/Qwen3.5-2B-4bit \
  --host 127.0.0.1 --port 8081 \
  >artifacts/logs/serve_base.log 2>&1 &
echo $! >artifacts/logs/serve_base.pid

echo "waiting for servers..."
for i in $(seq 1 90); do
  if curl -sf http://127.0.0.1:8080/v1/models >/dev/null \
    && curl -sf http://127.0.0.1:8081/v1/models >/dev/null; then
    echo "servers ready after ${i}s"
    break
  fi
  sleep 2
done

echo "=== Phase 3: A/B compare (smoke — may saturate) ==="
sonec compare \
  --suite examples/benchmarks/ab_agent_2b_hard.json \
  --out docs/results \
  --lora-url http://127.0.0.1:8080/v1 \
  --base-url http://127.0.0.1:8081/v1

echo "=== Phase 3b reminder: Cap200 is the decision gate ==="
echo "Run when ready (hours): SKIP_SFT=1 ./scripts/capabilitybench_e2e.sh"
echo "Promote only if CapabilityBench pass rate improves (smoke can tie 8/8)."

echo "=== Phase 4: rebuild Ollama chat tag (runner only) ==="
ollama create sonec -f Modelfile || true

echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "results: docs/results/COMPARE_REPORT.md"
echo "weights: artifacts/train/checkpoints/sonec-sft-mlx"
echo "author: Suryanshu Nabheet"
echo "log: $LOG"
echo "Next: SKIP_SFT=1 ./scripts/capabilitybench_e2e.sh  # decision board"
echo "      SKIP_GRPO=1 ./scripts/world_rl_leaderboard.sh  # multi-model board"
