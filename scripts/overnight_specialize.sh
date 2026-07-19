#!/usr/bin/env bash
# Overnight specialization pipeline for sonec.
# Oracle-graded trajectories (structured tool_calls) → LoRA SFT → A/B compare.
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

echo "=== Phase 1: corpus + LoRA SFT ==="
# Reset specialization artifacts only (logs live under artifacts/logs/).
rm -rf artifacts/train
sonec train --step \
  --mock-fuel --mock-rl \
  --sft-iters 500 \
  --gold-n 240 \
  --train-n 64 \
  --rollout-group 4

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

echo "=== Phase 3: A/B compare ==="
sonec compare \
  --suite examples/benchmarks/ab_agent_v1.json \
  --out docs/results \
  --lora-url http://127.0.0.1:8080/v1 \
  --base-url http://127.0.0.1:8081/v1

echo "=== Phase 4: rebuild Ollama chat tag ==="
ollama create sonec -f Modelfile || true

echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "results: docs/results/COMPARE_REPORT.md"
echo "weights: artifacts/train/checkpoints/sonec-sft-mlx"
echo "log: $LOG"
