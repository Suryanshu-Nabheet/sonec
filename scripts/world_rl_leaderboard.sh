#!/usr/bin/env bash
# Strict 2B-only multi-model leaderboard + optional light GRPO-lite.
# Decision suite default: CapabilityBench (200 sealed tasks).
# GRPO is OFF by default — large live GRPO thrashs Macs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

mkdir -p artifacts/logs docs/results/leaderboard_2b examples/benchmarks
LOG="artifacts/logs/world_rl_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== world RL + strict 2B leaderboard ==="
echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Ensure sealed CapabilityBench exists
if [[ ! -f examples/benchmarks/capabilitybench_v1.json ]]; then
  echo "=== generate capabilitybench_v1.json ==="
  sonec capabilitybench
fi

# Exact ~2B tags only.
MODELS=(
  "qwen3.5:2b"
  "gemma2:2b"
  "codegemma:2b"
)

echo "=== pull strict 2B rivals ==="
for m in "${MODELS[@]}"; do
  if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$m"; then
    echo "have $m"
  else
    echo "ollama pull $m"
    ollama pull "$m" || echo "WARN: failed to pull $m (skipped)"
  fi
done
ollama list

echo "=== resolve arms from catalog ∩ installed 2B models ==="
ARMS_OUT="configs/leaderboard/arms_resolved.json"
python - <<'PY'
import json, subprocess
from pathlib import Path

catalog = json.loads(Path("configs/leaderboard/arms_2b.json").read_text(encoding="utf-8"))
listed = subprocess.check_output(["ollama", "list"], text=True)
installed = {line.split()[0] for line in listed.splitlines()[1:] if line.strip()}
arms = []
for a in catalog["arms"]:
    if a.get("kind") == "lora" or a["model"] in installed:
        arms.append(a)
    else:
        print(f"skip missing model: {a['model']}")
Path("configs/leaderboard/arms_resolved.json").write_text(
    json.dumps({**catalog, "arms": arms, "name": "2b-only-resolved"}, indent=2),
    encoding="utf-8",
)
print("arms:", [a["name"] for a in arms])
assert all(
    "1.5" not in a["model"] and ":1b" not in a["model"] and ":3b" not in a["model"]
    for a in arms
    if a.get("kind") != "lora"
), "non-2B arm slipped into resolved catalog"
PY

echo "=== serve specialized sonec (LoRA) on :8080 ==="
pkill -f "mlx_lm server.*8080" 2>/dev/null || true
sleep 1
python -m mlx_lm server \
  --model mlx-community/Qwen3.5-2B-4bit \
  --adapter-path artifacts/train/checkpoints/sonec-sft-mlx \
  --host 127.0.0.1 --port 8080 \
  >artifacts/logs/serve_lora_world.log 2>&1 &
echo $! >artifacts/logs/serve_lora_world.pid
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8080/v1/models >/dev/null; then
    echo "sonec ready (${i}s)"
    break
  fi
  sleep 2
done

if [[ "${SKIP_GRPO:-1}" != "1" ]]; then
  echo "=== GRPO-lite light densify (default off — set SKIP_GRPO=0) ==="
  # Laptop-safe: mock G=2 n=8. Live only if LIVE_GRPO=1 and still capped.
  if [[ "${LIVE_GRPO:-0}" == "1" ]]; then
    sonec grpo --group-size 2 --train-n 8 --sft-iters 80 --live \
      || sonec grpo --group-size 2 --train-n 8 --sft-iters 80 --mock
  else
    sonec grpo --group-size 2 --train-n 8 --sft-iters 80 --mock
  fi
  pkill -f "mlx_lm server.*8080" 2>/dev/null || true
  sleep 2
  python -m mlx_lm server \
    --model mlx-community/Qwen3.5-2B-4bit \
    --adapter-path artifacts/train/checkpoints/sonec-sft-mlx \
    --host 127.0.0.1 --port 8080 \
    >artifacts/logs/serve_lora_world.log 2>&1 &
  echo $! >artifacts/logs/serve_lora_world.pid
  for i in $(seq 1 60); do
    curl -sf http://127.0.0.1:8080/v1/models >/dev/null && break
    sleep 2
  done
fi

SUITE="${SUITE:-examples/benchmarks/capabilitybench_v1.json}"
echo "=== multi-model leaderboard ($SUITE) ==="
# Default resume for 200-task runs (abort-safe). FORCE_FRESH=1 to re-run all arms.
LB_FLAGS=(--suite "$SUITE" --arms "$ARMS_OUT" --out docs/results/leaderboard_2b)
if [[ "${FORCE_FRESH:-0}" == "1" ]]; then
  LB_FLAGS+=(--fresh)
else
  LB_FLAGS+=(--resume)
fi
sonec leaderboard "${LB_FLAGS[@]}"

echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "leaderboard: docs/results/leaderboard_2b/LEADERBOARD.md"
echo "chart: docs/results/leaderboard_2b/LEADERBOARD_CHART.html"
echo "log: $LOG"
echo "Note: GRPO is OFF by default (SKIP_GRPO=1). Optional light mock densify:"
echo "  SKIP_GRPO=0 ./scripts/world_rl_leaderboard.sh"
echo "  # or: sonec grpo --mock   (never use large --live G on a laptop)"
