#!/usr/bin/env bash
# Strict 2B-only multi-model leaderboard + optional GRPO-lite.
# No 1B / 1.5B / 3B+ peers. Decision suite: ab_agent_2b_hard.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

mkdir -p artifacts/logs docs/results/leaderboard_2b
LOG="artifacts/logs/world_rl_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== world RL + strict 2B leaderboard ==="
echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

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

if [[ "${SKIP_GRPO:-0}" != "1" ]]; then
  echo "=== GRPO-lite (group-relative RL → LoRA) ==="
  sonec grpo --group-size 8 --train-n 24 --sft-iters 200 --live \
    || sonec grpo --group-size 8 --train-n 24 --sft-iters 200 --mock
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

SUITE="${SUITE:-examples/benchmarks/ab_agent_2b_hard.json}"
echo "=== multi-model leaderboard ($SUITE) ==="
# Default fresh board for 2B-only runs; RESUME=1 to keep arm dumps.
LB_FLAGS=(--suite "$SUITE" --arms "$ARMS_OUT" --out docs/results/leaderboard_2b)
if [[ "${RESUME:-0}" == "1" ]]; then
  LB_FLAGS+=(--resume)
else
  LB_FLAGS+=(--fresh)
fi
sonec leaderboard "${LB_FLAGS[@]}"

echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "leaderboard: docs/results/leaderboard_2b/LEADERBOARD.md"
echo "chart: docs/results/leaderboard_2b/LEADERBOARD_CHART.html"
echo "log: $LOG"
