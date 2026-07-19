#!/usr/bin/env bash
# CapabilityBench 200 end-to-end: optional densify → dual-serve → compare → 2B board.
# No heavy live GRPO.
#
# Usage:
#   ./scripts/capabilitybench_e2e.sh              # densify + full Cap200 (hours)
#   SKIP_SFT=1 ./scripts/capabilitybench_e2e.sh   # eval only (adapters already ready)
#   SFT_ITERS=120 GOLD_N=200 ./scripts/capabilitybench_e2e.sh
#
# Smoke (minutes, not Cap200):
#   sonec compare -s examples/benchmarks/ab_agent_2b_hard.json -o docs/results
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

mkdir -p artifacts/logs docs/results/leaderboard_cap docs/results
LOG="artifacts/logs/cap200_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== capabilitybench 200 pipeline ==="
echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "=== regenerate sealed CapabilityBench ==="
sonec capabilitybench

if [[ "${SKIP_SFT:-0}" == "1" ]]; then
  echo "=== densify skipped (SKIP_SFT=1) ==="
else
  echo "=== densify LoRA on expanded gold (offline, no live LLM) ==="
  python3 - <<'PY'
from pathlib import Path
import shutil
from sonec.training.specialize import assemble_sft_corpus

out = Path("artifacts/train/sft_corpus_cap")
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
empty = out / "empty.jsonl"
empty.write_text("", encoding="utf-8")
gold_n = int(__import__("os").environ.get("GOLD_N", "360"))
paths = assemble_sft_corpus(rollouts_jsonl=empty, out_dir=out / "corpus", gold_n=gold_n)
src = Path(paths["mlx_train"])
mlx = out / "mlx_data"
mlx.mkdir(exist_ok=True)
text = src.read_text(encoding="utf-8")
(mlx / "train.jsonl").write_text(text, encoding="utf-8")
lines = text.splitlines()
(mlx / "valid.jsonl").write_text("\n".join(lines[:48]) + "\n", encoding="utf-8")
print("train_lines", len(lines))
PY
  SFT_ITERS="${SFT_ITERS:-280}"
  python -m mlx_lm lora \
    --model mlx-community/Qwen3.5-2B-4bit \
    --train \
    --data artifacts/train/sft_corpus_cap/mlx_data \
    --adapter-path artifacts/train/checkpoints/sonec-sft-mlx \
    --batch-size 1 \
    --num-layers 8 \
    --iters "$SFT_ITERS" \
    --learning-rate 2e-5 \
    --steps-per-report 20 \
    --save-every 70 \
    --grad-checkpoint \
    --max-seq-length 2048
  echo "SFT done iters=$SFT_ITERS"
fi

echo "=== serve LoRA :8080 + base :8081 ==="
pkill -f "mlx_lm server.*8080" 2>/dev/null || true
pkill -f "mlx_lm server.*8081" 2>/dev/null || true
sleep 2
python -m mlx_lm server \
  --model mlx-community/Qwen3.5-2B-4bit \
  --adapter-path artifacts/train/checkpoints/sonec-sft-mlx \
  --host 127.0.0.1 --port 8080 \
  >artifacts/logs/serve_lora_cap.log 2>&1 &
echo $! >artifacts/logs/serve_lora_cap.pid
python -m mlx_lm server \
  --model mlx-community/Qwen3.5-2B-4bit \
  --host 127.0.0.1 --port 8081 \
  >artifacts/logs/serve_base_cap.log 2>&1 &
echo $! >artifacts/logs/serve_base_cap.pid

for i in $(seq 1 90); do
  if curl -sf http://127.0.0.1:8080/v1/models >/dev/null \
    && curl -sf http://127.0.0.1:8081/v1/models >/dev/null; then
    echo "both servers ready (${i}s)"
    break
  fi
  sleep 2
done
curl -sf http://127.0.0.1:8080/v1/models >/dev/null || { echo "LoRA serve failed"; exit 1; }
curl -sf http://127.0.0.1:8081/v1/models >/dev/null || { echo "base serve failed"; exit 1; }

echo "=== MLX compare on CapabilityBench ==="
PYTHONUNBUFFERED=1 sonec compare \
  --suite examples/benchmarks/capabilitybench_v1.json \
  --out docs/results \
  --lora-url http://127.0.0.1:8080/v1 \
  --base-url http://127.0.0.1:8081/v1

echo "=== resolve Ollama 2B arms ==="
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
out = Path("configs/leaderboard/arms_resolved.json")
out.write_text(
    json.dumps({**catalog, "arms": arms, "name": "2b-cap-resolved"}, indent=2),
    encoding="utf-8",
)
print("arms:", [a["name"] for a in arms])
PY

echo "=== full 200-task multi-model board ==="
OUT="${OUT:-docs/results/leaderboard_cap}"
mkdir -p "$OUT"
# Fresh board for CapabilityBench (do not reuse ab_agent_2b_hard dumps).
# Use find — zsh/bash globs fail (and abort set -e) when no arm_*.json exist yet.
find "$OUT" -maxdepth 1 -name 'arm_*.json' -delete 2>/dev/null || true
find docs/results -maxdepth 1 -name 'arm_*.json' -delete 2>/dev/null || true
PYTHONUNBUFFERED=1 sonec leaderboard \
  --suite examples/benchmarks/capabilitybench_v1.json \
  --arms configs/leaderboard/arms_resolved.json \
  --out "$OUT" \
  --fresh

# Mirror primary snapshot into leaderboard_2b for docs
cp -f "$OUT/LEADERBOARD.md" docs/results/leaderboard_2b/LEADERBOARD.md
cp -f "$OUT/LEADERBOARD.json" docs/results/leaderboard_2b/LEADERBOARD.json 2>/dev/null || true

echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "compare: docs/results/COMPARE_REPORT.md"
echo "board: $OUT/LEADERBOARD.md"
echo "log: $LOG"
cat docs/results/COMPARE_REPORT.md
echo "---"
cat "$OUT/LEADERBOARD.md"
