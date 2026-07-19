"""Group-relative RL (GRPO-lite) for Apple Silicon / MLX.

Full TRL/verl GRPO needs CUDA + a separate trainer. On MLX we implement the
same *signal*: for each prompt, sample G rollouts, score with harness rewards,
compute group-relative advantages, densify positive-advantage trajectories,
then continue LoRA SFT. Policy improves from relative wins — not mock chat.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sonec.eval.trainbench import build_trainbench_tasks
from sonec.models import BASE_HF_MLX
from sonec.training.export import _trajectory_to_messages
from sonec.training.rollouts import run_rollouts_sync
from sonec.training.specialize import TrainReport, _write_json, run_mlx_sft


@dataclass
class GrpoLiteResult:
    prompts: int
    rollouts: int
    positive_advantage: int
    absolute_passers: int
    corpus_lines: int
    sft: TrainReport
    stats_path: Path


def _group_advantages(rewards: list[float]) -> list[float]:
    if not rewards:
        return []
    mean = statistics.fmean(rewards)
    # Dr.GRPO-style: relative to group mean (no length std).
    return [r - mean for r in rewards]


def run_grpo_lite(
    *,
    root: Path,
    group_size: int = 8,
    train_n: int = 24,
    sft_iters: int = 200,
    live: bool = True,
    model: str | None = None,
    mlx_model: str = BASE_HF_MLX,
    adapter_path: Path | None = None,
    oversample_pos: int = 3,
) -> GrpoLiteResult:
    """Roll out G completions per TrainBench prompt → advantage-weighted SFT."""
    art = root / "artifacts" / "train" / "grpo_lite"
    art.mkdir(parents=True, exist_ok=True)
    adapter = adapter_path or (root / "artifacts" / "train" / "checkpoints" / "sonec-sft-mlx")
    inference_model = model or mlx_model

    tasks = build_trainbench_tasks(n=train_n)
    # Prefer write-first / python / architecture fuel for agentic RL.
    focus = [
        t
        for t in tasks
        if any(tag in (t.tags or []) for tag in ("write_first", "python", "architecture", "multifile"))
    ]
    if len(focus) < max(8, train_n // 3):
        focus = tasks[:train_n]
    else:
        focus = focus[:train_n]

    records = run_rollouts_sync(
        focus,
        art / "rollouts",
        group_size=group_size,
        use_mock=not live,
        provider_name="local",
        model=inference_model,
    )

    by_task: dict[str, list] = {}
    for r in records:
        by_task.setdefault(r.task_id, []).append(r)

    selected: list[dict[str, Any]] = []
    n_pos = 0
    n_pass = 0
    group_stats: list[dict[str, Any]] = []
    for tid, group in by_task.items():
        rewards = [float(g.reward) for g in group]
        adv = _group_advantages(rewards)
        group_stats.append(
            {
                "task_id": tid,
                "n": len(group),
                "mean_reward": statistics.fmean(rewards) if rewards else 0.0,
                "pass_rate": sum(1 for g in group if g.passed) / max(len(group), 1),
            }
        )
        for rec, a in zip(group, adv, strict=True):
            if rec.passed:
                n_pass += 1
            if a <= 0 and not rec.passed:
                continue
            if a > 0:
                n_pos += 1
            messages = _trajectory_to_messages(rec.trajectory_path)
            if not messages:
                continue
            weight = oversample_pos if a > 0 else 1
            row = {
                "id": f"grpo-{tid}-{rec.rollout_index}",
                "task_id": tid,
                "messages": messages,
                "reward": rec.reward,
                "advantage": a,
                "passed": rec.passed,
            }
            for _ in range(weight):
                selected.append(row)

    mlx_dir = art / "mlx_data"
    mlx_dir.mkdir(parents=True, exist_ok=True)
    train = mlx_dir / "train.jsonl"
    with train.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")
    lines = train.read_text(encoding="utf-8").splitlines() if train.exists() else []
    if not lines:
        raise RuntimeError(
            "GRPO-lite produced empty corpus — need live inference with some positive rewards. "
            "Ensure sonec serve-llm is up and SONEC_BASE_URL points at it."
        )
    (mlx_dir / "valid.jsonl").write_text(
        "\n".join(lines[: max(1, len(lines) // 10)]) + "\n", encoding="utf-8"
    )

    sft = run_mlx_sft(
        data_dir=mlx_dir,
        adapter_path=adapter,
        model=mlx_model,
        iters=sft_iters,
        learning_rate=2e-5,
        lora_layers=12,
    )
    stats_path = art / "grpo_stats.json"
    _write_json(
        stats_path,
        {
            "algorithm": "grpo_lite_group_relative_advantage_sft",
            "group_size": group_size,
            "prompts": len(by_task),
            "rollouts": len(records),
            "positive_advantage": n_pos,
            "absolute_passers": n_pass,
            "corpus_lines": len(lines),
            "live": live,
            "model": inference_model,
            "sft": sft.__dict__,
            "groups": group_stats,
            "elapsed_note": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    return GrpoLiteResult(
        prompts=len(by_task),
        rollouts=len(records),
        positive_advantage=n_pos,
        absolute_passers=n_pass,
        corpus_lines=len(lines),
        sft=sft,
        stats_path=stats_path,
    )
