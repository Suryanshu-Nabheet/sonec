"""Iterative specialization loop — SFT + rejection RL for sonec.

Start small, repeat: fuel → corpus → LoRA → rejection winners → next step.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sonec.eval.trainbench import build_trainbench_tasks, write_trainbench
from sonec.models import BASE_HF, BASE_HF_MLX, BASE_MODEL, PRODUCT_MODEL
from sonec.training.export import export_from_rollouts
from sonec.training.pipeline import DatasetGenerator, TrainingPipeline
from sonec.training.rollouts import run_rollouts_sync


@dataclass
class TrainReport:
    phase: str
    ok: bool
    detail: str
    paths: dict[str, str]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_gold_agent_examples(gen: DatasetGenerator, *, n: int = 40) -> int:
    """Compact tool-use curriculum for agentic coding."""
    templates = [
        (
            "localize-then-patch",
            "Fix the off-by-one in util.py and verify with pytest.",
            [
                {"role": "user", "content": "Fix the off-by-one in util.py and verify with pytest."},
                {"role": "assistant", "content": "Calling tool index_search", "name": "index_search"},
                {"role": "tool", "name": "index_search", "content": "util.py:12 count_to"},
                {"role": "assistant", "content": "Calling tool fs_read", "name": "fs_read"},
                {"role": "tool", "name": "fs_read", "content": "return list(range(n+1))"},
                {"role": "assistant", "content": "Calling tool fs_edit", "name": "fs_edit"},
                {"role": "tool", "name": "fs_edit", "content": "patched range(n)"},
                {"role": "assistant", "content": "Calling tool terminal_run", "name": "terminal_run"},
                {"role": "tool", "name": "terminal_run", "content": "exit_code: 0\n1 passed"},
                {
                    "role": "assistant",
                    "content": "Fixed util.py off-by-one. Verification: pytest exit 0, 1 passed.",
                },
            ],
        ),
        (
            "restraint-question",
            "What is a regression test? Do not edit files.",
            [
                {"role": "user", "content": "What is a regression test? Do not edit files."},
                {
                    "role": "assistant",
                    "content": (
                        "A regression test checks that previously working behavior "
                        "still passes after a change. No files edited."
                    ),
                },
            ],
        ),
        (
            "verify-before-done",
            "Add scripts/verify.sh and document it in VERIFY.md.",
            [
                {"role": "user", "content": "Add scripts/verify.sh and document it in VERIFY.md."},
                {"role": "assistant", "content": "Calling tool fs_write", "name": "fs_write"},
                {"role": "tool", "name": "fs_write", "content": "wrote scripts/verify.sh"},
                {"role": "assistant", "content": "Calling tool fs_write", "name": "fs_write"},
                {"role": "tool", "name": "fs_write", "content": "wrote VERIFY.md"},
                {"role": "assistant", "content": "Calling tool terminal_run", "name": "terminal_run"},
                {"role": "tool", "name": "terminal_run", "content": "exit_code: 0\nOK"},
                {
                    "role": "assistant",
                    "content": "Added verify script and docs. Verification: ./scripts/verify.sh exit 0.",
                },
            ],
        ),
    ]
    created = 0
    while created < n:
        for key, task, messages in templates:
            if created >= n:
                break
            gen.add_from_messages(
                example_id=f"gold-{key}-{created:04d}",
                task=task,
                messages=messages,
                outcome="success",
                metadata={"source": "gold_curriculum", "curriculum": key},
            )
            created += 1
    return created


def prepare_rollout_fuel(
    out_dir: Path,
    *,
    group_size: int = 2,
    train_n: int = 16,
    limit: int = 0,
) -> Path:
    write_trainbench(Path("examples/benchmarks/trainbench_v1.json"), n=max(train_n, 16))
    tasks = build_trainbench_tasks(n=train_n)
    if limit:
        tasks = tasks[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    records = run_rollouts_sync(tasks, out_dir, group_size=group_size, use_mock=True)
    _write_json(
        out_dir / "fuel_meta.json",
        {
            "suite": "trainbench",
            "tasks": len(tasks),
            "group_size": group_size,
            "records": len(records),
            "passed": sum(1 for r in records if r.passed),
        },
    )
    return out_dir / "rollouts.jsonl"


def assemble_sft_corpus(
    *,
    rollouts_jsonl: Path,
    out_dir: Path,
    gold_n: int = 40,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sealed: set[str] = set()
    for suite in (
        Path("examples/benchmarks/sonecbench_v1.json"),
        Path("examples/benchmarks/worldbench_v1.json"),
    ):
        if suite.exists():
            data = json.loads(suite.read_text(encoding="utf-8"))
            for t in data.get("tasks") or []:
                sealed.add(t["id"])

    written = export_from_rollouts(rollouts_jsonl, out_dir / "from_rollouts", sealed_ids=sealed)
    gen = DatasetGenerator("sonec-sft")
    generate_gold_agent_examples(gen, n=gold_n)
    rollout_i = 0
    chat_path = written.get("chat")
    if chat_path and chat_path.exists():
        for line in chat_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            msgs = [
                {"role": m["role"], "content": m.get("content") or "", "name": m.get("name")}
                for m in row.get("messages") or []
            ]
            if len(msgs) < 2:
                continue
            gen.add_from_messages(
                example_id=str(row.get("id") or f"rollout-{rollout_i:05d}"),
                task=str(row.get("task_id") or "rollout"),
                messages=msgs,
                outcome="success",
                metadata={"source": "rollout"},
            )
            rollout_i += 1

    pipeline = TrainingPipeline(out_dir)
    manifest = gen.manifest()
    jsonl = pipeline.export_jsonl(manifest, filename="train_chat.jsonl")
    mlx_dir = out_dir / "mlx_data"
    mlx_dir.mkdir(parents=True, exist_ok=True)
    mlx_train = mlx_dir / "train.jsonl"
    with mlx_train.open("w", encoding="utf-8") as handle:
        for ex in manifest.examples:
            messages = [{"role": s.role, "content": s.content} for s in ex.trajectory]
            handle.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
    lines = mlx_train.read_text(encoding="utf-8").splitlines()
    (mlx_dir / "valid.jsonl").write_text(
        "\n".join(lines[: max(1, len(lines) // 10)]) + "\n", encoding="utf-8"
    )
    manifest.save(out_dir / "manifest.json")
    pipeline.write_config(model=PRODUCT_MODEL, dataset_file="train_chat.jsonl")
    _write_json(
        out_dir / "corpus_stats.json",
        {"examples": len(manifest.examples), "gold_n": gold_n, "mlx_train": str(mlx_train)},
    )
    return {"chat": jsonl, "mlx_train": mlx_train, "mlx_dir": mlx_dir}


def resolve_mlx_base(preferred: str | None = None) -> str:
    return preferred or BASE_HF_MLX


def run_mlx_sft(
    *,
    data_dir: Path,
    adapter_path: Path,
    model: str,
    iters: int = 80,
    batch_size: int = 1,
    lora_layers: int = 8,
    learning_rate: float = 1e-5,
) -> TrainReport:
    adapter_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        model,
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(adapter_path),
        "--batch-size",
        str(batch_size),
        "--num-layers",
        str(lora_layers),
        "--iters",
        str(iters),
        "--learning-rate",
        str(learning_rate),
        "--steps-per-report",
        "10",
        "--save-every",
        "40",
        "--grad-checkpoint",
        "--max-seq-length",
        "2048",
    ]
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        return TrainReport("sft", False, str(exc), {})
    log_path = adapter_path / "sft_log.txt"
    log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
    ok = proc.returncode == 0 and (
        (adapter_path / "adapters.safetensors").exists()
        or (adapter_path / "adapter_config.json").exists()
        or any(adapter_path.glob("*.safetensors"))
    )
    return TrainReport(
        phase="sft",
        ok=ok,
        detail=(
            f"iters={iters} model={model} elapsed_s={time.perf_counter() - started:.1f} "
            f"rc={proc.returncode}"
        ),
        paths={"adapter": str(adapter_path), "log": str(log_path)},
    )


def run_rl_rejection_round(
    *,
    out_dir: Path,
    group_size: int = 2,
    limit: int = 8,
    live: bool = False,
    model: str = PRODUCT_MODEL,
) -> TrainReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_trainbench_tasks(n=max(limit, 8))[:limit]
    records = run_rollouts_sync(
        tasks,
        out_dir / "rollouts",
        group_size=group_size,
        use_mock=not live,
        provider_name="local",
        model=model,
    )
    by_task: dict[str, list] = {}
    for r in records:
        by_task.setdefault(r.task_id, []).append(r)
    winners = []
    for group in by_task.values():
        best = max(group, key=lambda x: (x.reward, -x.duration_s))
        if best.passed:
            winners.append(best)
    winners_path = out_dir / "winners.jsonl"
    with winners_path.open("w", encoding="utf-8") as handle:
        for w in winners:
            handle.write(json.dumps(w.to_json()) + "\n")
    if winners:
        export_from_rollouts(
            out_dir / "rollouts" / "rollouts.jsonl",
            out_dir / "rft_export",
            sealed_ids=set(),
        )
    _write_json(
        out_dir / "rl_stats.json",
        {
            "groups": len(by_task),
            "records": len(records),
            "winners": len(winners),
            "pass_rate": sum(1 for r in records if r.passed) / max(len(records), 1),
            "live": live,
        },
    )
    return TrainReport(
        phase="rl_rejection",
        ok=True,
        detail=f"winners={len(winners)}/{len(by_task)} groups",
        paths={"winners": str(winners_path), "stats": str(out_dir / "rl_stats.json")},
    )


def write_product_modelfile(*, adapter_path: Path, modelfile_out: Path) -> Path:
    """Write an explicit non-product note. Modelfile alone is never sonec."""
    from sonec.training.weights import weight_status

    status = weight_status(adapter_path)
    body = f"""# NOT THE PRODUCT
# A SYSTEM prompt on top of {BASE_MODEL} is a wrapper, not sonec.
# sonec = LoRA adapter weights under: {adapter_path}
# Ready: {status.ready} — {status.detail}
#
# Serve specialized weights:
#   sonec serve-llm
#   # or: python -m mlx_lm server --model {BASE_HF_MLX} --adapter-path {adapter_path} --port 8080
#
# Do not treat this file as a specialized model.

FROM {BASE_MODEL}

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER num_ctx 32768

SYSTEM \"\"\"You are sonec — a coding-agent model on Qwen 3.5.
This prompt is harness identity only. Specialized behavior requires trained adapter weights.
\"\"\"
"""
    modelfile_out.write_text(body, encoding="utf-8")
    return modelfile_out


def run_train_step(
    *,
    root: Path,
    sft_iters: int = 80,
    gold_n: int = 40,
    train_n: int = 16,
    rollout_group: int = 2,
    rl_group: int = 2,
    rl_limit: int = 8,
    skip_sft: bool = False,
    skip_fuel: bool = False,
    live_rl: bool = False,
    mlx_model: str | None = None,
    reset: bool = False,
    corpus_dir: Path | None = None,
) -> list[TrainReport]:
    """One small specialization step. Product is ready only when adapter *.safetensors exist."""
    from sonec.training.weights import write_product_manifest, weight_status

    reports: list[TrainReport] = []
    art = root / "artifacts" / "train"
    if reset and art.exists():
        shutil.rmtree(art)
    art.mkdir(parents=True, exist_ok=True)

    adapter = art / "checkpoints" / "sonec-sft-mlx"
    model = resolve_mlx_base(mlx_model)

    if corpus_dir is not None:
        corpus_path = corpus_dir.expanduser().resolve()
        paths = {
            "mlx_dir": corpus_path if (corpus_path / "train.jsonl").exists() else corpus_path / "mlx_data",
            "mlx_train": (
                corpus_path / "train.jsonl"
                if (corpus_path / "train.jsonl").exists()
                else corpus_path / "mlx_data" / "train.jsonl"
            ),
        }
        reports.append(
            TrainReport("corpus", True, f"reused {paths['mlx_dir']}", {k: str(v) for k, v in paths.items()})
        )
    else:
        if not skip_fuel:
            fuel_dir = art / "fuel"
            rollouts = prepare_rollout_fuel(fuel_dir, group_size=rollout_group, train_n=train_n)
            reports.append(TrainReport("fuel", True, f"rollouts={rollouts}", {"rollouts": str(rollouts)}))
        else:
            rollouts = art / "fuel" / "rollouts.jsonl"
            if not rollouts.exists():
                reports.append(TrainReport("fuel", False, f"missing {rollouts}", {}))
                return reports
            reports.append(TrainReport("fuel", True, f"reused {rollouts}", {"rollouts": str(rollouts)}))

        corpus_out = art / "sft_corpus"
        paths = assemble_sft_corpus(rollouts_jsonl=rollouts, out_dir=corpus_out, gold_n=gold_n)
        reports.append(
            TrainReport(
                "corpus",
                True,
                f"examples in {paths['mlx_dir']}",
                {k: str(v) for k, v in paths.items()},
            )
        )

    if not skip_sft:
        sft_report = run_mlx_sft(
            data_dir=Path(paths["mlx_dir"]),
            adapter_path=adapter,
            model=model,
            iters=sft_iters,
        )
        status = weight_status(adapter)
        if sft_report.ok and not status.ready:
            sft_report = TrainReport(
                "sft",
                False,
                f"{sft_report.detail}; NO WEIGHTS: {status.detail}",
                sft_report.paths,
            )
        reports.append(sft_report)
    else:
        reports.append(TrainReport("sft", True, "skipped", {}))

    reports.append(
        run_rl_rejection_round(
            out_dir=art / "rl",
            group_size=rl_group,
            limit=rl_limit,
            live=live_rl,
            model=PRODUCT_MODEL,
        )
    )

    manifest = write_product_manifest(adapter_dir=adapter, mlx_base=model, root=root)
    status = weight_status(adapter)
    reports.append(
        TrainReport(
            "product",
            status.ready,
            status.detail if status.ready else f"NOT READY — {status.detail}",
            {"manifest": str(manifest), "adapter": str(adapter)},
        )
    )

    # Keep Modelfile as an explicit anti-wrapper note, not product definition.
    mf = write_product_modelfile(adapter_path=adapter, modelfile_out=root / "Modelfile")
    reports.append(TrainReport("modelfile_note", True, f"wrote non-product note {mf}", {"modelfile": str(mf)}))

    _write_json(
        art / "TRAIN_REPORT.json",
        {
            "product": PRODUCT_MODEL,
            "base": BASE_MODEL,
            "base_hf": BASE_HF,
            "mlx_base": model,
            "weights_ready": status.ready,
            "phases": [r.__dict__ for r in reports],
        },
    )
    return reports


# Back-compat name
run_enterprise_train = run_train_step
