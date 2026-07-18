"""Head-to-head: specialized sonec (LoRA) vs unmodified base checkpoint.

Same harness, same suite, different inference endpoints.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sonec.app import build_runtime
from sonec.core.config import load_settings
from sonec.eval.harness import BenchmarkReport, EvalHarness

# Focused tool surface for 2B-class A/B (matches gold curriculum).
COMPARE_TOOL_ALLOWLIST = {
    "fs_write",
    "fs_read",
    "fs_edit",
    "fs_list",
    "terminal_run",
}


@dataclass
class ArmSpec:
    name: str
    base_url: str
    model: str
    kind: str  # "lora" | "base"


@dataclass
class CompareSummary:
    suite: str
    arms: list[dict[str, Any]]
    winner: str | None
    delta_pass_rate: float
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def run_arm(
    *,
    arm: ArmSpec,
    suite: Path,
    workspace_root: Path,
    out_path: Path,
) -> BenchmarkReport:
    """Run one live arm; fresh workspace per task."""
    data = json.loads(suite.read_text(encoding="utf-8"))
    name = data.get("name", suite.stem) if isinstance(data, dict) else suite.stem
    tasks = EvalHarness.load_tasks(suite)
    if isinstance(data, dict) and data.get("limit"):
        tasks = tasks[: int(data["limit"])]

    arm_root = (workspace_root / arm.name).resolve()
    if arm_root.exists():
        shutil.rmtree(arm_root)
    arm_root.mkdir(parents=True, exist_ok=True)

    results = []
    for task in tasks:
        ws = arm_root / task.id
        if ws.exists():
            shutil.rmtree(ws)
        ws.mkdir(parents=True, exist_ok=True)
        settings = load_settings(
            workspace=ws,
            provider="local",
            base_url=arm.base_url.rstrip("/"),
            model=arm.model,
            api_key="local",
        )
        harness = EvalHarness(workspace=ws)

        def factory(t, _settings=settings, _ws=ws):
            runtime, *_ = build_runtime(
                settings=_settings,
                provider=None,
                persist_memory=False,
                log_dir=_ws / ".trajectories",
                goal_for_prompt=t.prompt,
                tool_allowlist=COMPARE_TOOL_ALLOWLIST,
            )
            return runtime

        try:
            results.append(await harness.run_task(task, factory(task)))
        except Exception as exc:  # noqa: BLE001
            from sonec.eval.harness import EvalResult

            results.append(
                EvalResult(
                    task_id=task.id,
                    passed=False,
                    score=0.0,
                    details=[f"ERROR: {exc}"],
                    difficulty=task.difficulty,
                    tags=list(task.tags),
                )
            )
        await asyncio.sleep(0.75)

    from sonec.eval.harness import build_report

    report = build_report(f"{name}__{arm.name}", results)
    report.save(out_path)
    return report


def summarize(
    *,
    suite: Path,
    arms: list[ArmSpec],
    reports: dict[str, BenchmarkReport],
) -> CompareSummary:
    rows: list[dict[str, Any]] = []
    for arm in arms:
        r = reports[arm.name]
        rows.append(
            {
                "name": arm.name,
                "kind": arm.kind,
                "base_url": arm.base_url,
                "model": arm.model,
                "pass_rate": r.pass_rate,
                "passed": r.passed,
                "total": r.total,
                "mean_score": r.mean_score,
                "mean_duration_s": r.mean_duration_s,
                "by_difficulty": r.by_difficulty,
                "by_tag": r.by_tag,
            }
        )
    lora = next((x for x in rows if x["kind"] == "lora"), None)
    base = next((x for x in rows if x["kind"] == "base"), None)
    winner: str | None = None
    delta = 0.0
    note = "Same frozen harness; arms differ only by weights/endpoint."
    if lora and base:
        delta = float(lora["pass_rate"]) - float(base["pass_rate"])
        if delta > 1e-9:
            winner = lora["name"]
        elif delta < -1e-9:
            winner = base["name"]
        else:
            winner = None
            note += " Tie on pass_rate — inspect mean_score / per-task diffs."
    return CompareSummary(
        suite=str(suite),
        arms=rows,
        winner=winner,
        delta_pass_rate=delta,
        note=note,
    )


def write_compare_report(
    *,
    out_dir: Path,
    summary: CompareSummary,
    reports: dict[str, BenchmarkReport],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary.to_dict(),
        "per_arm_task_ids": {
            name: [
                {"task_id": r.task_id, "passed": r.passed, "score": r.score}
                for r in report.results
            ]
            for name, report in reports.items()
        },
    }
    path = out_dir / "COMPARE_REPORT.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = out_dir / "COMPARE_REPORT.md"
    lines = [
        "# sonec vs base — live compare",
        "",
        f"Suite: `{summary.suite}`",
        "",
        "| Arm | Kind | Pass rate | Passed | Mean score | Mean duration |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for a in summary.arms:
        lines.append(
            f"| {a['name']} | {a['kind']} | {a['pass_rate']:.0%} | "
            f"{a['passed']}/{a['total']} | {a['mean_score']:.2f} | "
            f"{a['mean_duration_s']:.1f}s |"
        )
    lines.extend(
        [
            "",
            f"**Winner:** {summary.winner or 'tie'}",
            f"**Delta pass_rate (lora − base):** {summary.delta_pass_rate:+.0%}",
            "",
            summary.note,
            "",
            "Product sonec = LoRA adapter served via sonec serve-llm.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_compare_sync(
    *,
    suite: Path,
    lora_url: str,
    base_url: str,
    lora_model: str,
    base_model: str,
    out_dir: Path,
    workspace_root: Path | None = None,
) -> CompareSummary:
    arms = [
        ArmSpec(name="sonec_lora", base_url=lora_url, model=lora_model, kind="lora"),
        ArmSpec(name="qwen35_2b_base", base_url=base_url, model=base_model, kind="base"),
    ]
    ws_root = workspace_root or Path(".sonec/compare-ws")
    reports: dict[str, BenchmarkReport] = {}

    async def _all() -> None:
        for arm in arms:
            reports[arm.name] = await run_arm(
                arm=arm,
                suite=suite,
                workspace_root=ws_root,
                out_path=out_dir / f"arm_{arm.name}.json",
            )

    asyncio.run(_all())
    summary = summarize(suite=suite, arms=arms, reports=reports)
    write_compare_report(out_dir=out_dir, summary=summary, reports=reports)
    return summary
