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
from sonec.harness.versioning import CORE_TOOL_NAMES

# Align live A/B with the frozen CORE tool surface used for training rollouts.
# (Previously a write-first subset understated Cap200 readonly / search skill.)
COMPARE_TOOL_ALLOWLIST = set(CORE_TOOL_NAMES)


@dataclass
class ArmSpec:
    name: str
    base_url: str
    model: str
    kind: str = "external"  # lora | base | external
    provider: str = "local"
    api_key: str = "local"


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
    limit: int | None = None,
) -> BenchmarkReport:
    """Run one live arm; fresh workspace per task."""
    data = json.loads(suite.read_text(encoding="utf-8"))
    name = data.get("name", suite.stem) if isinstance(data, dict) else suite.stem
    tasks = EvalHarness.load_tasks(suite)
    if isinstance(data, dict) and data.get("limit"):
        tasks = tasks[: int(data["limit"])]
    if limit is not None and limit > 0:
        tasks = tasks[:limit]

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
            provider=arm.provider,
            base_url=arm.base_url.rstrip("/"),
            model=arm.model,
            api_key=arm.api_key,
        )
        harness = EvalHarness(workspace=ws)

        def factory(t, _settings=settings, _ws=ws, _arm_root=arm_root):
            # Keep trajectory logs outside the graded workspace so only_files
            # restraint checks are not poisoned by .trajectories/*.jsonl.
            runtime, *_ = build_runtime(
                settings=_settings,
                provider=None,
                persist_memory=False,
                log_dir=_arm_root / "_logs" / t.id,
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
    speed_ratio: float | None = None
    note = "Same frozen harness; arms differ only by weights/endpoint."
    if lora and base:
        delta = float(lora["pass_rate"]) - float(base["pass_rate"])
        ld = float(lora["mean_duration_s"] or 0.0)
        bd = float(base["mean_duration_s"] or 0.0)
        if ld > 0 and bd > 0:
            speed_ratio = bd / ld
        if delta > 1e-9:
            winner = lora["name"]
        elif delta < -1e-9:
            winner = base["name"]
        else:
            # Pass-rate tie: prefer clear speed win, else declare tie.
            if speed_ratio is not None and speed_ratio >= 1.15:
                winner = lora["name"]
                note += (
                    f" Pass-rate tie; sonec faster "
                    f"(base/lora duration ≈ {speed_ratio:.2f}×)."
                )
            elif speed_ratio is not None and speed_ratio <= (1 / 1.15):
                winner = base["name"]
                note += (
                    f" Pass-rate tie; base faster "
                    f"(base/lora duration ≈ {speed_ratio:.2f}×)."
                )
            else:
                winner = None
                note += " Tie on pass_rate — inspect mean_score / per-task diffs."
                if speed_ratio is not None:
                    note += f" Speed ratio base/lora ≈ {speed_ratio:.2f}×."
    for row in rows:
        if speed_ratio is not None and row.get("kind") in {"lora", "base"}:
            row["speed_ratio_base_over_lora"] = round(speed_ratio, 4)
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
        f"Generated: `{payload['generated_at']}`",
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
    speed = None
    for a in summary.arms:
        if a.get("speed_ratio_base_over_lora") is not None:
            speed = a["speed_ratio_base_over_lora"]
            break
    lines.extend(
        [
            "",
            f"**Winner:** {summary.winner or 'tie'}",
            f"**Delta pass_rate (lora − base):** {summary.delta_pass_rate:+.0%}",
        ]
    )
    if speed is not None:
        lines.append(f"**Speed (base / lora duration):** {speed:.2f}×")
    lines.extend(
        [
            "",
            summary.note,
            "",
            "Product sonec = LoRA adapter served via sonec serve-llm.",
            "Author: Suryanshu Nabheet. Smoke suites may saturate — prefer CapabilityBench 200 for pass-rate claims.",
            "",
            "## Per-task",
            "",
        ]
    )
    for name, report in reports.items():
        lines.append(f"### {name}")
        lines.append("")
        for r in report.results:
            flag = "PASS" if r.passed else "FAIL"
            lines.append(f"- `{r.task_id}`: {flag} ({r.score:.2f})")
        lines.append("")
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
