"""Multi-arm agent leaderboard — any OpenAI-compatible endpoints."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sonec.eval.compare import ArmSpec, run_arm
from sonec.eval.harness import BenchmarkReport


@dataclass
class LeaderboardSummary:
    suite: str
    arms: list[dict[str, Any]]
    winner: str | None
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_arms(path: Path) -> list[ArmSpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("arms") if isinstance(data, dict) else data
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"No arms in {path}")
    arms: list[ArmSpec] = []
    for row in raw:
        base_url = str(row["base_url"]).rstrip("/")
        # local provider accepts any OpenAI-compatible URL (incl. Ollama :11434).
        provider = str(row.get("provider") or "local")
        arms.append(
            ArmSpec(
                name=str(row["name"]),
                base_url=base_url,
                model=str(row["model"]),
                kind=str(row.get("kind") or "external"),
                provider=provider,
                api_key=str(row.get("api_key") or "local"),
            )
        )
    return arms


def rank_arms(reports: dict[str, BenchmarkReport], arms: list[ArmSpec]) -> LeaderboardSummary:
    rows: list[dict[str, Any]] = []
    for arm in arms:
        report = reports[arm.name]
        rows.append(
            {
                "name": arm.name,
                "kind": arm.kind,
                "base_url": arm.base_url,
                "model": arm.model,
                "pass_rate": report.pass_rate,
                "passed": report.passed,
                "total": report.total,
                "mean_score": report.mean_score,
                "mean_duration_s": report.mean_duration_s,
                "by_difficulty": report.by_difficulty,
                "by_tag": report.by_tag,
            }
        )
    # Prefer specialized sonec (lora) on ties; then higher score; then speed.
    rows.sort(
        key=lambda r: (
            r["pass_rate"],
            r["mean_score"],
            1 if r["kind"] == "lora" else 0,
            -r["mean_duration_s"],
        ),
        reverse=True,
    )
    winner = rows[0]["name"] if rows else None
    return LeaderboardSummary(
        suite="",
        arms=rows,
        winner=winner,
        note="Ranked by pass_rate, then mean_score, then speed. Same frozen harness.",
    )


def write_leaderboard(
    *,
    out_dir: Path,
    suite: Path,
    summary: LeaderboardSummary,
    reports: dict[str, BenchmarkReport],
    catalog: dict[str, Any] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.suite = str(suite)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary.to_dict(),
        "catalog": catalog or {},
        "per_arm_task_ids": {
            name: [
                {"task_id": r.task_id, "passed": r.passed, "score": r.score}
                for r in report.results
            ]
            for name, report in reports.items()
        },
    }
    path = out_dir / "LEADERBOARD.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = out_dir / "LEADERBOARD.md"
    lines = [
        "# 2B-class agent leaderboard",
        "",
        f"Suite: `{suite}`",
        "",
        f"**Winner:** `{summary.winner}`",
        "",
        "| Rank | Model | Kind | Pass rate | Passed | Mean score | Mean duration |",
        "| ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for i, a in enumerate(summary.arms, start=1):
        lines.append(
            f"| {i} | {a['name']} | {a['kind']} | {a['pass_rate']:.0%} | "
            f"{a['passed']}/{a['total']} | {a['mean_score']:.2f} | "
            f"{a['mean_duration_s']:.1f}s |"
        )
    lines.extend(["", summary.note, ""])
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    chart = {
        "title": "Agent pass rate — 2B-class models",
        "suite": str(suite),
        "labels": [a["name"] for a in summary.arms],
        "pass_rates": [round(a["pass_rate"] * 100, 1) for a in summary.arms],
        "mean_scores": [round(a["mean_score"], 3) for a in summary.arms],
        "mean_duration_s": [round(a["mean_duration_s"], 2) for a in summary.arms],
        "kinds": [a["kind"] for a in summary.arms],
    }
    (out_dir / "chart_data.json").write_text(json.dumps(chart, indent=2), encoding="utf-8")

    # Self-contained HTML chart (no CDN required for axes; inline SVG bars).
    bars = []
    max_w = 420
    for i, a in enumerate(summary.arms):
        w = int(max_w * a["pass_rate"])
        y = 28 + i * 36
        color = "#16a34a" if a["kind"] == "lora" else "#64748b"
        bars.append(
            f'<text x="8" y="{y}" font-size="12" fill="#0f172a">{a["name"]}</text>'
            f'<rect x="160" y="{y - 12}" width="{w}" height="18" fill="{color}" rx="3"/>'
            f'<text x="{170 + w}" y="{y}" font-size="12" fill="#0f172a">{a["pass_rate"]:.0%}</text>'
        )
    height = 28 + len(summary.arms) * 36 + 24
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>sonec 2B leaderboard</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:32px;background:#f8fafc;color:#0f172a}}
h1{{font-size:20px;margin:0 0 8px}}
p{{color:#475569;margin:0 0 24px}}
svg{{background:#fff;border:1px solid #e2e8f0;border-radius:8px}}
</style></head><body>
<h1>Agent pass rate — 2B-class models</h1>
<p>Suite: {suite} · Winner: {summary.winner} · Source: LEADERBOARD.json</p>
<svg width="640" height="{height}" xmlns="http://www.w3.org/2000/svg">
{''.join(bars)}
</svg>
</body></html>
"""
    (out_dir / "LEADERBOARD_CHART.html").write_text(html, encoding="utf-8")
    return path


def _load_cached_arm(path: Path) -> BenchmarkReport | None:
    """Load a prior arm dump if it looks complete enough to resume from."""
    if not path.is_file():
        return None
    try:
        report = BenchmarkReport.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if report.total <= 0 or not report.results:
        return None
    return report


def _arm_reachable(arm: ArmSpec, timeout: float = 3.0) -> str | None:
    """Return None if /models responds; else an error string."""
    import httpx

    url = arm.base_url.rstrip("/") + "/models"
    try:
        r = httpx.get(url, timeout=timeout)
        if r.status_code >= 400:
            return f"HTTP {r.status_code} at {url}"
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


def run_leaderboard_sync(
    *,
    suite: Path,
    arms: list[ArmSpec],
    out_dir: Path,
    workspace_root: Path | None = None,
    catalog: dict[str, Any] | None = None,
    resume: bool = True,
    limit: int | None = None,
) -> LeaderboardSummary:
    """Rank all arms. With resume=True (default), reuse existing arm_*.json dumps."""
    ws_root = workspace_root or Path(".sonec/leaderboard-ws")
    out_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, BenchmarkReport] = {}

    async def _all() -> None:
        for arm in arms:
            out_path = out_dir / f"arm_{arm.name}.json"
            if resume:
                cached = _load_cached_arm(out_path)
                if cached is not None and (limit is None or cached.total == limit):
                    all_err = bool(cached.results) and all(
                        any(
                            "connection" in d.lower() or d.startswith("ERROR:")
                            for d in r.details
                        )
                        for r in cached.results
                    )
                    if not all_err:
                        reports[arm.name] = cached
                        continue
            unreachable = _arm_reachable(arm)
            if unreachable:
                raise RuntimeError(
                    f"Arm {arm.name!r} unreachable at {arm.base_url}: {unreachable}. "
                    "Start sonec serve-llm (LoRA) or ollama before leaderboard."
                )
            reports[arm.name] = await run_arm(
                arm=arm,
                suite=suite,
                workspace_root=ws_root,
                out_path=out_path,
                limit=limit,
            )
            # Checkpoint ranking after each arm so an abort still leaves a usable board.
            partial = rank_arms(reports, [a for a in arms if a.name in reports])
            write_leaderboard(
                out_dir=out_dir,
                suite=suite,
                summary=partial,
                reports=reports,
                catalog=catalog,
            )

    asyncio.run(_all())
    summary = rank_arms(reports, arms)
    write_leaderboard(
        out_dir=out_dir, suite=suite, summary=summary, reports=reports, catalog=catalog
    )
    return summary
