#!/usr/bin/env python3
"""Daily CI stats — suite integrity + mock harness smoke (no live GPU required).

Writes docs/results/DAILY_STATUS.md and docs/results/DAILY_STATUS.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.strip()


def main() -> int:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    checks: list[dict] = []

    # 1) CapabilityBench shape
    from sonec.eval.capabilitybench import (
        CATEGORIES,
        build_capabilitybench_tasks,
        write_capabilitybench,
    )

    suite_path = ROOT / "examples/benchmarks/capabilitybench_v1.json"
    write_capabilitybench(suite_path)
    tasks = build_capabilitybench_tasks()
    by_diff = {d: 0 for d in ("easy", "medium", "hard")}
    for t in tasks:
        by_diff[t.difficulty] += 1
    ok_cap = len(tasks) == 200 and by_diff == {"easy": 70, "medium": 70, "hard": 60}
    checks.append(
        {
            "name": "capabilitybench_shape",
            "ok": ok_cap,
            "detail": f"tasks={len(tasks)} by_diff={by_diff} cats={len(CATEGORIES)}",
        }
    )

    # 2) Pytest
    code, out = _run([sys.executable, "-m", "pytest", "-q", "--tb=no"])
    checks.append(
        {
            "name": "pytest",
            "ok": code == 0,
            "detail": out.splitlines()[-1] if out else f"exit={code}",
        }
    )

    # 3) Mock bench (smoke suite if present, else capabilitybench limit via mock)
    smoke = ROOT / "examples/benchmarks/ab_agent_2b_hard.json"
    bench_suite = smoke if smoke.exists() else suite_path
    out_bench = ROOT / "artifacts/benchmarks/daily_mock.json"
    out_bench.parent.mkdir(parents=True, exist_ok=True)
    code, out = _run(
        [
            "sonec",
            "bench",
            "--mock",
            "--suite",
            str(bench_suite),
            "--out",
            str(out_bench),
        ]
    )
    mock_pass = None
    mock_total = None
    if out_bench.exists():
        try:
            data = json.loads(out_bench.read_text(encoding="utf-8"))
            mock_pass = data.get("passed")
            mock_total = data.get("total")
        except json.JSONDecodeError:
            pass
    checks.append(
        {
            "name": "mock_bench",
            "ok": code == 0,
            "detail": f"suite={bench_suite.name} passed={mock_pass}/{mock_total}",
        }
    )

    # 4) Published smoke snapshot (read-only; do not invent live scores)
    compare = ROOT / "docs/results/COMPARE_REPORT.json"
    board = ROOT / "docs/results/leaderboard_2b/LEADERBOARD.json"
    published: dict = {}
    if compare.exists():
        published["compare"] = json.loads(compare.read_text(encoding="utf-8")).get(
            "summary", {}
        )
    if board.exists():
        published["leaderboard"] = json.loads(board.read_text(encoding="utf-8")).get(
            "summary", {}
        )

    all_ok = all(c["ok"] for c in checks)
    payload = {
        "generated_at": started,
        "ok": all_ok,
        "checks": checks,
        "capabilitybench": {
            "task_count": len(tasks),
            "by_difficulty": by_diff,
            "categories": [{"id": c, "label": lab} for c, lab in CATEGORIES],
        },
        "published_smoke": {
            "note": "Live Cap200 scores require local Apple Silicon; CI records suite integrity + mock harness.",
            "compare_winner": (published.get("compare") or {}).get("winner"),
            "board_winner": (published.get("leaderboard") or {}).get("winner"),
            "compare_arms": (published.get("compare") or {}).get("arms"),
            "board_arms": (published.get("leaderboard") or {}).get("arms"),
        },
    }

    out_dir = ROOT / "docs/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "DAILY_STATUS.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Daily status",
        "",
        f"Generated: `{started}`",
        "",
        f"**Overall:** {'PASS' if all_ok else 'FAIL'}",
        "",
        "| Check | OK | Detail |",
        "| --- | --- | --- |",
    ]
    for c in checks:
        lines.append(f"| `{c['name']}` | {'yes' if c['ok'] else 'no'} | {c['detail']} |")
    lines.extend(
        [
            "",
            "## CapabilityBench",
            "",
            f"- Tasks: **{len(tasks)}** (sealed)",
            f"- Difficulty: easy={by_diff['easy']} medium={by_diff['medium']} hard={by_diff['hard']}",
            "",
            "## Published smoke snapshot (from repo files)",
            "",
            f"- Compare winner: `{payload['published_smoke'].get('compare_winner')}`",
            f"- Board winner: `{payload['published_smoke'].get('board_winner')}`",
            "",
            "Live Cap200 / MLX A/B is run locally (`SKIP_SFT=1 ./scripts/capabilitybench_e2e.sh`).",
            "This daily job validates the codebase and sealed suite on GitHub Actions.",
            "",
        ]
    )
    (out_dir / "DAILY_STATUS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"ok": all_ok, "generated_at": started}, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
