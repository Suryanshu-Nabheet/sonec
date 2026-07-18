"""Compare summary unit tests (no live LLM)."""

from __future__ import annotations

from pathlib import Path

from sonec.eval.compare import ArmSpec, summarize, write_compare_report
from sonec.eval.harness import BenchmarkReport, EvalResult


def _report(name: str, passed: list[bool]) -> BenchmarkReport:
    results = [
        EvalResult(task_id=f"t{i}", passed=p, score=1.0 if p else 0.0)
        for i, p in enumerate(passed)
    ]
    total = len(results)
    n_pass = sum(1 for r in results if r.passed)
    return BenchmarkReport(
        name=name,
        results=results,
        pass_rate=n_pass / total,
        mean_duration_s=1.0,
        mean_score=n_pass / total,
        passed=n_pass,
        total=total,
    )


def test_summarize_lora_wins(tmp_path: Path) -> None:
    arms = [
        ArmSpec("sonec_lora", "http://127.0.0.1:8080/v1", "m", "lora"),
        ArmSpec("qwen35_2b_base", "http://127.0.0.1:8081/v1", "m", "base"),
    ]
    reports = {
        "sonec_lora": _report("lora", [True, True, True, False]),
        "qwen35_2b_base": _report("base", [True, False, False, False]),
    }
    summary = summarize(suite=Path("suite.json"), arms=arms, reports=reports)
    assert summary.winner == "sonec_lora"
    assert summary.delta_pass_rate == 0.5
    path = write_compare_report(out_dir=tmp_path, summary=summary, reports=reports)
    assert path.exists()
    assert (tmp_path / "COMPARE_REPORT.md").exists()
