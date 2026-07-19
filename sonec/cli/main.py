"""SONEC command-line interface."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from sonec import __version__
from sonec.analysis.architecture import ArchitectureAnalyzer, report_to_mermaid
from sonec.analysis.debug import parse_traceback, suggest_debug_plan
from sonec.analysis.refactor import RefactorAnalyzer
from sonec.analysis.review import CodeReviewer, findings_to_markdown
from sonec.app import build_runtime
from sonec.core.config import load_settings
from sonec.core.types import AgentEvent, AgentEventKind
from sonec.docsgen.generator import DocGenerator
from sonec.eval.harness import EvalHarness
from sonec.indexing.index import RepositoryIndex
from sonec.llm.provider import MockProvider
from sonec.training.pipeline import DatasetGenerator, TrainingPipeline

app = typer.Typer(
    name="sonec",
    help="sonec by Suryanshu Nabheet — coding model",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _print_event(event: AgentEvent) -> None:
    phase = event.payload.get("phase")
    phase_prefix = f"[dim]{phase}[/] " if phase else ""
    if event.kind == AgentEventKind.TOOL_CALL:
        console.print(
            f"{phase_prefix}[cyan]→ tool[/] {event.message} {event.payload.get('arguments', {})}"
        )
    elif event.kind == AgentEventKind.TOOL_RESULT:
        ok = event.payload.get("ok", True)
        color = "green" if ok else "red"
        console.print(f"{phase_prefix}[{color}]← result[/] {event.message[:200]}")
    elif event.kind == AgentEventKind.MESSAGE and event.message:
        title = f"assistant:{phase}" if phase else "assistant"
        console.print(Panel(event.message, title=title, border_style="blue"))
    elif event.kind == AgentEventKind.PLAN:
        console.print("[magenta]plan ready[/]")
    elif event.kind == AgentEventKind.STEP and event.payload.get("phase"):
        console.print(f"[yellow]▸ phase[/] {event.payload.get('phase')} — {event.message}")
    elif event.kind in {AgentEventKind.FAILED, AgentEventKind.WARNING}:
        console.print(f"[red]{event.kind.value}[/]: {event.message}")


@app.callback()
def main_callback() -> None:
    """SONEC CLI root."""


@app.command("version")
def version_cmd() -> None:
    """Print the sonec version."""
    console.print(__version__)


@app.command("run")
def run_cmd(
    goal: str = typer.Argument(..., help="Engineering goal for the agent"),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w", help="Workspace root"),
    provider: str | None = typer.Option(
        None, "--provider", help="local|openai|openai_compatible|mock"
    ),
    model: str | None = typer.Option(None, "--model", "-m"),
    max_iterations: int | None = typer.Option(None, "--max-iterations"),
    mock: bool = typer.Option(False, "--mock", help="Use scripted mock provider (offline)"),
    phases: bool = typer.Option(False, "--phases", help="Inject optional phase guidance hints"),
) -> None:
    """Run the frozen production runtime (same loop as eval / training / IDE gateway)."""

    async def _run() -> None:
        from sonec.app import build_runtime
        from sonec.cli.providers import provider_overrides
        from sonec.harness.versioning import HARNESS_VERSION

        overrides: dict[str, object] = {"workspace": workspace}
        overrides.update(provider_overrides(mock=mock, provider=provider, model=model))
        if max_iterations:
            overrides["max_iterations"] = max_iterations
        settings = load_settings(**overrides)
        llm = MockProvider.harness_smoke(goal) if (mock or settings.provider == "mock") else None

        runtime, cfg, ws, _registry = build_runtime(
            settings=settings,
            provider=llm,
            persist_memory=True,
            enable_phase_hints=phases,
            goal_for_prompt=goal,
        )
        runtime.events.subscribe(_print_event)
        console.print(
            Panel.fit(
                f"[bold]{goal}[/]\nworkspace: {ws.root}\n"
                f"harness={HARNESS_VERSION} tool_hash={runtime.tool_hash}\n"
                f"provider: {cfg.provider} / {cfg.model}",
                title="sonec",
            )
        )
        result = await runtime.run(goal)
        console.print()
        title = "completed" if result.completed else "stopped"
        # Ungraded CLI runs: completed ≠ environment success
        console.print(
            Panel.fit(
                f"{result.final_message}\n\n"
                f"completed={result.completed} success={result.success} "
                f"iters={result.iterations}",
                title=title,
                border_style="green" if result.completed else "yellow",
            )
        )

    asyncio.run(_run())


@app.command("skills")
def skills_cmd() -> None:
    """List packaged SONEC skills."""
    from sonec.skills.registry import SkillsRegistry

    table = Table("ID", "Name", "Always", "Description")
    for item in SkillsRegistry().catalog():
        table.add_row(
            str(item["id"]),
            str(item["name"]),
            "yes" if item["always"] else "",
            str(item["description"])[:80],
        )
    console.print(table)


@app.command("rules")
def rules_cmd() -> None:
    """List operating rules (including prebuilt rules)."""
    from sonec.rules.engine import RulesEngine

    table = Table("ID", "Always", "Tags", "Chars")
    for item in RulesEngine().list_rules():
        table.add_row(
            str(item["id"]),
            "yes" if item["always"] else "",
            ",".join(item["tags"]),  # type: ignore[arg-type]
            str(item["chars"]),
        )
    console.print(table)


@app.command("index")
def index_cmd(
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
    out: Path | None = typer.Option(None, "--out", help="Write JSON summary"),
) -> None:
    """Build a repository index."""
    from sonec.core.workspace import Workspace

    ws = Workspace(workspace)
    index = RepositoryIndex(ws)
    count = index.build()
    summary = index.summary()
    console.print(json.dumps({"indexed": count, **summary}, indent=2))
    if out:
        index.dump(out)
        console.print(f"Wrote {out}")


@app.command("review")
def review_cmd(
    path: str = typer.Argument(".", help="File or directory to review"),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
) -> None:
    """Run deterministic code review heuristics."""
    from sonec.core.workspace import Workspace

    reviewer = CodeReviewer(Workspace(workspace))
    findings = reviewer.review_path(path)
    console.print(Markdown(findings_to_markdown(findings)))


@app.command("refactor")
def refactor_cmd(
    path: str = typer.Argument(".", help="Path to analyze"),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
) -> None:
    """Suggest refactoring opportunities."""
    from sonec.core.workspace import Workspace

    analyzer = RefactorAnalyzer(Workspace(workspace))
    opportunities = analyzer.analyze(path)
    table = Table("Score", "Kind", "Location", "Message")
    for item in opportunities[:50]:
        table.add_row(f"{item.score:.1f}", item.kind, f"{item.path}:{item.line}", item.message)
    console.print(table if opportunities else "No opportunities found.")


@app.command("architecture")
def architecture_cmd(
    path: str = typer.Argument("src", help="Package root to analyze"),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
    mermaid: bool = typer.Option(False, "--mermaid"),
) -> None:
    """Analyze module dependencies."""
    from sonec.core.workspace import Workspace

    ws = Workspace(workspace)
    target = path if (ws.root / path).exists() else "."
    report = ArchitectureAnalyzer(ws).analyze(target)
    if mermaid:
        console.print(report_to_mermaid(report))
        return
    console.print(
        json.dumps(
            {
                "modules": len(report.modules),
                "edges": len(report.edges),
                "cycles": report.cycles,
                "fan_in_top": sorted(report.fan_in.items(), key=lambda x: x[1], reverse=True)[:10],
                "fan_out_top": sorted(report.fan_out.items(), key=lambda x: x[1], reverse=True)[:10],
            },
            indent=2,
        )
    )


@app.command("debug-trace")
def debug_trace_cmd(
    file: Path = typer.Argument(..., exists=True, readable=True, help="Traceback text file"),
) -> None:
    """Parse a Python traceback and suggest a debug plan."""
    tb = parse_traceback(file.read_text(encoding="utf-8", errors="replace"))
    console.print(f"[bold]{tb.error_type}[/]: {tb.error_message}")
    for frame in tb.frames:
        console.print(f"  {frame.path}:{frame.line} in {frame.function}")
    console.print("\n[bold]Suggested plan[/]")
    for step in suggest_debug_plan(tb):
        console.print(f"- {step}")


@app.command("docs")
def docs_cmd(
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
    out: str = typer.Option("docs/GENERATED.md", "--out"),
) -> None:
    """Generate repository documentation overview."""
    from sonec.core.workspace import Workspace

    path = DocGenerator(Workspace(workspace)).write(out)
    console.print(f"Wrote {path}")


@app.command("eval")
def eval_cmd(
    tasks: Path = typer.Argument(..., exists=True, help="JSON task suite"),
    workspace: Path = typer.Option(Path("."), "--workspace", "-w"),
    mock: bool = typer.Option(True, "--mock/--live", help="Task-aware mock solver by default"),
    out: Path | None = typer.Option(None, "--out", help="Write BenchmarkReport JSON"),
) -> None:
    """Run an evaluation suite with deterministic grading."""

    async def _eval() -> None:
        from sonec.eval.harness import mock_provider_for_task

        ws = workspace.expanduser().resolve()
        ws.mkdir(parents=True, exist_ok=True)
        settings = load_settings(workspace=ws, provider="mock" if mock else "local")
        harness = EvalHarness(workspace=settings.workspace)
        task_list = EvalHarness.load_tasks(tasks)

        def factory(task):
            llm = mock_provider_for_task(task) if mock else None
            runtime, *_ = build_runtime(
                settings=settings,
                provider=llm,
                persist_memory=False,
                log_dir=ws / ".trajectories",
                goal_for_prompt=task.prompt,
            )
            return runtime

        report = await harness.run_suite(task_list, factory, name=tasks.stem)
        if out:
            report.save(out)
            console.print(f"Wrote {out}")
        console.print(
            json.dumps(
                {
                    "name": report.name,
                    "pass_rate": report.pass_rate,
                    "mean_score": report.mean_score,
                    "passed": report.passed,
                    "total": report.total,
                    "mean_duration_s": report.mean_duration_s,
                    "by_difficulty": report.by_difficulty,
                    "by_tag": report.by_tag,
                    "results": [
                        {
                            "task_id": r.task_id,
                            "passed": r.passed,
                            "score": r.score,
                            "difficulty": r.difficulty,
                            "details": r.details,
                        }
                        for r in report.results
                    ],
                },
                indent=2,
            )
        )

    asyncio.run(_eval())


@app.command("bench")
def bench_cmd(
    suite: Path = typer.Option(
        Path("examples/benchmarks/smoke.json"),
        "--suite",
        "-s",
        help="Benchmark suite JSON",
        exists=True,
    ),
    workspace: Path = typer.Option(Path(".sonec/bench-workspace"), "--workspace", "-w"),
    mock: bool = typer.Option(True, "--mock/--live"),
    provider: str = typer.Option("local", "--provider"),
    model: str | None = typer.Option(None, "--model", "-m"),
    out: Path = typer.Option(Path("artifacts/benchmarks/latest.json"), "--out"),
) -> None:
    """Run a benchmark suite with environment-evidence grading."""

    async def _bench() -> None:
        from sonec.app import build_runtime
        from sonec.cli.providers import provider_overrides
        from sonec.eval.harness import mock_provider_for_task

        ws = workspace.expanduser().resolve()
        if ws.exists():
            import shutil

            shutil.rmtree(ws)
        ws.mkdir(parents=True, exist_ok=True)
        overrides = {
            "workspace": ws,
            **provider_overrides(mock=mock, live=not mock, provider=provider, model=model),
        }
        settings = load_settings(**overrides)
        harness = EvalHarness(workspace=ws)
        data = json.loads(suite.read_text(encoding="utf-8"))
        name = data.get("name", suite.stem) if isinstance(data, dict) else suite.stem
        task_list = EvalHarness.load_tasks(suite)

        def factory(task):
            llm = mock_provider_for_task(task) if mock else None
            runtime, *_ = build_runtime(
                settings=settings,
                provider=llm,
                persist_memory=False,
                log_dir=ws / ".trajectories",
                goal_for_prompt=task.prompt,
            )
            return runtime

        report = await harness.run_suite(task_list, factory, name=str(name))
        report.save(out)
        console.print(
            Panel.fit(
                f"[bold]{report.name}[/]\n"
                f"pass_rate={report.pass_rate:.0%} ({report.passed}/{report.total})\n"
                f"mean_score={report.mean_score:.2f}\n"
                f"report={out}",
                title="sonec bench",
                border_style="green" if report.pass_rate == 1.0 else "yellow",
            )
        )
        if report.pass_rate < 1.0 and mock:
            raise typer.Exit(code=1)

    asyncio.run(_bench())


@app.command("sonecbench")
def sonecbench_cmd(
    out: Path = typer.Option(Path("examples/benchmarks/sonecbench_v1.json"), "--out"),
    run: bool = typer.Option(False, "--run", help="Also run mock graded pass on the suite"),
    limit: int = typer.Option(0, "--limit", help="Optional task limit when --run"),
) -> None:
    """Generate (and optionally run) SonecBench v1 — the private decision metric."""
    from sonec.eval.sonecbench import build_sonecbench_tasks, write_sonecbench

    path = write_sonecbench(out)
    tasks = build_sonecbench_tasks()
    console.print(f"Wrote {path} ({len(tasks)} tasks, sealed=true)")
    if run:
        subset = tasks[: limit or len(tasks)]

        async def _run() -> None:
            from sonec.app import build_runtime
            from sonec.eval.harness import mock_provider_for_task

            ws = Path(".sonec/sonecbench-ws").resolve()
            if ws.exists():
                import shutil

                shutil.rmtree(ws)
            ws.mkdir(parents=True)
            settings = load_settings(workspace=ws, provider="mock")
            harness = EvalHarness(workspace=ws)

            def factory(task):
                runtime, *_ = build_runtime(
                    settings=settings,
                    provider=mock_provider_for_task(task),
                    persist_memory=False,
                    log_dir=ws / ".trajectories",
                    goal_for_prompt=task.prompt,
                )
                return runtime

            report = await harness.run_suite(subset, factory, name="sonecbench-v1-mock")
            report_path = Path("artifacts/benchmarks/sonecbench_mock.json")
            report.save(report_path)
            console.print(
                f"mock pass_rate={report.pass_rate:.0%} ({report.passed}/{report.total}) → {report_path}"
            )
            if report.pass_rate < 1.0:
                raise typer.Exit(code=1)

        asyncio.run(_run())


@app.command("rollout")
def rollout_cmd(
    suite: Path = typer.Option(
        Path("examples/benchmarks/smoke.json"), "--suite", "-s", exists=True
    ),
    out: Path = typer.Option(Path("artifacts/rollouts"), "--out"),
    group_size: int = typer.Option(8, "--group-size", "-g", help="G rollouts per task"),
    limit: int = typer.Option(40, "--limit", help="Max tasks (0=all)"),
    mock: bool = typer.Option(
        False, "--mock/--live", help="Live graded fuel by default (use --mock offline)"
    ),
    provider: str = typer.Option("local", "--provider"),
    model: str | None = typer.Option(None, "--model", "-m"),
) -> None:
    """Graded rollout factory — live winners for SFT/RL (mock only for offline tests)."""
    from sonec.training.rollouts import run_rollouts_sync

    tasks = EvalHarness.load_tasks(suite)
    if limit:
        tasks = tasks[:limit]
    records = run_rollouts_sync(
        tasks,
        out,
        group_size=group_size,
        use_mock=mock,
        provider_name=provider,
        model=model,
    )
    passed = sum(1 for r in records if r.passed)
    console.print(
        Panel.fit(
            f"records={len(records)} passed={passed} "
            f"group_size={group_size} mock={mock}\njsonl={out / 'rollouts.jsonl'}",
            title="sonec rollout",
            border_style="green",
        )
    )


@app.command("dataset")
def dataset_cmd(
    out: Path = typer.Option(Path("artifacts/dataset"), "--out"),
) -> None:
    """Generate a synthetic software-engineering training dataset shard."""
    gen = DatasetGenerator()
    gen.synthesize_smoke_examples()
    manifest = gen.manifest()
    pipeline = TrainingPipeline(out)
    jsonl = pipeline.export_jsonl(manifest)
    config = pipeline.write_config(model="sonec")
    manifest.save(out / "manifest.json")
    console.print(f"Wrote {jsonl}")
    console.print(f"Wrote {config}")
    console.print(f"Examples: {len(manifest.examples)}")


@app.command("worldbench")
def worldbench_cmd(
    out: Path = typer.Option(Path("examples/benchmarks/worldbench_v1.json"), "--out"),
    run: bool = typer.Option(False, "--run"),
    limit: int = typer.Option(0, "--limit"),
    mock: bool = typer.Option(True, "--mock/--live"),
    provider: str = typer.Option("local", "--provider"),
    model: str | None = typer.Option(None, "--model", "-m"),
) -> None:
    """Generate / run WorldBench — VS Code / Bun / Codex-shaped real-world tasks."""
    from sonec.eval.worldbench import build_worldbench_tasks, write_worldbench

    path = write_worldbench(out)
    tasks = build_worldbench_tasks()
    console.print(f"Wrote {path} ({len(tasks)} tasks, sealed=true)")
    if not run:
        return
    subset = tasks[: limit or len(tasks)]

    async def _run() -> None:
        from sonec.app import build_runtime
        from sonec.cli.providers import provider_overrides
        from sonec.eval.harness import mock_provider_for_task

        ws = Path(".sonec/worldbench-ws").resolve()
        if ws.exists():
            import shutil

            shutil.rmtree(ws)
        ws.mkdir(parents=True)
        overrides = {"workspace": ws, **provider_overrides(mock=mock, live=not mock, provider=provider, model=model)}
        settings = load_settings(**overrides)
        harness = EvalHarness(workspace=ws)

        def factory(task):
            llm = mock_provider_for_task(task) if mock else None
            runtime, *_ = build_runtime(
                settings=settings,
                provider=llm,
                persist_memory=False,
                log_dir=ws / ".trajectories",
                goal_for_prompt=task.prompt,
            )
            return runtime

        report = await harness.run_suite(subset, factory, name="worldbench-v1")
        report_path = Path("artifacts/benchmarks/worldbench_latest.json")
        report.save(report_path)
        console.print(
            f"pass_rate={report.pass_rate:.0%} ({report.passed}/{report.total}) → {report_path}"
        )
        if report.pass_rate < 1.0 and mock:
            raise typer.Exit(code=1)

    asyncio.run(_run())


@app.command("corpora")
def corpora_cmd(
    sync: bool = typer.Option(False, "--sync", help="Shallow-clone OSS corpora"),
    config: Path = typer.Option(Path("examples/corpora.yaml"), "--config"),
    root: Path = typer.Option(Path("corpora"), "--root"),
    include_optional: bool = typer.Option(False, "--optional", help="Include huge repos (bun)"),
    only: str = typer.Option("", "--only", help="Comma-separated repo ids"),
) -> None:
    """Manage open-source corpora for hard live agent workspaces."""
    from sonec.eval.corpora import default_manifest, load_manifest, sync_all, write_default_yaml

    if not config.exists():
        write_default_yaml(config)
        console.print(f"Wrote {config}")
    manifest = load_manifest(config, root=root) if config.exists() else default_manifest(root)
    if not sync:
        for repo in manifest.repos:
            flag = "optional" if repo.optional else "default"
            console.print(f"- {repo.id} [{flag}] {repo.url}")
        return
    only_ids = [x.strip() for x in only.split(",") if x.strip()] or None
    results = sync_all(manifest, include_optional=include_optional, only=only_ids)
    for row in results:
        console.print(f"{row['id']}: {row['status']} → {row.get('path')}")
        if row.get("error"):
            console.print(f"  error: {row['error']}")


@app.command("compare")
def compare_cmd(
    suite: Path = typer.Option(
        Path("examples/benchmarks/capabilitybench_v1.json"),
        "--suite",
        "-s",
        exists=True,
    ),
    lora_url: str = typer.Option(
        "http://127.0.0.1:8080/v1",
        "--lora-url",
        help="OpenAI-compatible URL for specialized sonec (serve-llm)",
    ),
    base_url: str = typer.Option(
        "http://127.0.0.1:8081/v1",
        "--base-url",
        help="OpenAI-compatible URL for base Qwen 3.5 2B (no adapter)",
    ),
    lora_model: str = typer.Option(
        "mlx-community/Qwen3.5-2B-4bit",
        "--lora-model",
        help="Model id as advertised by the LoRA server",
    ),
    base_model: str = typer.Option(
        "mlx-community/Qwen3.5-2B-4bit",
        "--base-model",
        help="Model id as advertised by the base server",
    ),
    out: Path = typer.Option(Path("docs/results"), "--out", "-o"),
) -> None:
    """Live A/B: specialized sonec LoRA versus the unmodified base checkpoint."""
    from sonec.eval.compare import run_compare_sync
    from sonec.training.weights import weight_status

    status = weight_status()
    if not status.ready:
        console.print(f"[red]{status.detail}[/]")
        console.print("Specialize first: sonec train --step && sonec serve-llm")
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"suite={suite}\nlora={lora_url} ({lora_model})\n"
            f"base={base_url} ({base_model})\n"
            "Same harness; only weights/endpoint differ.",
            title="sonec compare",
            border_style="cyan",
        )
    )
    summary = run_compare_sync(
        suite=suite,
        lora_url=lora_url,
        base_url=base_url,
        lora_model=lora_model,
        base_model=base_model,
        out_dir=out,
    )
    for arm in summary.arms:
        color = "green" if arm["kind"] == "lora" else "white"
        console.print(
            f"[{color}]{arm['name']}[/]: pass_rate={arm['pass_rate']:.0%} "
            f"({arm['passed']}/{arm['total']}) mean_score={arm['mean_score']:.2f}"
        )
    console.print(f"winner={summary.winner or 'tie'} delta={summary.delta_pass_rate:+.0%}")
    console.print(f"Wrote {out / 'COMPARE_REPORT.json'} and COMPARE_REPORT.md")


@app.command("leaderboard")
def leaderboard_cmd(
    suite: Path = typer.Option(
        Path("examples/benchmarks/capabilitybench_v1.json"),
        "--suite",
        "-s",
        exists=True,
        help="Sealed CapabilityBench (200) — or ab_agent_2b_hard for a short smoke",
    ),
    arms: Path = typer.Option(
        Path("configs/leaderboard/arms_2b.json"),
        "--arms",
        "-a",
        exists=True,
        help="Strict 2B-only OpenAI-compatible arms (no 1B/1.5B/3B+)",
    ),
    out: Path = typer.Option(Path("docs/results/leaderboard_2b"), "--out", "-o"),
    resume: bool = typer.Option(
        True,
        "--resume/--fresh",
        help="Reuse existing arm_*.json dumps (default); --fresh re-runs every arm",
    ),
    limit: int = typer.Option(
        0,
        "--limit",
        help="Optional max tasks (0=all). Useful for stratified probes before full 200.",
    ),
) -> None:
    """Multi-model agent leaderboard (2B-class rivals + specialized sonec)."""
    from sonec.eval.leaderboard import load_arms, run_leaderboard_sync

    data = json.loads(arms.read_text(encoding="utf-8"))
    catalog = data.get("catalog") if isinstance(data, dict) else None
    arm_list = load_arms(arms)
    console.print(
        Panel.fit(
            f"suite={suite}\narms={len(arm_list)} from {arms}\nout={out}\n"
            f"resume={resume} limit={limit or 'all'}",
            title="sonec leaderboard",
            border_style="cyan",
        )
    )
    summary = run_leaderboard_sync(
        suite=suite,
        arms=arm_list,
        out_dir=out,
        catalog={"entries": catalog} if catalog else None,
        resume=resume,
        limit=limit or None,
    )
    for i, arm in enumerate(summary.arms, start=1):
        color = "green" if arm["kind"] == "lora" else "white"
        console.print(
            f"[{color}]#{i} {arm['name']}[/]: {arm['pass_rate']:.0%} "
            f"({arm['passed']}/{arm['total']}) {arm['mean_duration_s']:.1f}s"
        )
    console.print(f"winner={summary.winner}")
    console.print(f"Wrote {out / 'LEADERBOARD.md'} + LEADERBOARD_CHART.html")


@app.command("grpo")
def grpo_cmd(
    group_size: int = typer.Option(
        2,
        "--group-size",
        "-g",
        help="Rollouts per prompt (keep ≤4 for --live; default 2 is laptop-safe)",
    ),
    train_n: int = typer.Option(
        8,
        "--train-n",
        help="TrainBench prompts (keep ≤16 for --live; default 8 is laptop-safe)",
    ),
    sft_iters: int = typer.Option(80, "--sft-iters", help="LoRA iters after densify"),
    live: bool = typer.Option(
        False,
        "--live/--mock",
        help="Live agent rollouts (heavy) vs oracle mock densify (default, safe)",
    ),
    mlx_model: str = typer.Option("mlx-community/Qwen3.5-2B-4bit", "--mlx-model"),
) -> None:
    """Light GRPO-lite densify. Default is mock + small G/n so it won't thrash the machine."""
    from sonec.training.grpo_lite import run_grpo_lite
    from sonec.training.weights import weight_status

    status = weight_status()
    if not status.ready:
        console.print(f"[red]{status.detail}[/]")
        console.print("Run sonec train --step first.")
        raise typer.Exit(code=1)
    if live and (group_size > 4 or train_n > 16):
        console.print(
            "[red]Refusing heavy live GRPO[/] — use --group-size≤4 and --train-n≤16, "
            "or --mock. Large live GRPO OOMs Apple Silicon hosts."
        )
        raise typer.Exit(code=2)
    mode = "LIVE (heavy)" if live else "mock (safe)"
    console.print(
        Panel.fit(
            f"GRPO-lite {mode}\nG={group_size} train_n={train_n} sft_iters={sft_iters}\n"
            f"Expected rollouts ≈ {group_size * train_n}",
            title="sonec grpo",
            border_style="magenta",
        )
    )
    result = run_grpo_lite(
        root=Path.cwd(),
        group_size=group_size,
        train_n=train_n,
        sft_iters=sft_iters,
        live=live,
        mlx_model=mlx_model,
    )
    color = "green" if result.sft.ok else "red"
    console.print(
        f"prompts={result.prompts} rollouts={result.rollouts} "
        f"pos_adv={result.positive_advantage} passers={result.absolute_passers} "
        f"corpus={result.corpus_lines}"
    )
    console.print(f"[{color}]sft[/]: {result.sft.detail}")
    console.print(f"stats: {result.stats_path}")


@app.command("capabilitybench")
def capabilitybench_cmd(
    out: Path = typer.Option(
        Path("examples/benchmarks/capabilitybench_v1.json"),
        "--out",
        "-o",
    ),
    write_only: bool = typer.Option(
        True,
        "--write-only/--no-write-only",
        help="Only generate the sealed JSON (default)",
    ),
) -> None:
    """Generate sealed 200-task CapabilityBench (10×20 categories, easy/medium/hard)."""
    from sonec.eval.capabilitybench import (
        CATEGORIES,
        build_capabilitybench_tasks,
        write_capabilitybench,
    )

    path = write_capabilitybench(out)
    tasks = build_capabilitybench_tasks()
    by_diff: dict[str, int] = {}
    for t in tasks:
        by_diff[t.difficulty] = by_diff.get(t.difficulty, 0) + 1
    console.print(
        Panel.fit(
            f"wrote {path}\ntasks={len(tasks)}\n"
            f"categories={len(CATEGORIES)}\n"
            f"difficulty={by_diff}",
            title="capabilitybench-v1",
            border_style="cyan",
        )
    )
    if write_only:
        return
    console.print("[yellow]Live run not requested — use sonec leaderboard -s "
                  "examples/benchmarks/capabilitybench_v1.json[/]")

@app.command("train")
def train_cmd(
    export: bool = typer.Option(False, "--export", help="Export trainer shards from rollouts"),
    step: bool = typer.Option(False, "--step", help="One small specialize step (recommended)"),
    full: bool = typer.Option(False, "--full", help="Alias of --step (kept for scripts)"),
    rollouts: Path = typer.Option(Path("artifacts/rollouts/rollouts.jsonl"), "--rollouts", "-r"),
    out: Path = typer.Option(Path("artifacts/train"), "--out", "-o"),
    exclude_sealed: bool = typer.Option(True, "--exclude-sealed/--include-all"),
    sft_iters: int = typer.Option(300, "--sft-iters", help="LoRA iters on clean live data"),
    gold_n: int = typer.Option(0, "--gold-n", help="Optional gold seed (0 = live only)"),
    train_n: int = typer.Option(40, "--train-n", help="TrainBench tasks for live fuel"),
    skip_sft: bool = typer.Option(False, "--skip-sft"),
    skip_fuel: bool = typer.Option(False, "--skip-fuel", help="Reuse existing fuel rollouts"),
    corpus: Path | None = typer.Option(
        None, "--corpus", help="Reuse existing mlx_data dir (skip fuel+assemble)"
    ),
    live_fuel: bool = typer.Option(
        True,
        "--live-fuel/--mock-fuel",
        help="Live model fuel, or oracle graded trajectories (structured tool_calls)",
    ),
    live_rl: bool = typer.Option(True, "--live-rl/--mock-rl", help="Live or oracle rejection RL"),
    reset: bool = typer.Option(False, "--reset", help="Wipe artifacts/train before step"),
    mlx_model: str = typer.Option(
        "mlx-community/Qwen3.5-2B-4bit",
        "--mlx-model",
        help="MLX base checkpoint for LoRA",
    ),
    rollout_group: int = typer.Option(8, "--rollout-group", help="Rollouts per TrainBench task"),
    rl_group: int = typer.Option(4, "--rl-group", help="Rejection sampling group size"),
    rl_limit: int = typer.Option(24, "--rl-limit", help="TrainBench tasks for rejection round"),
) -> None:
    """Specialize sonec via graded trajectories and LoRA."""
    if step or full:
        from sonec.training.specialize import run_train_step
        from sonec.training.weights import weight_status

        console.print(
            Panel.fit(
                f"Specialize — live_fuel={live_fuel} SFT={sft_iters} "
                f"gold={gold_n} train_n={train_n} group={rollout_group}\n"
                f"rl_group={rl_group} rl_limit={rl_limit}\n"
                f"mlx={mlx_model}\n"
                "Retain the adapter only when sonec compare pass rate improves.",
                title="sonec train",
                border_style="cyan",
            )
        )
        reports = run_train_step(
            root=Path.cwd(),
            sft_iters=sft_iters,
            gold_n=gold_n,
            train_n=train_n,
            skip_sft=skip_sft,
            skip_fuel=skip_fuel,
            live_fuel=live_fuel,
            live_rl=live_rl,
            mlx_model=mlx_model,
            reset=reset,
            corpus_dir=corpus,
            rollout_group=rollout_group,
            rl_group=rl_group,
            rl_limit=rl_limit,
        )
        for r in reports:
            color = "green" if r.ok else "red"
            console.print(f"[{color}]{r.phase}[/]: {r.detail}")
        status = weight_status()
        console.print("Report: artifacts/train/TRAIN_REPORT.json")
        console.print(f"Weights: {'ready' if status.ready else 'not ready'} — {status.detail}")
        if status.ready:
            console.print("Serve specialized model: sonec serve-llm")
        if any(not r.ok for r in reports) or not status.ready:
            raise typer.Exit(code=1)
        return

    from sonec.training.export import export_from_rollouts

    if not export:
        console.print("sonec train --step                 # LoRA SFT + RL (writes *.safetensors)")
        console.print("sonec train --step --corpus …      # reuse mlx_data, skip fuel")
        console.print("sonec train --export -r …          # export shards only")
        raise typer.Exit(code=0)
    sealed: set[str] = set()
    if exclude_sealed:
        for suite in (
            Path("examples/benchmarks/sonecbench_v1.json"),
            Path("examples/benchmarks/worldbench_v1.json"),
            Path("examples/benchmarks/ab_agent_v1.json"),
            Path("examples/benchmarks/ab_agent_2b_hard.json"),
            Path("examples/benchmarks/capabilitybench_v1.json"),
        ):
            if suite.exists():
                data = json.loads(suite.read_text(encoding="utf-8"))
                for t in data.get("tasks") or []:
                    sealed.add(t["id"])
    written = export_from_rollouts(rollouts, out, sealed_ids=sealed)
    for name, path in written.items():
        console.print(f"{name}: {path}")
    console.print(f"manifest: {out / 'manifest.json'}")


@app.command("weights")
def weights_cmd() -> None:
    """Show whether product LoRA weights exist (not a Modelfile)."""
    from sonec.training.weights import weight_status

    status = weight_status()
    table = Table("Field", "Value")
    for k, v in status.to_dict().items():
        table.add_row(str(k), str(v))
    console.print(table)
    if not status.ready:
        console.print("[red]Adapter weights missing. Run sonec train --step, then sonec serve-llm.[/]")
        raise typer.Exit(code=1)


@app.command("serve-llm")
def serve_llm_cmd(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8080, "--port"),
    adapter: Path = typer.Option(
        Path("artifacts/train/checkpoints/sonec-sft-mlx"), "--adapter"
    ),
    model: str = typer.Option("mlx-community/Qwen3.5-2B-4bit", "--model", "-m"),
) -> None:
    """Serve specialized sonec = base Qwen 3.5 2B + trained LoRA adapter (OpenAI-compatible)."""
    import subprocess

    from sonec.training.weights import mlx_server_command, weight_status

    status = weight_status(adapter)
    if not status.ready:
        console.print(f"[red]{status.detail}[/]")
        console.print("Run: sonec train --step")
        raise typer.Exit(code=1)
    cmd = mlx_server_command(adapter_dir=adapter, model=model, host=host, port=port)
    console.print(
        Panel.fit(
            f"Serving specialized sonec\nbase={model}\nadapter={adapter}\n"
            f"OpenAI API: http://{host}:{port}/v1\n"
            f"Set SONEC_BASE_URL=http://{host}:{port}/v1",
            title="sonec serve-llm",
            border_style="green",
        )
    )
    raise SystemExit(subprocess.call(cmd))


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8787, "--port"),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
    provider: str = typer.Option("local", "--provider"),
    model: str = typer.Option("sonec", "--model", "-m"),
) -> None:
    """IDE/CLI agent gateway (harness). Point SONEC_BASE_URL at serve-llm for specialized weights."""
    from sonec.serve import serve_blocking

    serve_blocking(host=host, port=port, workspace=workspace, provider=provider, model=model)


@app.command("mcp")
def mcp_cmd() -> None:
    """Run MCP stdio server for Cursor / VS Code / Claude Desktop."""
    from sonec.ide.mcp_server import main as mcp_main

    mcp_main()


@app.command("doctor")
def doctor_cmd() -> None:
    """Check specialized weights + inference endpoint + harness."""
    import httpx

    from sonec.core.config import load_settings
    from sonec.harness.versioning import HARNESS_VERSION
    from sonec.models import BASE_HF, BASE_MODEL, PRODUCT_MODEL
    from sonec.training.weights import weight_status

    settings = load_settings()
    status = weight_status()
    rows: list[tuple[str, str]] = [
        ("sonec", __version__),
        ("harness", HARNESS_VERSION),
        ("product", PRODUCT_MODEL),
        ("base", BASE_MODEL),
        ("base_hf", BASE_HF),
        ("provider", settings._normalized_provider()),
        ("base_url", settings.resolved_base_url()),
        ("weights", "READY" if status.ready else "MISSING"),
        ("weights_detail", status.detail),
    ]
    base = settings.resolved_base_url()
    try:
        r = httpx.get(f"{base}/models", timeout=3.0)
        if r.status_code == 200:
            data = r.json()
            ids = [m.get("id", "") for m in data.get("data", [])] if isinstance(data, dict) else []
            rows.append(("inference", "ok"))
            rows.append(("models", ", ".join(ids[:8]) or "(empty list)"))
            if not status.ready:
                rows.append(
                    (
                        "warning",
                        "endpoint may be base+prompt only — no LoRA *.safetensors yet",
                    )
                )
        else:
            rows.append(("inference", f"http {r.status_code} at {base}/models"))
    except Exception as exc:  # noqa: BLE001
        rows.append(("inference", f"unreachable ({exc})"))
    for p in (
        Path("NOTICE"),
        Path("LICENSE"),
        Path("artifacts/train/PRODUCT.json"),
        Path("examples/benchmarks/worldbench_v1.json"),
        Path("configs/sft/mlx_lora.yaml"),
    ):
        rows.append((str(p), "ok" if p.exists() else "MISSING"))
    table = Table("Check", "Status")
    for k, v in rows:
        table.add_row(k, v)
    console.print(table)
    console.print(
        "\nProduction path:\n"
        "1) sonec train --step          # or ./scripts/overnight_specialize.sh\n"
        "2) sonec weights\n"
        "3) sonec serve-llm             # LoRA on :8080\n"
        "4) SONEC_BASE_URL=http://127.0.0.1:8080/v1 sonec run \"…\"\n"
        "5) sonec compare               # LoRA vs base A/B\n"
        "6) sonec grpo --mock           # light densify only (never heavy live on laptop)\n"
        "7) sonec capabilitybench       # sealed 200-task decision suite\n"
        "8) sonec leaderboard           # multi-model 2B board (CapabilityBench default)\n"
        "   SKIP_GRPO=1 ./scripts/world_rl_leaderboard.sh\n"
        "Product = LoRA *.safetensors (see sonec weights). Ollama Modelfile is a chat runner only."
    )
    if not status.ready:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
