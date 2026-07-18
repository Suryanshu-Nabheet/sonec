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
    help="sonec — coding-specialist agentic harness",
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
    group_size: int = typer.Option(2, "--group-size", "-g", help="G rollouts per task"),
    limit: int = typer.Option(3, "--limit", help="Max tasks (0=all)"),
    mock: bool = typer.Option(True, "--mock/--live"),
    provider: str = typer.Option("local", "--provider"),
    model: str | None = typer.Option(None, "--model", "-m"),
) -> None:
    """Graded rollout factory — fuel for SFT/RL (live open-weight or mock)."""
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


@app.command("train")
def train_cmd(
    export: bool = typer.Option(False, "--export", help="Export trainer shards from rollouts"),
    step: bool = typer.Option(False, "--step", help="One small specialize step (recommended)"),
    full: bool = typer.Option(False, "--full", help="Alias of --step (kept for scripts)"),
    rollouts: Path = typer.Option(Path("artifacts/rollouts/rollouts.jsonl"), "--rollouts", "-r"),
    out: Path = typer.Option(Path("artifacts/train"), "--out", "-o"),
    exclude_sealed: bool = typer.Option(True, "--exclude-sealed/--include-all"),
    sft_iters: int = typer.Option(80, "--sft-iters", help="Small by default; raise over time"),
    gold_n: int = typer.Option(40, "--gold-n"),
    train_n: int = typer.Option(16, "--train-n", help="TrainBench tasks this step"),
    skip_sft: bool = typer.Option(False, "--skip-sft"),
    live_rl: bool = typer.Option(False, "--live-rl"),
    reset: bool = typer.Option(False, "--reset", help="Wipe artifacts/train before step"),
    mlx_model: str = typer.Option(
        "mlx-community/Qwen3.5-2B-4bit",
        "--mlx-model",
        help="HF/MLX base for LoRA (Qwen 3.5 2B lineage)",
    ),
) -> None:
    """Specialize sonec in small steps (SFT + rejection RL). Start small, iterate."""
    if step or full:
        from sonec.training.specialize import run_train_step

        console.print(
            Panel.fit(
                f"Specialize step — SFT iters={sft_iters} gold={gold_n} train_n={train_n}\n"
                f"mlx={mlx_model}",
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
            live_rl=live_rl,
            mlx_model=mlx_model,
            reset=reset,
        )
        for r in reports:
            color = "green" if r.ok else "red"
            console.print(f"[{color}]{r.phase}[/]: {r.detail}")
        console.print("Report: artifacts/train/TRAIN_REPORT.json")
        if any(not r.ok for r in reports):
            raise typer.Exit(code=1)
        return

    from sonec.training.export import export_from_rollouts

    if not export:
        console.print("sonec train --step                 # small SFT+RL step (repeat)")
        console.print("sonec train --export -r …          # export shards only")
        raise typer.Exit(code=0)
    sealed: set[str] = set()
    if exclude_sealed:
        for suite in (
            Path("examples/benchmarks/sonecbench_v1.json"),
            Path("examples/benchmarks/worldbench_v1.json"),
        ):
            if suite.exists():
                data = json.loads(suite.read_text(encoding="utf-8"))
                for t in data.get("tasks") or []:
                    sealed.add(t["id"])
    written = export_from_rollouts(rollouts, out, sealed_ids=sealed)
    for name, path in written.items():
        console.print(f"{name}: {path}")
    console.print(f"manifest: {out / 'manifest.json'}")


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8787, "--port"),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
    provider: str = typer.Option("local", "--provider"),
    model: str = typer.Option("sonec", "--model", "-m"),
) -> None:
    """IDE/CLI gateway — OpenAI-compatible agent HTTP server."""
    from sonec.serve import serve_blocking

    serve_blocking(host=host, port=port, workspace=workspace, provider=provider, model=model)


@app.command("mcp")
def mcp_cmd() -> None:
    """Run MCP stdio server for Cursor / VS Code / Claude Desktop."""
    from sonec.ide.mcp_server import main as mcp_main

    mcp_main()


@app.command("doctor")
def doctor_cmd() -> None:
    """Check inference endpoint + harness readiness."""
    import httpx

    from sonec.core.config import load_settings
    from sonec.harness.versioning import HARNESS_VERSION
    from sonec.models import BASE_HF, BASE_MODEL, PRODUCT_MODEL

    settings = load_settings()
    rows: list[tuple[str, str]] = [
        ("sonec", __version__),
        ("harness", HARNESS_VERSION),
        ("product", PRODUCT_MODEL),
        ("base", BASE_MODEL),
        ("base_hf", BASE_HF),
        ("provider", settings._normalized_provider()),
        ("base_url", settings.resolved_base_url()),
    ]
    base = settings.resolved_base_url()
    try:
        r = httpx.get(f"{base}/models", timeout=3.0)
        if r.status_code == 200:
            data = r.json()
            ids = [m.get("id", "") for m in data.get("data", [])] if isinstance(data, dict) else []
            rows.append(("inference", "ok"))
            rows.append(("models", ", ".join(ids[:8]) or "(empty list)"))
        else:
            rows.append(("inference", f"http {r.status_code} at {base}/models"))
    except Exception as exc:  # noqa: BLE001
        rows.append(("inference", f"unreachable ({exc})"))
    for p in (
        Path("NOTICE"),
        Path("LICENSE"),
        Path("examples/benchmarks/worldbench_v1.json"),
        Path("configs/sft/mlx_lora.yaml"),
    ):
        rows.append((str(p), "ok" if p.exists() else "MISSING"))
    table = Table("Check", "Status")
    for k, v in rows:
        table.add_row(k, v)
    console.print(table)
    console.print(
        "\nIterate:\n"
        "1) Point SONEC_BASE_URL at any OpenAI-compatible server serving qwen3.5:2b / sonec\n"
        "2) sonec train --step\n"
        "3) sonec run \"…\" -w .\n"
        "4) sonec worldbench --run --live --limit 5\n"
        "5) Repeat --step with higher --sft-iters / --train-n over time"
    )


if __name__ == "__main__":
    app()
