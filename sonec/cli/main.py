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
from sonec.app import build_agent
from sonec.core.config import load_settings
from sonec.core.types import AgentEvent, AgentEventKind, Message, Role, ToolCall
from sonec.docsgen.generator import DocGenerator
from sonec.eval.harness import EvalHarness
from sonec.indexing.index import RepositoryIndex
from sonec.llm.provider import MockProvider
from sonec.training.pipeline import DatasetGenerator, TrainingPipeline

app = typer.Typer(
    name="sonec",
    help="SONEC — Senior Open-source Neural Engineering Companion",
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
    """Print the SONEC version."""
    console.print(__version__)


@app.command("run")
def run_cmd(
    goal: str = typer.Argument(..., help="Engineering goal for the agent"),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w", help="Workspace root"),
    provider: str | None = typer.Option(None, "--provider", help="moonshot|openai|mock|..."),
    model: str | None = typer.Option(None, "--model", "-m"),
    max_iterations: int | None = typer.Option(None, "--max-iterations"),
    mock: bool = typer.Option(False, "--mock", help="Use scripted mock provider (offline)"),
    simple: bool = typer.Option(
        False, "--simple", help="Use single-loop runtime instead of multi-phase harness"
    ),
) -> None:
    """Run the advanced multi-phase harness (default) against a goal."""

    async def _run() -> None:
        from sonec.app import build_harness

        overrides: dict[str, object] = {"workspace": workspace}
        if provider:
            overrides["provider"] = provider
        if model:
            overrides["model"] = model
        if max_iterations:
            overrides["max_iterations"] = max_iterations
        if mock:
            overrides["provider"] = "mock"
        settings = load_settings(**overrides)
        llm = None
        if mock or settings.provider == "mock":
            llm = MockProvider.harness_smoke(goal)

        console.print(
            Panel.fit(
                f"[bold]{goal}[/]\nworkspace: {workspace.resolve()}\n"
                f"mode: {'simple' if simple else 'harness'} · "
                f"provider: {settings.provider} / {settings.model}",
                title="SONEC",
            )
        )

        if simple:
            runtime, cfg, ws, _registry = build_agent(
                settings=settings, provider=llm, persist_memory=True
            )
            runtime.events.subscribe(_print_event)
            result = await runtime.run(goal)
            provider_obj = runtime.provider
        else:
            harness, cfg, ws, _registry = build_harness(
                settings=settings, provider=llm, persist_memory=True
            )
            harness.events.subscribe(_print_event)
            result = await harness.run(goal)
            provider_obj = harness.provider

        del cfg, ws
        console.print()
        console.print(
            Panel(
                Markdown(result.final_message or "(empty)"),
                title="final" if result.success else "failed",
                border_style="green" if result.success else "red",
            )
        )
        if hasattr(provider_obj, "aclose"):
            await provider_obj.aclose()  # type: ignore[attr-defined]

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
    mock: bool = typer.Option(True, "--mock/--live", help="Mock agent responses by default"),
) -> None:
    """Run an evaluation suite."""

    async def _eval() -> None:
        settings = load_settings(workspace=workspace.resolve(), provider="mock" if mock else "moonshot")
        harness = EvalHarness(workspace=settings.workspace)
        task_list = EvalHarness.load_tasks(tasks)

        def factory():
            llm = None
            if mock:
                # Offline demo: perform a representative write, then finish.
                llm = MockProvider(
                    [
                        Message(
                            role=Role.ASSISTANT,
                            content=None,
                            tool_calls=[
                                ToolCall(
                                    id="eval_write",
                                    name="fs_write",
                                    arguments={
                                        "path": "notes/hello.txt",
                                        "content": "hello sonec",
                                    },
                                )
                            ],
                        ),
                        Message(
                            role=Role.ASSISTANT,
                            content="Created notes/hello.txt and completed the evaluation task.",
                        ),
                    ]
                )
            runtime, *_ = build_agent(settings=settings, provider=llm, persist_memory=False)
            return runtime

        report = await harness.run_suite(task_list, factory, name=tasks.stem)
        console.print(
            json.dumps(
                {
                    "name": report.name,
                    "pass_rate": report.pass_rate,
                    "mean_duration_s": report.mean_duration_s,
                    "results": [
                        {
                            "task_id": r.task_id,
                            "passed": r.passed,
                            "score": r.score,
                            "details": r.details,
                        }
                        for r in report.results
                    ],
                },
                indent=2,
            )
        )

    asyncio.run(_eval())


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
    config = pipeline.write_config()
    manifest.save(out / "manifest.json")
    console.print(f"Wrote {jsonl}")
    console.print(f"Wrote {config}")
    console.print(f"Examples: {len(manifest.examples)}")


if __name__ == "__main__":
    app()
