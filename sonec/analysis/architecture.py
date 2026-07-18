"""Architecture analysis: dependency graph of Python modules."""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from sonec.core.workspace import Workspace


@dataclass(frozen=True)
class ArchitectureReport:
    modules: list[str]
    edges: list[tuple[str, str]]
    cycles: list[list[str]]
    fan_in: dict[str, int]
    fan_out: dict[str, int]


class ArchitectureAnalyzer:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def analyze(self, package_root: str = ".") -> ArchitectureReport:
        root = self.workspace.resolve(package_root)
        module_map: dict[Path, str] = {}
        for file in root.rglob("*.py"):
            rel = file.relative_to(self.workspace.root)
            parts = list(rel.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            module_map[file] = ".".join(parts) if parts else rel.stem

        edges: set[tuple[str, str]] = set()
        for file, module in module_map.items():
            text = file.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        target = alias.name
                        if any(target == m or target.startswith(m + ".") for m in module_map.values()):
                            edges.add((module, target))
                elif isinstance(node, ast.ImportFrom) and node.module:
                    target = node.module
                    if any(target == m or target.startswith(m + ".") for m in module_map.values()):
                        edges.add((module, target))

        modules = sorted(set(module_map.values()))
        fan_out: dict[str, int] = defaultdict(int)
        fan_in: dict[str, int] = defaultdict(int)
        for src, dst in edges:
            fan_out[src] += 1
            fan_in[dst] += 1
        cycles = _find_cycles(modules, edges)
        return ArchitectureReport(
            modules=modules,
            edges=sorted(edges),
            cycles=cycles,
            fan_in=dict(fan_in),
            fan_out=dict(fan_out),
        )


def _find_cycles(modules: list[str], edges: set[tuple[str, str]]) -> list[list[str]]:
    graph: dict[str, list[str]] = defaultdict(list)
    for src, dst in edges:
        graph[src].append(dst)
    cycles: list[list[str]] = []
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def dfs(node: str) -> None:
        visiting.add(node)
        stack.append(node)
        for nxt in graph.get(node, []):
            if nxt in visiting:
                if nxt in stack:
                    idx = stack.index(nxt)
                    cycle = stack[idx:] + [nxt]
                    if cycle not in cycles and cycle[::-1] not in cycles:
                        cycles.append(cycle)
            elif nxt not in visited:
                dfs(nxt)
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for module in modules:
        if module not in visited:
            dfs(module)
    return cycles[:20]


def report_to_mermaid(report: ArchitectureReport) -> str:
    lines = ["graph LR"]
    for src, dst in report.edges:
        lines.append(f'  "{src}" --> "{dst}"')
    if not report.edges:
        lines.append('  empty["No internal edges"]')
    return "\n".join(lines)
