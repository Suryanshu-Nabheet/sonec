"""Refactoring analysis: find duplication and oversized units."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass

from sonec.core.workspace import Workspace


@dataclass(frozen=True)
class RefactorOpportunity:
    kind: str
    path: str
    line: int
    message: str
    score: float


class RefactorAnalyzer:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def analyze(self, relative: str = ".") -> list[RefactorOpportunity]:
        root = self.workspace.resolve(relative)
        files = [root] if root.is_file() else list(root.rglob("*.py"))
        opportunities: list[RefactorOpportunity] = []
        bodies: dict[str, list[tuple[str, int, str]]] = {}

        for file in files:
            if not file.is_file() or file.suffix != ".py":
                continue
            rel = self.workspace.relative_to_root(file)
            text = file.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    segment = ast.get_source_segment(text, node) or ""
                    line_count = segment.count("\n") + 1
                    if line_count >= 80:
                        opportunities.append(
                            RefactorOpportunity(
                                kind="large_unit",
                                path=rel,
                                line=getattr(node, "lineno", 1),
                                message=f"{type(node).__name__} '{node.name}' is {line_count} lines",
                                score=min(line_count / 80.0, 3.0),
                            )
                        )
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        digest = hashlib.sha1(
                            ast.dump(node, annotate_fields=False).encode("utf-8")
                        ).hexdigest()
                        bodies.setdefault(digest, []).append((rel, getattr(node, "lineno", 1), node.name))

        for group in bodies.values():
            if len(group) < 2:
                continue
            for rel, line, name in group:
                others = [f"{p}:{n}" for p, _, n in group if not (p == rel and n == name)]
                opportunities.append(
                    RefactorOpportunity(
                        kind="duplicate_function",
                        path=rel,
                        line=line,
                        message=f"Function '{name}' appears duplicated with {', '.join(others)}",
                        score=2.0,
                    )
                )
        opportunities.sort(key=lambda item: item.score, reverse=True)
        return opportunities
