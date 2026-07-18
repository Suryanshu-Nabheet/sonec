"""Static code review heuristics (no LLM required)."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from sonec.core.workspace import Workspace


@dataclass(frozen=True)
class ReviewFinding:
    path: str
    line: int
    severity: str
    rule: str
    message: str


class CodeReviewer:
    """Fast, deterministic review pass for Python workspaces."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def review_path(self, relative: str) -> list[ReviewFinding]:
        path = self.workspace.resolve(relative)
        if path.is_dir():
            findings: list[ReviewFinding] = []
            for file in path.rglob("*.py"):
                rel = self.workspace.relative_to_root(file)
                findings.extend(self.review_file(rel))
            return findings
        return self.review_file(relative)

    def review_file(self, relative: str) -> list[ReviewFinding]:
        path = self.workspace.resolve(relative)
        text = path.read_text(encoding="utf-8", errors="replace")
        findings: list[ReviewFinding] = []
        findings.extend(self._regex_rules(relative, text))
        if path.suffix == ".py":
            findings.extend(self._python_ast_rules(relative, text))
        return findings

    def _regex_rules(self, relative: str, text: str) -> list[ReviewFinding]:
        findings: list[ReviewFinding] = []
        patterns = [
            (r"TODO|FIXME", "low", "todo_marker", "Leftover TODO/FIXME marker"),
            (r"password\s*=\s*['\"][^'\"]+['\"]", "high", "hardcoded_secret", "Possible hardcoded password"),
            (r"api[_-]?key\s*=\s*['\"][^'\"]+['\"]", "high", "hardcoded_secret", "Possible hardcoded API key"),
            (r"except\s*:", "medium", "bare_except", "Bare except clause"),
            (r"print\(", "low", "debug_print", "print() left in code"),
        ]
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern, severity, rule, message in patterns:
                if re.search(pattern, line, flags=re.IGNORECASE):
                    findings.append(
                        ReviewFinding(
                            path=relative,
                            line=line_no,
                            severity=severity,
                            rule=rule,
                            message=message,
                        )
                    )
        return findings

    def _python_ast_rules(self, relative: str, text: str) -> list[ReviewFinding]:
        findings: list[ReviewFinding] = []
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            return [
                ReviewFinding(
                    path=relative,
                    line=exc.lineno or 1,
                    severity="high",
                    rule="syntax_error",
                    message=f"SyntaxError: {exc.msg}",
                )
            ]
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and not ast.get_docstring(node)
                and not node.name.startswith("_")
            ):
                findings.append(
                    ReviewFinding(
                        path=relative,
                        line=getattr(node, "lineno", 1),
                        severity="low",
                        rule="missing_docstring",
                        message=f"Public function '{node.name}' lacks a docstring",
                    )
                )
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                for arg in list(node.args.args) + list(node.args.kwonlyargs):
                    if arg.annotation is None and arg.arg not in {"self", "cls"}:
                        findings.append(
                            ReviewFinding(
                                path=relative,
                                line=getattr(node, "lineno", 1),
                                severity="low",
                                rule="missing_type_hint",
                                message=f"Parameter '{arg.arg}' in '{node.name}' lacks a type hint",
                            )
                        )
        return findings


def findings_to_markdown(findings: list[ReviewFinding]) -> str:
    if not findings:
        return "No findings."
    lines = ["| Severity | Rule | Location | Message |", "| --- | --- | --- | --- |"]
    for item in findings:
        lines.append(
            f"| {item.severity} | `{item.rule}` | `{item.path}:{item.line}` | {item.message} |"
        )
    return "\n".join(lines)
