"""Debugging helpers: traceback parsing and failing-test localization."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class StackFrame:
    path: str
    line: int
    function: str
    text: str = ""


@dataclass(frozen=True)
class ParsedTraceback:
    error_type: str
    error_message: str
    frames: list[StackFrame]


TRACE_FRAME = re.compile(
    r'File "(?P<path>[^"]+)", line (?P<line>\d+)(?:, in (?P<func>.+))?'
)
TRACE_ERROR = re.compile(r"^(?P<type>[A-Za-z_][\w.]*)(?::\s*(?P<msg>.*))?$")


def parse_traceback(text: str) -> ParsedTraceback:
    frames: list[StackFrame] = []
    lines = text.splitlines()
    error_type = "Error"
    error_message = ""
    i = 0
    while i < len(lines):
        match = TRACE_FRAME.search(lines[i])
        if match:
            snippet = lines[i + 1].strip() if i + 1 < len(lines) else ""
            frames.append(
                StackFrame(
                    path=match.group("path"),
                    line=int(match.group("line")),
                    function=(match.group("func") or "").strip(),
                    text=snippet,
                )
            )
            i += 2
            continue
        err = TRACE_ERROR.match(lines[i].strip())
        if err and frames:
            error_type = err.group("type")
            error_message = (err.group("msg") or "").strip()
        i += 1
    return ParsedTraceback(error_type=error_type, error_message=error_message, frames=frames)


def suggest_debug_plan(tb: ParsedTraceback) -> list[str]:
    steps = [
        f"Identify failure: {tb.error_type}: {tb.error_message or '(no message)'}",
    ]
    if tb.frames:
        top = tb.frames[-1]
        steps.append(f"Inspect top frame `{top.path}:{top.line}` in `{top.function or '<module>'}`")
        if len(tb.frames) > 1:
            origin = tb.frames[0]
            steps.append(f"Trace call origin `{origin.path}:{origin.line}`")
    steps.extend(
        [
            "Reproduce with the smallest failing command or test",
            "Form a hypothesis, apply a minimal fix, re-run the failing command",
        ]
    )
    return steps
