"""Analysis module tests."""

from __future__ import annotations

from pathlib import Path

from sonec.analysis.architecture import ArchitectureAnalyzer
from sonec.analysis.debug import parse_traceback, suggest_debug_plan
from sonec.analysis.refactor import RefactorAnalyzer
from sonec.analysis.review import CodeReviewer
from sonec.core.workspace import Workspace


def test_review_finds_bare_except(tmp_path: Path) -> None:
    path = tmp_path / "bad.py"
    path.write_text("try:\n    x=1\nexcept:\n    pass\n", encoding="utf-8")
    findings = CodeReviewer(Workspace(tmp_path)).review_file("bad.py")
    assert any(f.rule == "bare_except" for f in findings)


def test_parse_traceback() -> None:
    text = '''Traceback (most recent call last):
  File "app.py", line 10, in main
    boom()
  File "app.py", line 4, in boom
    raise ValueError("nope")
ValueError: nope
'''
    tb = parse_traceback(text)
    assert tb.error_type == "ValueError"
    assert tb.frames
    assert suggest_debug_plan(tb)


def test_refactor_and_architecture(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "a.py").write_text("from pkg import b\n\ndef foo():\n    return 1\n", encoding="utf-8")
    (pkg / "b.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    ws = Workspace(tmp_path)
    ops = RefactorAnalyzer(ws).analyze(".")
    assert any(o.kind == "duplicate_function" for o in ops)
    report = ArchitectureAnalyzer(ws).analyze(".")
    assert report.modules
