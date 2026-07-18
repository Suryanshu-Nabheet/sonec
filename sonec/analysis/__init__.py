"""Analysis package: review, debug, refactor, architecture."""

from sonec.analysis.architecture import ArchitectureAnalyzer, ArchitectureReport, report_to_mermaid
from sonec.analysis.debug import ParsedTraceback, StackFrame, parse_traceback, suggest_debug_plan
from sonec.analysis.refactor import RefactorAnalyzer, RefactorOpportunity
from sonec.analysis.review import CodeReviewer, ReviewFinding, findings_to_markdown

__all__ = [
    "ArchitectureAnalyzer",
    "ArchitectureReport",
    "CodeReviewer",
    "ParsedTraceback",
    "RefactorAnalyzer",
    "RefactorOpportunity",
    "ReviewFinding",
    "StackFrame",
    "findings_to_markdown",
    "parse_traceback",
    "report_to_mermaid",
    "suggest_debug_plan",
]
