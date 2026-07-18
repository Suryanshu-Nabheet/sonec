"""Qwen tool-call markup parser."""

from __future__ import annotations

from sonec.llm.tool_parse import parse_qwen_tool_calls


def test_parse_qwen_fs_write() -> None:
    content = """<tool_call>
<function=fs_write>
<parameter=path>
notes/hello.txt
</parameter>
<parameter=content>
hello sonec
</parameter>
</function>
</tool_call>"""
    calls = parse_qwen_tool_calls(content)
    assert len(calls) == 1
    assert calls[0].name == "fs_write"
    assert calls[0].arguments["path"] == "notes/hello.txt"
    assert calls[0].arguments["content"] == "hello sonec"


def test_parse_empty() -> None:
    assert parse_qwen_tool_calls(None) == []
    assert parse_qwen_tool_calls("just text") == []
