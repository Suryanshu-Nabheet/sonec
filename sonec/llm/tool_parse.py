"""Parse Qwen-native tool call markup from assistant content.

Qwen 3.5 chat templates teach:

    <tool_call>
    <function=fs_write>
    <parameter=path>
    notes/hello.txt
    </parameter>
    ...
    </function>
    </tool_call>

OpenAI-compatible servers may surface this as ``content`` instead of
``tool_calls``. Specialization trains this format; the runtime must execute it.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from sonec.core.coerce import coerce_bool, coerce_int
from sonec.core.types import ToolCall

_TOOL_BLOCK = re.compile(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_PARAM = re.compile(
    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)

_BOOL_KEYS = frozenset({"recursive", "enabled", "force", "all"})
_INT_KEYS = frozenset(
    {"max_entries", "offset", "limit", "max_matches", "max_results", "timeout"}
)


def _coerce_param(key: str, raw: str) -> Any:
    text = raw.strip()
    if key in _BOOL_KEYS:
        return coerce_bool(text, default=False)
    if key in _INT_KEYS:
        coerced = coerce_int(text, default=None)
        return coerced if coerced is not None else text
    return text


def parse_qwen_tool_calls(content: str | None) -> list[ToolCall]:
    if not content or "<tool_call>" not in content:
        return []
    calls: list[ToolCall] = []
    for match in _TOOL_BLOCK.finditer(content):
        name = match.group(1).strip()
        body = match.group(2)
        arguments: dict[str, Any] = {}
        for pm in _PARAM.finditer(body):
            key = pm.group(1).strip()
            arguments[key] = _coerce_param(key, pm.group(2))
        if not name:
            continue
        calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}",
                name=name,
                arguments=arguments,
            )
        )
    return calls
