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

from sonec.core.types import ToolCall

_TOOL_BLOCK = re.compile(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_PARAM = re.compile(
    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)


def parse_qwen_tool_calls(content: str | None) -> list[ToolCall]:
    if not content or "<tool_call>" not in content:
        return []
    calls: list[ToolCall] = []
    for match in _TOOL_BLOCK.finditer(content):
        name = match.group(1).strip()
        body = match.group(2)
        arguments: dict[str, Any] = {}
        for pm in _PARAM.finditer(body):
            arguments[pm.group(1).strip()] = pm.group(2)
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
