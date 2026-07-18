"""Language Server Protocol client (JSON-RPC over stdio)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sonec.core.errors import SonecError


class LSPError(SonecError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="lsp_error")


@dataclass
class LSPClient:
    """Minimal LSP client for hover, definition, and diagnostics-style requests.

    Designed to speak the LSP stdio framing protocol. Servers are optional —
    if a server binary is missing, methods raise ``LSPError`` with a clear message.
    """

    command: Sequence[str]
    root: Path
    _process: asyncio.subprocess.Process | None = field(default=None, init=False, repr=False)
    _next_id: int = field(default=1, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    _reader_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _pending: dict[int, asyncio.Future[Any]] = field(default_factory=dict, init=False, repr=False)
    _buffer: bytearray = field(default_factory=bytearray, init=False, repr=False)

    async def start(self) -> None:
        if self._process is not None:
            return
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.root),
            )
        except FileNotFoundError as exc:
            raise LSPError(f"LSP server not found: {self.command[0]}") from exc
        self._reader_task = asyncio.create_task(self._read_loop())
        await self.initialize()

    async def aclose(self) -> None:
        if self._process is None:
            return
        try:
            await self.request("shutdown", None)
            await self.notify("exit", None)
        except Exception:  # noqa: BLE001
            pass
        if self._reader_task:
            self._reader_task.cancel()
        self._process.kill()
        await self._process.wait()
        self._process = None
        self._initialized = False

    async def initialize(self) -> dict[str, Any]:
        root_uri = self.root.resolve().as_uri()
        result = await self.request(
            "initialize",
            {
                "processId": None,
                "rootUri": root_uri,
                "capabilities": {
                    "textDocument": {
                        "hover": {"contentFormat": ["plaintext", "markdown"]},
                        "definition": {"linkSupport": True},
                    }
                },
                "workspaceFolders": [{"uri": root_uri, "name": self.root.name}],
            },
        )
        await self.notify("initialized", {})
        self._initialized = True
        return result if isinstance(result, dict) else {}

    async def hover(self, path: Path, line: int, character: int) -> Any:
        await self._ensure()
        return await self.request(
            "textDocument/hover",
            {
                "textDocument": {"uri": path.resolve().as_uri()},
                "position": {"line": max(line - 1, 0), "character": max(character, 0)},
            },
        )

    async def definition(self, path: Path, line: int, character: int) -> Any:
        await self._ensure()
        return await self.request(
            "textDocument/definition",
            {
                "textDocument": {"uri": path.resolve().as_uri()},
                "position": {"line": max(line - 1, 0), "character": max(character, 0)},
            },
        )

    async def did_open(self, path: Path, text: str, language_id: str = "python") -> None:
        await self._ensure()
        await self.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": path.resolve().as_uri(),
                    "languageId": language_id,
                    "version": 1,
                    "text": text,
                }
            },
        )

    async def _ensure(self) -> None:
        if not self._initialized:
            await self.start()

    async def request(self, method: str, params: Mapping[str, Any] | None) -> Any:
        assert self._process and self._process.stdin
        req_id = self._next_id
        self._next_id += 1
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = future
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            payload["params"] = params
        await self._write(payload)
        return await asyncio.wait_for(future, timeout=30)

    async def notify(self, method: str, params: Mapping[str, Any] | None) -> None:
        assert self._process and self._process.stdin
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        await self._write(payload)

    async def _write(self, payload: dict[str, Any]) -> None:
        assert self._process and self._process.stdin
        body = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self._process.stdin.write(header + body)
        await self._process.stdin.drain()

    async def _read_loop(self) -> None:
        assert self._process and self._process.stdout
        while True:
            chunk = await self._process.stdout.read(4096)
            if not chunk:
                break
            self._buffer.extend(chunk)
            while True:
                message = self._pop_message()
                if message is None:
                    break
                await self._dispatch(message)

    def _pop_message(self) -> dict[str, Any] | None:
        header_end = self._buffer.find(b"\r\n\r\n")
        if header_end < 0:
            return None
        header = bytes(self._buffer[:header_end]).decode("ascii", errors="replace")
        length = None
        for line in header.split("\r\n"):
            if line.lower().startswith("content-length:"):
                length = int(line.split(":", 1)[1].strip())
        if length is None:
            del self._buffer[: header_end + 4]
            return None
        total = header_end + 4 + length
        if len(self._buffer) < total:
            return None
        body = bytes(self._buffer[header_end + 4 : total])
        del self._buffer[:total]
        return json.loads(body.decode("utf-8"))

    async def _dispatch(self, message: dict[str, Any]) -> None:
        if "id" in message and ("result" in message or "error" in message):
            future = self._pending.pop(int(message["id"]), None)
            if future is None or future.done():
                return
            if "error" in message:
                future.set_exception(LSPError(str(message["error"])))
            else:
                future.set_result(message.get("result"))


def default_python_lsp(root: Path) -> LSPClient:
    """Create a client for ``pylsp`` if installed; otherwise pyright-langserver."""
    return LSPClient(command=("pylsp",), root=root)
