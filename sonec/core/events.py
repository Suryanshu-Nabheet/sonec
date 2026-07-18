"""Simple synchronous event bus for agent observability."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from sonec.core.types import AgentEvent

EventHandler = Callable[[AgentEvent], None]


class EventEmitter(Protocol):
    def emit(self, event: AgentEvent) -> None: ...

    def subscribe(self, handler: EventHandler) -> None: ...


class EventBus:
    """In-process pub/sub for agent lifecycle events."""

    def __init__(self) -> None:
        self._handlers: list[EventHandler] = []
        self.history: list[AgentEvent] = []

    def subscribe(self, handler: EventHandler) -> None:
        self._handlers.append(handler)

    def emit(self, event: AgentEvent) -> None:
        self.history.append(event)
        for handler in list(self._handlers):
            handler(event)
