"""Agent memory: short-term transcript + durable notes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from sonec.core.types import Message, new_id, utc_now


class MemoryStore(Protocol):
    def add_message(self, message: Message) -> None: ...

    def messages(self) -> list[Message]: ...

    def add_note(self, text: str, *, tags: list[str] | None = None) -> str: ...

    def search_notes(self, query: str, *, limit: int = 10) -> list[dict[str, object]]: ...

    def clear_messages(self) -> None: ...


class InMemoryStore:
    """Process-local memory suitable for single runs and tests."""

    def __init__(self) -> None:
        self._messages: list[Message] = []
        self._notes: list[dict[str, object]] = []

    def add_message(self, message: Message) -> None:
        self._messages.append(message)

    def messages(self) -> list[Message]:
        return list(self._messages)

    def add_note(self, text: str, *, tags: list[str] | None = None) -> str:
        note_id = new_id("note_")
        self._notes.append(
            {
                "id": note_id,
                "text": text,
                "tags": list(tags or []),
                "created_at": utc_now().isoformat(),
            }
        )
        return note_id

    def search_notes(self, query: str, *, limit: int = 10) -> list[dict[str, object]]:
        needle = query.lower().strip()
        if not needle:
            return self._notes[:limit]
        hits = [
            note
            for note in self._notes
            if needle in str(note["text"]).lower()
            or any(needle in str(tag).lower() for tag in note.get("tags", []))  # type: ignore[union-attr]
        ]
        return hits[:limit]

    def clear_messages(self) -> None:
        self._messages.clear()


class FileMemoryStore(InMemoryStore):
    """Persists notes to disk; transcript stays in-memory per run."""

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._notes_path = self.directory / "notes.jsonl"
        self._load_notes()

    def _load_notes(self) -> None:
        if not self._notes_path.exists():
            return
        with self._notes_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                self._notes.append(json.loads(line))

    def add_note(self, text: str, *, tags: list[str] | None = None) -> str:
        note_id = super().add_note(text, tags=tags)
        note = self._notes[-1]
        with self._notes_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(note, ensure_ascii=False) + "\n")
        return note_id
