"""Dataset generation and training pipeline scaffolding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from sonec.core.types import utc_now


class TrajectoryStep(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_name: str | None = None


class TrainingExample(BaseModel):
    id: str
    task: str
    trajectory: list[TrajectoryStep]
    outcome: Literal["success", "failure"]
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())


class DatasetManifest(BaseModel):
    name: str
    version: str = "0.1.0"
    examples: list[TrainingExample] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> DatasetManifest:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class DatasetGenerator:
    """Builds training examples from agent transcripts / synthetic templates."""

    def __init__(self, name: str = "sonec-se") -> None:
        self.name = name
        self._examples: list[TrainingExample] = []

    def add_example(self, example: TrainingExample) -> None:
        self._examples.append(example)

    def add_from_messages(
        self,
        *,
        example_id: str,
        task: str,
        messages: list[dict[str, Any]],
        outcome: Literal["success", "failure"],
        metadata: dict[str, Any] | None = None,
    ) -> TrainingExample:
        trajectory = [
            TrajectoryStep(
                role=msg["role"],
                content=str(msg.get("content") or ""),
                tool_name=msg.get("name"),
            )
            for msg in messages
            if msg.get("role") in {"system", "user", "assistant", "tool"}
        ]
        example = TrainingExample(
            id=example_id,
            task=task,
            trajectory=trajectory,
            outcome=outcome,
            metadata=metadata or {},
        )
        self.add_example(example)
        return example

    def synthesize_smoke_examples(self) -> list[TrainingExample]:
        samples = [
            (
                "write-readme",
                "Create a README.md describing the project",
                [
                    {"role": "user", "content": "Create a README.md describing the project"},
                    {
                        "role": "assistant",
                        "content": "I will write README.md with purpose, install, and usage.",
                    },
                    {"role": "tool", "name": "fs_write", "content": "Wrote README.md"},
                    {"role": "assistant", "content": "Created README.md."},
                ],
            ),
            (
                "fix-bug",
                "Fix the failing test in tests/test_math.py",
                [
                    {"role": "user", "content": "Fix the failing test in tests/test_math.py"},
                    {"role": "assistant", "content": "Reading the test and implementation."},
                    {"role": "tool", "name": "fs_read", "content": "..."},
                    {"role": "tool", "name": "fs_edit", "content": "Edited src/math.py"},
                    {"role": "tool", "name": "terminal_run", "content": "exit_code: 0"},
                    {"role": "assistant", "content": "Fixed off-by-one; tests pass."},
                ],
            ),
        ]
        created: list[TrainingExample] = []
        for example_id, task, messages in samples:
            created.append(
                self.add_from_messages(
                    example_id=example_id,
                    task=task,
                    messages=messages,
                    outcome="success",
                    metadata={"synthetic": True},
                )
            )
        return created

    def manifest(self) -> DatasetManifest:
        return DatasetManifest(name=self.name, examples=list(self._examples))


class TrainingPipeline:
    """Prepares dataset shards and a training config for external trainers.

    SONEC does not ship model weights or a GPU trainer. It produces the
    artifacts an external training stack (Axolotl, torchtune, etc.) expects.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_jsonl(self, manifest: DatasetManifest, filename: str = "train.jsonl") -> Path:
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            for example in manifest.examples:
                record = {
                    "id": example.id,
                    "task": example.task,
                    "messages": [step.model_dump() for step in example.trajectory],
                    "outcome": example.outcome,
                    "metadata": example.metadata,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path

    def write_config(
        self,
        *,
        model: str = "sonec",
        dataset_file: str = "train.jsonl",
        epochs: int = 1,
        learning_rate: float = 1e-5,
    ) -> Path:
        config = {
            "model": model,
            "base": "qwen3.5:2b",
            "dataset": str(self.output_dir / dataset_file),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "format": "chat_trajectory",
            "notes": (
                "Specialize the local sonec model (Qwen 3.5 class). "
                "Use configs/sft/mlx_lora.yaml on Apple Silicon."
            ),
        }
        path = self.output_dir / "train_config.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return path
