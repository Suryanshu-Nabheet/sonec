"""Dataset generation and training pipeline scaffolding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from sonec.core.types import utc_now


class TrajectoryStep(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


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
        trajectory: list[TrajectoryStep] = []
        for msg in messages:
            role = msg.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            trajectory.append(
                TrajectoryStep(
                    role=role,
                    content=str(msg.get("content") or ""),
                    tool_name=msg.get("name"),
                    tool_call_id=msg.get("tool_call_id"),
                    tool_calls=msg.get("tool_calls"),
                )
            )
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
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "fs_write",
                                    "arguments": {
                                        "path": "README.md",
                                        "content": "# Project\n\nInstall and usage.\n",
                                    },
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "fs_write",
                        "tool_call_id": "c1",
                        "content": "Wrote README.md",
                    },
                    {"role": "assistant", "content": "Created README.md."},
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
    """Prepares dataset shards and a training config for external trainers."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_jsonl(self, manifest: DatasetManifest, filename: str = "train.jsonl") -> Path:
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            for example in manifest.examples:
                messages: list[dict[str, Any]] = []
                for step in example.trajectory:
                    msg: dict[str, Any] = {"role": step.role, "content": step.content}
                    if step.tool_name:
                        msg["name"] = step.tool_name
                    if step.tool_call_id:
                        msg["tool_call_id"] = step.tool_call_id
                    if step.tool_calls:
                        msg["tool_calls"] = step.tool_calls
                    messages.append(msg)
                record = {
                    "id": example.id,
                    "task": example.task,
                    "messages": messages,
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
            "format": "openai_tool_calls",
            "notes": (
                "Specialize sonec with real assistant.tool_calls trajectories. "
                "Use configs/sft/mlx_lora.yaml on Apple Silicon."
            ),
        }
        path = self.output_dir / "train_config.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return path
