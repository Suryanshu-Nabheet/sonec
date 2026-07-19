"""CUDA / Unsloth / Axolotl / CPU backend selection tests."""

from __future__ import annotations

import json
from pathlib import Path

from sonec.training.backends import (
    ADAPTER_DIRS,
    CPU_BASE_HF,
    detect_backend,
    write_axolotl_config,
    write_unsloth_dataset,
)


def test_detect_backend_explicit_mlx() -> None:
    info = detect_backend("mlx")
    assert info.name == "mlx"
    assert info.adapter_dir == ADAPTER_DIRS["mlx"]


def test_detect_backend_explicit_unsloth() -> None:
    info = detect_backend("unsloth")
    assert info.name == "unsloth"
    assert info.adapter_dir == ADAPTER_DIRS["unsloth"]
    # This CI host has no CUDA — must report unavailable, not fake-ready.
    assert info.available is False
    assert "CUDA" in info.detail or "unsloth" in info.detail.lower()


def test_detect_backend_explicit_cpu() -> None:
    info = detect_backend("cpu")
    assert info.name == "cpu"
    assert info.adapter_dir == ADAPTER_DIRS["cpu"]
    assert info.base_model == CPU_BASE_HF
    # Availability depends on whether torch/peft are installed in this env.
    assert isinstance(info.available, bool)
    assert info.detail


def test_detect_backend_auto_never_crashes() -> None:
    info = detect_backend("auto")
    assert info.name in {"mlx", "unsloth", "axolotl", "cpu"}
    assert info.adapter_dir.name.startswith("sonec-sft-")


def test_adapter_dirs_include_cpu() -> None:
    assert "cpu" in ADAPTER_DIRS
    assert ADAPTER_DIRS["cpu"].name == "sonec-sft-cpu"


def test_write_unsloth_dataset(tmp_path: Path) -> None:
    mlx = tmp_path / "mlx_data"
    mlx.mkdir()
    (mlx / "train.jsonl").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "yo"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = write_unsloth_dataset(mlx, tmp_path / "messages.jsonl")
    rows = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["messages"][0]["role"] == "user"


def test_write_axolotl_config(tmp_path: Path) -> None:
    template = Path("configs/sft/axolotl_qlora.yml")
    assert template.exists()
    out = write_axolotl_config(
        template=template,
        out_path=tmp_path / "run.yml",
        dataset_path=tmp_path / "train.jsonl",
        output_dir=tmp_path / "ckpt",
        max_steps=120,
    )
    text = out.read_text(encoding="utf-8")
    assert "max_steps: 120" in text
    assert "Qwen/Qwen3.5-2B" in text
    assert "chat_template" in text
