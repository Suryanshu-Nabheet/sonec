"""Weight readiness — product is LoRA, not a Modelfile."""

from __future__ import annotations

from pathlib import Path

from sonec.training.weights import weight_status, write_product_manifest


def test_weight_status_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    status = weight_status(tmp_path / "missing")
    assert status.ready is False
    assert "no adapter" in status.detail.lower() or "train" in status.detail.lower()


def test_weight_status_config_only_not_ready(tmp_path: Path) -> None:
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    status = weight_status(d)
    assert status.ready is False
    assert status.has_config is True
    assert "safetensors" in status.detail.lower()


def test_weight_status_ready_with_safetensors(tmp_path: Path) -> None:
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "adapters.safetensors").write_bytes(b"0" * 2048)
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    status = weight_status(d)
    assert status.ready is True
    assert status.author == "Suryanshu Nabheet"
    manifest = write_product_manifest(adapter_dir=d, mlx_base="mlx-community/Qwen3.5-2B-4bit", root=tmp_path)
    assert manifest.exists()
    assert "lora_adapter" in manifest.read_text(encoding="utf-8") or "mlx_lora" in manifest.read_text(
        encoding="utf-8"
    )


def test_weight_status_tiny_tensor_not_ready(tmp_path: Path) -> None:
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "adapters.safetensors").write_bytes(b"fake")
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    status = weight_status(d)
    assert status.ready is False
    assert status.has_safetensors is False
