"""Product weight readiness — specialized LoRA checkpoints for sonec."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sonec.models import BASE_HF_MLX, BASE_MODEL, PRODUCT_MODEL

DEFAULT_ADAPTER_DIR = Path("artifacts/train/checkpoints/sonec-sft-mlx")
PRODUCT_MANIFEST = Path("artifacts/train/PRODUCT.json")


@dataclass(frozen=True)
class WeightStatus:
    ready: bool
    adapter_dir: Path
    has_safetensors: bool
    has_config: bool
    detail: str
    base_model: str = BASE_HF_MLX
    product: str = PRODUCT_MODEL

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "product": self.product,
            "base_model": self.base_model,
            "adapter_dir": str(self.adapter_dir),
            "has_safetensors": self.has_safetensors,
            "has_config": self.has_config,
            "detail": self.detail,
        }


def adapter_weight_files(adapter_dir: Path) -> list[Path]:
    if not adapter_dir.is_dir():
        return []
    return sorted(adapter_dir.glob("*.safetensors"))


def weight_status(adapter_dir: Path | None = None) -> WeightStatus:
    path = (adapter_dir or DEFAULT_ADAPTER_DIR).expanduser().resolve()
    tensors = adapter_weight_files(path)
    has_cfg = (path / "adapter_config.json").exists()
    if tensors:
        return WeightStatus(
            ready=True,
            adapter_dir=path,
            has_safetensors=True,
            has_config=has_cfg,
            detail=f"adapter weights: {', '.join(p.name for p in tensors)}",
        )
    if has_cfg:
        return WeightStatus(
            ready=False,
            adapter_dir=path,
            has_safetensors=False,
            has_config=True,
            detail=(
                "adapter_config.json present but no *.safetensors — "
                "training incomplete; run sonec train --step"
            ),
        )
    return WeightStatus(
        ready=False,
        adapter_dir=path,
        has_safetensors=False,
        has_config=False,
        detail="no adapter directory — run: sonec train --step",
    )


def write_product_manifest(
    *,
    adapter_dir: Path,
    mlx_base: str,
    root: Path | None = None,
) -> Path:
    """Record product weight location and readiness."""
    root = root or Path.cwd()
    status = weight_status(adapter_dir)
    resolved = adapter_dir.expanduser().resolve()
    try:
        adapter_rel = str(resolved.relative_to(root.resolve()))
    except ValueError:
        adapter_rel = str(resolved)
    payload = {
        "product": PRODUCT_MODEL,
        "author": "Suryanshu Nabheet",
        "kind": "mlx_lora_adapter",
        "base_tag": BASE_MODEL,
        "base_mlx": mlx_base,
        "adapter_dir": adapter_rel,
        "ready": status.ready,
        "weights": [p.name for p in adapter_weight_files(adapter_dir)],
        "serve": (
            f"sonec serve-llm  # or: python -m mlx_lm server --model {mlx_base} "
            f"--adapter-path {adapter_rel} --port 8080"
        ),
        "note": (
            "Product sonec is the LoRA adapter under the adapter path. "
            "Optional chat Modelfiles are runners only."
        ),
    }
    out = root / PRODUCT_MANIFEST
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def mlx_server_command(
    *,
    adapter_dir: Path | None = None,
    model: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> list[str]:
    status = weight_status(adapter_dir)
    if not status.ready:
        raise RuntimeError(status.detail)
    base = model or BASE_HF_MLX
    return [
        "python",
        "-m",
        "mlx_lm",
        "server",
        "--model",
        base,
        "--adapter-path",
        str(status.adapter_dir),
        "--host",
        host,
        "--port",
        str(port),
    ]
