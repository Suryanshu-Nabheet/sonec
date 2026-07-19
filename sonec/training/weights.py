"""Product weight readiness — specialized LoRA checkpoints for sonec."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sonec.models import BASE_HF, BASE_HF_MLX, BASE_MODEL, PRODUCT_MODEL
from sonec.training.backends import ADAPTER_DIRS

DEFAULT_ADAPTER_DIR = ADAPTER_DIRS["mlx"]
PRODUCT_MANIFEST = Path("artifacts/train/PRODUCT.json")
# Reject empty / placeholder tensors as "ready".
MIN_ADAPTER_BYTES = 1024


@dataclass(frozen=True)
class WeightStatus:
    ready: bool
    adapter_dir: Path
    has_safetensors: bool
    has_config: bool
    detail: str
    base_model: str = BASE_HF_MLX
    product: str = PRODUCT_MODEL
    author: str = "Suryanshu Nabheet"
    backend: str = "unknown"

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "product": self.product,
            "author": self.author,
            "backend": self.backend,
            "base_model": self.base_model,
            "adapter_dir": str(self.adapter_dir),
            "has_safetensors": self.has_safetensors,
            "has_config": self.has_config,
            "detail": self.detail,
        }


def _backend_for_dir(path: Path) -> str:
    name = path.name.lower()
    if "unsloth" in name:
        return "unsloth"
    if "axolotl" in name:
        return "axolotl"
    if "mlx" in name:
        return "mlx"
    return "unknown"


def adapter_weight_files(adapter_dir: Path) -> list[Path]:
    if not adapter_dir.is_dir():
        return []
    found: list[Path] = []
    for pattern in ("*.safetensors", "adapter_model.safetensors", "**/adapter_model.safetensors"):
        for p in adapter_dir.glob(pattern):
            if p.is_file() and p.stat().st_size >= MIN_ADAPTER_BYTES:
                found.append(p)
    # de-dupe
    uniq = {p.resolve(): p for p in found}
    return sorted(uniq.values(), key=lambda p: str(p))


def _has_adapter_config(adapter_dir: Path) -> bool:
    return (adapter_dir / "adapter_config.json").is_file() or any(
        adapter_dir.glob("**/adapter_config.json")
    )


def weight_status(adapter_dir: Path | None = None) -> WeightStatus:
    if adapter_dir is not None:
        candidates = [adapter_dir.expanduser().resolve()]
    else:
        # Prefer newest ready adapter across backends.
        candidates = [p.expanduser().resolve() for p in ADAPTER_DIRS.values()]

    best: WeightStatus | None = None
    for path in candidates:
        tensors = adapter_weight_files(path)
        has_cfg = _has_adapter_config(path)
        backend = _backend_for_dir(path)
        base = BASE_HF if backend in {"unsloth", "axolotl"} else BASE_HF_MLX
        if tensors and has_cfg:
            status = WeightStatus(
                ready=True,
                adapter_dir=path,
                has_safetensors=True,
                has_config=True,
                detail=f"adapter weights ({backend}): {', '.join(p.name for p in tensors[:4])}",
                base_model=base,
                backend=backend,
            )
            if best is None or not best.ready:
                best = status
            continue
        if best is not None and best.ready:
            continue
        if tensors and not has_cfg:
            best = WeightStatus(
                ready=False,
                adapter_dir=path,
                has_safetensors=True,
                has_config=False,
                detail="*.safetensors present but adapter_config.json missing",
                base_model=base,
                backend=backend,
            )
        elif has_cfg:
            best = WeightStatus(
                ready=False,
                adapter_dir=path,
                has_safetensors=False,
                has_config=True,
                detail=(
                    "adapter_config.json present but no usable *.safetensors — "
                    "training incomplete; run sonec train --step"
                ),
                base_model=base,
                backend=backend,
            )

    if best is not None:
        return best
    path = (adapter_dir or DEFAULT_ADAPTER_DIR).expanduser().resolve()
    return WeightStatus(
        ready=False,
        adapter_dir=path,
        has_safetensors=False,
        has_config=False,
        detail=(
            "no adapter directory — run: sonec train --step "
            "(--backend auto|mlx|unsloth|axolotl)"
        ),
        backend=_backend_for_dir(path),
    )


def write_product_manifest(
    *,
    adapter_dir: Path,
    mlx_base: str,
    root: Path | None = None,
    backend: str | None = None,
    base_hf: str | None = None,
) -> Path:
    """Record product weight location and readiness."""
    root = root or Path.cwd()
    status = weight_status(adapter_dir)
    resolved = adapter_dir.expanduser().resolve()
    try:
        adapter_rel = str(resolved.relative_to(root.resolve()))
    except ValueError:
        adapter_rel = str(resolved)
    kind = {
        "mlx": "mlx_lora_adapter",
        "unsloth": "peft_qlora_adapter",
        "axolotl": "peft_qlora_adapter",
    }.get(backend or status.backend, "lora_adapter")
    serve_hint = (
        f"sonec serve-llm --adapter {adapter_rel}"
        if (backend or status.backend) == "mlx"
        else (
            f"sonec serve-llm --backend peft --adapter {adapter_rel} "
            f"--model {base_hf or BASE_HF}"
        )
    )
    payload = {
        "product": PRODUCT_MODEL,
        "author": "Suryanshu Nabheet",
        "kind": kind,
        "backend": backend or status.backend,
        "base_tag": BASE_MODEL,
        "base_mlx": mlx_base,
        "base_hf": base_hf or BASE_HF,
        "adapter_dir": adapter_rel,
        "ready": status.ready,
        "weights": [p.name for p in adapter_weight_files(adapter_dir)],
        "serve": serve_hint,
        "note": (
            "Product sonec is the LoRA adapter under the adapter path. "
            "Optional chat Modelfiles are runners only. "
            "Linux CUDA: Unsloth (preferred) or Axolotl; Apple Silicon: MLX."
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


def peft_server_command(
    *,
    adapter_dir: Path | None = None,
    model: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> list[str]:
    """Serve PEFT/Unsloth/Axolotl adapters via text-generation-inference style helper.

    Uses a small local OpenAI-compatible server script when vLLM/TGI are absent.
    """
    status = weight_status(adapter_dir)
    if not status.ready:
        raise RuntimeError(status.detail)
    base = model or BASE_HF
    script = Path(__file__).with_name("peft_serve.py")
    return [
        "python",
        str(script),
        "--model",
        base,
        "--adapter",
        str(status.adapter_dir),
        "--host",
        host,
        "--port",
        str(port),
    ]
