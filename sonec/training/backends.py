"""LoRA training backends — MLX (Apple Silicon) and CUDA (Unsloth / Axolotl).

Auto-select:
  - Apple Silicon + mlx-lm → mlx
  - CUDA + unsloth → unsloth (preferred on Linux)
  - CUDA + axolotl → axolotl
  - else → dry guidance (no fake success)
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sonec.models import BASE_HF, BASE_HF_MLX

BackendName = Literal["auto", "mlx", "unsloth", "axolotl"]

ADAPTER_DIRS = {
    "mlx": Path("artifacts/train/checkpoints/sonec-sft-mlx"),
    "unsloth": Path("artifacts/train/checkpoints/sonec-sft-unsloth"),
    "axolotl": Path("artifacts/train/checkpoints/sonec-sft-axolotl"),
}


@dataclass(frozen=True)
class BackendInfo:
    name: BackendName
    available: bool
    detail: str
    adapter_dir: Path
    base_model: str


@dataclass
class SFTReport:
    phase: str
    ok: bool
    detail: str
    paths: dict[str, str]
    backend: str = ""


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def cuda_available() -> bool:
    if not _module_available("torch"):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }


def detect_backend(preferred: BackendName = "auto") -> BackendInfo:
    """Pick a concrete backend. Never silently claim success without deps."""
    if preferred == "mlx":
        ok = _module_available("mlx_lm")
        return BackendInfo(
            name="mlx",
            available=ok,
            detail="mlx-lm ready" if ok else "pip install 'sonec[train]' (Apple Silicon)",
            adapter_dir=ADAPTER_DIRS["mlx"],
            base_model=BASE_HF_MLX,
        )
    if preferred == "unsloth":
        ok = cuda_available() and _module_available("unsloth")
        detail = (
            "unsloth + CUDA ready"
            if ok
            else "need CUDA torch + unsloth (pip install 'sonec[train-cuda]')"
        )
        return BackendInfo(
            name="unsloth",
            available=ok,
            detail=detail,
            adapter_dir=ADAPTER_DIRS["unsloth"],
            base_model=BASE_HF,
        )
    if preferred == "axolotl":
        ok = cuda_available() and shutil.which("accelerate") is not None
        # axolotl may be installed as module without top-level import name always.
        ax_ok = ok and (
            _module_available("axolotl")
            or subprocess.run(
                [sys.executable, "-c", "import axolotl"],
                capture_output=True,
                check=False,
            ).returncode
            == 0
        )
        return BackendInfo(
            name="axolotl",
            available=ax_ok,
            detail=(
                "axolotl + CUDA ready"
                if ax_ok
                else "need CUDA + axolotl (pip install 'sonec[train-axolotl]')"
            ),
            adapter_dir=ADAPTER_DIRS["axolotl"],
            base_model=BASE_HF,
        )

    # auto
    if is_apple_silicon() and _module_available("mlx_lm"):
        return detect_backend("mlx")
    if cuda_available() and _module_available("unsloth"):
        return detect_backend("unsloth")
    if cuda_available():
        ax = detect_backend("axolotl")
        if ax.available:
            return ax
        return BackendInfo(
            name="unsloth",
            available=False,
            detail=(
                "CUDA detected but Unsloth/Axolotl missing — "
                "pip install 'sonec[train-cuda]' or 'sonec[train-axolotl]'"
            ),
            adapter_dir=ADAPTER_DIRS["unsloth"],
            base_model=BASE_HF,
        )
    if _module_available("mlx_lm"):
        return detect_backend("mlx")
    return BackendInfo(
        name="mlx",
        available=False,
        detail=(
            "No training backend ready. Apple Silicon: pip install 'sonec[train]'. "
            "Linux CUDA: pip install 'sonec[train-cuda]' (Unsloth) or "
            "'sonec[train-axolotl]'. H2O LLM Studio is a GUI alternative "
            "(import artifacts/train/sft_corpus/train_chat.jsonl)."
        ),
        adapter_dir=ADAPTER_DIRS["mlx"],
        base_model=BASE_HF_MLX,
    )


def write_unsloth_dataset(mlx_dir: Path, out_path: Path) -> Path:
    """Convert mlx chat JSONL → Unsloth/TRL messages JSONL."""
    src = mlx_dir / "train.jsonl"
    if not src.is_file():
        raise FileNotFoundError(f"missing {src}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open(encoding="utf-8") as inp, out_path.open("w", encoding="utf-8") as out:
        for line in inp:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages") or row
            if not isinstance(messages, list):
                continue
            out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            n += 1
    if n == 0:
        raise RuntimeError(f"empty dataset from {src}")
    return out_path


def write_axolotl_config(
    *,
    template: Path,
    out_path: Path,
    dataset_path: Path,
    output_dir: Path,
    base_model: str = BASE_HF,
    max_steps: int | None = None,
) -> Path:
    """Materialize an Axolotl YAML pinned to this run's corpus."""
    import yaml

    raw = yaml.safe_load(template.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"invalid axolotl template: {template}")
    raw["base_model"] = base_model
    raw["output_dir"] = str(output_dir)
    raw["datasets"] = [
        {
            "path": str(dataset_path),
            "type": "chat_template",
            "field_messages": "messages",
            "message_property_mappings": {
                "role": "role",
                "content": "content",
            },
            "roles_to_train": ["assistant"],
        }
    ]
    if max_steps is not None and max_steps > 0:
        raw["max_steps"] = int(max_steps)
        raw.pop("num_epochs", None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return out_path


def run_unsloth_sft(
    *,
    data_dir: Path,
    adapter_path: Path,
    model: str = BASE_HF,
    iters: int = 300,
    learning_rate: float = 1e-5,
    max_seq_length: int = 2048,
) -> SFTReport:
    adapter_path.mkdir(parents=True, exist_ok=True)
    dataset = write_unsloth_dataset(data_dir, adapter_path / "train_messages.jsonl")
    script = Path(__file__).with_name("unsloth_train.py")
    cmd = [
        sys.executable,
        str(script),
        "--model",
        model,
        "--data",
        str(dataset),
        "--out",
        str(adapter_path),
        "--steps",
        str(max(1, iters)),
        "--lr",
        str(learning_rate),
        "--max-seq-length",
        str(max_seq_length),
    ]
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        return SFTReport("sft", False, str(exc), {}, backend="unsloth")
    log_path = adapter_path / "sft_log.txt"
    log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
    ok = proc.returncode == 0 and _adapter_ready(adapter_path)
    return SFTReport(
        phase="sft",
        ok=ok,
        detail=(
            f"backend=unsloth steps={iters} model={model} "
            f"elapsed_s={time.perf_counter() - started:.1f} rc={proc.returncode}"
        ),
        paths={"adapter": str(adapter_path), "log": str(log_path), "data": str(dataset)},
        backend="unsloth",
    )


def run_axolotl_sft(
    *,
    data_dir: Path,
    adapter_path: Path,
    model: str = BASE_HF,
    iters: int = 300,
    root: Path | None = None,
) -> SFTReport:
    root = root or Path.cwd()
    adapter_path.mkdir(parents=True, exist_ok=True)
    # Prefer chat messages JSONL from mlx_data or sibling train_chat.
    chat = data_dir / "train.jsonl"
    if not chat.is_file():
        alt = data_dir.parent / "train_chat.jsonl"
        if alt.is_file():
            chat = alt
        else:
            return SFTReport(
                "sft",
                False,
                f"missing chat dataset under {data_dir}",
                {},
                backend="axolotl",
            )
    template = root / "configs" / "sft" / "axolotl_qlora.yml"
    if not template.is_file():
        return SFTReport("sft", False, f"missing {template}", {}, backend="axolotl")
    cfg = write_axolotl_config(
        template=template,
        out_path=adapter_path / "axolotl_run.yml",
        dataset_path=chat.resolve(),
        output_dir=adapter_path.resolve(),
        base_model=model,
        max_steps=iters,
    )
    cmd = [
        "accelerate",
        "launch",
        "-m",
        "axolotl.cli.train",
        str(cfg),
    ]
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        return SFTReport("sft", False, str(exc), {}, backend="axolotl")
    log_path = adapter_path / "sft_log.txt"
    log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
    ok = proc.returncode == 0 and _adapter_ready(adapter_path)
    return SFTReport(
        phase="sft",
        ok=ok,
        detail=(
            f"backend=axolotl steps={iters} model={model} "
            f"elapsed_s={time.perf_counter() - started:.1f} rc={proc.returncode}"
        ),
        paths={"adapter": str(adapter_path), "log": str(log_path), "config": str(cfg)},
        backend="axolotl",
    )


def _adapter_ready(adapter_path: Path) -> bool:
    tensors = list(adapter_path.glob("*.safetensors")) + list(
        adapter_path.glob("**/adapter_model.safetensors")
    )
    cfg = (adapter_path / "adapter_config.json").exists() or (
        adapter_path / "adapter_config.json"
    ).exists()
    # PEFT often writes adapter_model.safetensors + adapter_config.json
    peft_tensor = adapter_path / "adapter_model.safetensors"
    if peft_tensor.is_file() and peft_tensor.stat().st_size >= 1024:
        return (adapter_path / "adapter_config.json").is_file() or cfg
    return any(p.is_file() and p.stat().st_size >= 1024 for p in tensors) and (
        (adapter_path / "adapter_config.json").is_file()
        or any(adapter_path.glob("**/adapter_config.json"))
    )


def run_sft(
    *,
    backend: BackendName,
    data_dir: Path,
    adapter_path: Path | None = None,
    model: str | None = None,
    iters: int = 300,
    root: Path | None = None,
) -> SFTReport:
    info = detect_backend(backend)
    if not info.available:
        return SFTReport(
            "sft",
            False,
            info.detail,
            {},
            backend=str(info.name),
        )
    out = adapter_path or info.adapter_dir
    base = model or info.base_model
    if info.name == "mlx":
        from sonec.training.specialize import run_mlx_sft

        report = run_mlx_sft(
            data_dir=data_dir,
            adapter_path=out,
            model=base,
            iters=iters,
        )
        return SFTReport(
            phase=report.phase,
            ok=report.ok,
            detail=f"backend=mlx {report.detail}",
            paths=report.paths,
            backend="mlx",
        )
    if info.name == "unsloth":
        return run_unsloth_sft(
            data_dir=data_dir,
            adapter_path=out,
            model=base,
            iters=iters,
        )
    return run_axolotl_sft(
        data_dir=data_dir,
        adapter_path=out,
        model=base,
        iters=iters,
        root=root,
    )
