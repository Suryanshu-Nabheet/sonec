"""Canonical model identity for sonec.

Coding-agent model specialized from Qwen 3.5 (2B).
Serve via any OpenAI-compatible endpoint.
"""

from __future__ import annotations

# Base weights lineage (Apache-2.0 via Alibaba Qwen).
BASE_MODEL = "qwen3.5:2b"
BASE_HF = "Qwen/Qwen3.5-2B"
BASE_HF_MLX = "mlx-community/Qwen3.5-2B-4bit"

# Product checkpoint / served model name after specialization.
PRODUCT_MODEL = "sonec"

DEFAULT_MODEL = PRODUCT_MODEL
# "local" = OpenAI-compatible inference at SONEC_BASE_URL (any runner).
DEFAULT_PROVIDER = "local"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8080/v1"

BASE_HF_CANDIDATES: tuple[str, ...] = (
    BASE_HF,
    BASE_HF_MLX,
    "Qwen/Qwen3.5-4B",
)

# Back-compat aliases (older call sites).
BASE_OLLAMA_MODEL = BASE_MODEL
PRODUCT_OLLAMA_MODEL = PRODUCT_MODEL
BASE_HF_HINT = BASE_HF
BASE_PARAM_CLASS = "2B"
PRODUCT_TAGLINE = "Coding-agent model on Qwen 3.5 (2B)"
