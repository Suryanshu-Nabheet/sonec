"""Canonical product identity for sonec.

Chat and marketing copy: sonec is a coding model. Weight lineage lives in NOTICE.
"""

from __future__ import annotations

# Training / serve plumbing only (Apache-2.0 lineage — see NOTICE).
BASE_MODEL = "qwen3.5:2b"
BASE_HF = "Qwen/Qwen3.5-2B"
BASE_HF_MLX = "mlx-community/Qwen3.5-2B-4bit"

PRODUCT_MODEL = "sonec"
PRODUCT_AUTHOR = "Suryanshu Nabheet"
PRODUCT_TAGLINE = "sonec by Suryanshu Nabheet — coding model"

PRODUCT_IDENTITY = (
    f"You are {PRODUCT_MODEL}, a coding model created by {PRODUCT_AUTHOR}.\n"
    "Identify as sonec when asked. Prefer verified evidence. "
    "Do not invent unread file contents."
)

PRODUCT_SYSTEM = PRODUCT_IDENTITY

# API model id for serve-llm (mlx_lm). Product name remains PRODUCT_MODEL.
DEFAULT_MODEL = BASE_HF_MLX
DEFAULT_PROVIDER = "local"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8080/v1"

BASE_HF_CANDIDATES: tuple[str, ...] = (
    BASE_HF,
    BASE_HF_MLX,
    "Qwen/Qwen3.5-4B",
)

BASE_OLLAMA_MODEL = BASE_MODEL
PRODUCT_OLLAMA_MODEL = PRODUCT_MODEL
BASE_HF_HINT = BASE_HF
BASE_PARAM_CLASS = "2B"
