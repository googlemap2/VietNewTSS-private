from __future__ import annotations

from yourtts.base import BaseEngine
from yourtts.engines.standard import StandardEngine


def create_engine(mode: str = "standard", **kwargs) -> BaseEngine:
    normalized = mode.strip().lower()
    if normalized == "standard":
        return StandardEngine(**kwargs)
    raise ValueError(f"Unsupported engine mode: {mode}")
