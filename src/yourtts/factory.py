from __future__ import annotations

from yourtts.base import BaseEngine
from yourtts.engines.standard import StandardEngine
from yourtts.engines.vieneu_turbo import VieneuTurboEngine


def create_engine(mode: str = "standard", **kwargs) -> BaseEngine:
    normalized = mode.strip().lower()
    if normalized == "standard":
        return StandardEngine(**kwargs)
    if normalized in {"turbo", "vieneu", "gguf"}:
        return VieneuTurboEngine(**kwargs)
    raise ValueError(f"Unsupported engine mode: {mode}")
