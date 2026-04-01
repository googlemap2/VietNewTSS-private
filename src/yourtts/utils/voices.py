from __future__ import annotations

import json
from pathlib import Path


def load_voices(path: str = "assets/voices.json") -> dict:
    source = Path(path)
    if not source.exists():
        return {"default": {"base_hz": 220.0, "gain": 0.20, "description": "Fallback voice"}}

    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        return {"default": {"base_hz": 220.0, "gain": 0.20, "description": "Fallback voice"}}
    return payload


def voice_names(path: str = "assets/voices.json") -> list[str]:
    return list(load_voices(path).keys())
