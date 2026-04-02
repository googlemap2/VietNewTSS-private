from __future__ import annotations

import json
from pathlib import Path


def load_voices(path: str = "assets/voices.json") -> dict:
    source = Path(path)
    fallback = {"default": {"base_hz": 220.0, "gain": 0.20, "description": "Fallback voice"}}
    if not source.exists():
        return fallback

    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    if not isinstance(payload, dict) or not payload:
        return fallback
    return payload


def voice_names(path: str = "assets/voices.json") -> list[str]:
    return list(load_voices(path).keys())
