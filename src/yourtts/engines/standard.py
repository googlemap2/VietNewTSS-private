from __future__ import annotations

import numpy as np

from yourtts.base import BaseEngine
from yourtts.utils.voices import load_voices


class StandardEngine(BaseEngine):
    """MVP engine that generates deterministic test audio."""

    def list_voices(self) -> list[str]:
        return list(load_voices().keys())

    def infer(self, text: str, voice: str | None = None, ref_audio: str | None = None, **kwargs) -> np.ndarray:
        prompt = self.validate_text(text)

        duration_sec = max(0.35, min(4.0, len(prompt) * 0.04))
        t = np.linspace(0.0, duration_sec, int(self.sample_rate * duration_sec), endpoint=False)

        presets = load_voices()
        selected_voice = (voice or self.voice).lower()
        selected = presets.get(selected_voice, presets.get("default", {}))
        base_hz = float(selected.get("base_hz", 220.0))
        gain = float(selected.get("gain", 0.2))
        waveform = gain * np.sin(2.0 * np.pi * base_hz * t)

        fade_samples = max(1, int(0.02 * self.sample_rate))
        fade = np.linspace(0.0, 1.0, fade_samples)
        waveform[:fade_samples] *= fade
        waveform[-fade_samples:] *= fade[::-1]

        return waveform.astype(np.float32)
