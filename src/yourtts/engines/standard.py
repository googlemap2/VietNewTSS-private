from __future__ import annotations

import numpy as np

from yourtts.base import BaseEngine


class StandardEngine(BaseEngine):
    """MVP engine that generates deterministic test audio."""

    def infer(self, text: str, voice: str | None = None) -> np.ndarray:
        prompt = self.validate_text(text)

        duration_sec = max(0.35, min(4.0, len(prompt) * 0.04))
        t = np.linspace(0.0, duration_sec, int(self.sample_rate * duration_sec), endpoint=False)

        selected_voice = (voice or self.voice).lower()
        base_hz = 220.0 if selected_voice == "default" else 196.0
        waveform = 0.2 * np.sin(2.0 * np.pi * base_hz * t)

        fade_samples = max(1, int(0.02 * self.sample_rate))
        fade = np.linspace(0.0, 1.0, fade_samples)
        waveform[:fade_samples] *= fade
        waveform[-fade_samples:] *= fade[::-1]

        return waveform.astype(np.float32)
