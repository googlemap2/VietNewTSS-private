from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import soundfile as sf


class BaseEngine(ABC):
    """Shared base behavior for all engines."""

    def __init__(self, sample_rate: int = 22050, output_dir: str = "outputs", voice: str = "default") -> None:
        self.sample_rate = sample_rate
        self.output_dir = Path(output_dir)
        self.voice = voice
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def infer(self, text: str, voice: str | None = None) -> np.ndarray:
        """Return mono audio waveform as float32 array in range [-1, 1]."""

    def validate_text(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Input text must not be empty.")
        return cleaned

    def synthesize_to_file(self, text: str, output_path: str, voice: str | None = None) -> str:
        checked_text = self.validate_text(text)
        waveform = self.infer(checked_text, voice=voice)

        if waveform.ndim != 1:
            raise ValueError("Engine infer() must return a mono 1D waveform.")

        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, waveform, self.sample_rate)
        return str(target)
