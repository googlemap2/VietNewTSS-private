from __future__ import annotations

import numpy as np

from yourtts.base import BaseEngine


class VieneuTurboEngine(BaseEngine):
    """VieNeu Turbo engine wrapper targeting GGUF models via vieneu SDK."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._tts = None

    def _load_tts(self):
        if self._tts is not None:
            return self._tts

        try:
            from vieneu import Vieneu  # type: ignore
        except ImportError as exc:
            raise RuntimeError("vieneu is not installed. Run: pip install -e .[vieneu]") from exc

        model = self.model_name or "pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf"
        self._tts = Vieneu(mode="turbo", model_name=model)
        return self._tts

    def _resolve_voice(self, voice: str | None):
        tts = self._load_tts()
        selected = (voice or self.voice or "").strip()
        if not selected:
            return None

        try:
            return tts.get_preset_voice(selected)
        except Exception:
            # Fall back to raw voice id/string if preset lookup fails.
            return selected

    def infer(self, text: str, voice: str | None = None, ref_audio: str | None = None) -> np.ndarray:
        prompt = self.validate_text(text)
        tts = self._load_tts()

        kwargs = {"text": prompt}
        resolved_voice = self._resolve_voice(voice)
        if resolved_voice is not None:
            kwargs["voice"] = resolved_voice

        if ref_audio:
            try:
                ref_voice = tts.encode_reference(ref_audio)
                kwargs["voice"] = ref_voice
            except Exception:
                # Keep resolved preset/default voice fallback.
                pass

        audio = tts.infer(**kwargs)
        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)

        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if peak > 1.0:
            waveform = waveform / peak
        return waveform.astype(np.float32)
