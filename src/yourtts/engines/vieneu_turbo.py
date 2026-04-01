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
        # VieNeu turbo models output 24kHz audio. Sync runtime sample-rate to avoid
        # playback artifacts when writing WAV with an outdated project config.
        model_sr = getattr(self._tts, "sample_rate", None)
        if isinstance(model_sr, int) and model_sr > 0:
            self.sample_rate = model_sr
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

    def list_voices(self) -> list[str]:
        tts = self._load_tts()
        try:
            voices = tts.list_preset_voices()
        except Exception:
            fallback = (self.voice or "default").strip()
            return [fallback or "default"]

        if not isinstance(voices, list):
            fallback = (self.voice or "default").strip()
            return [fallback or "default"]

        normalized: list[str] = []
        for item in voices:
            if isinstance(item, tuple):
                if len(item) >= 2 and str(item[1]).strip():
                    normalized.append(str(item[1]).strip())
                    continue
                if len(item) >= 1 and str(item[0]).strip():
                    normalized.append(str(item[0]).strip())
                    continue
            text = str(item).strip()
            if text:
                normalized.append(text)

        if not normalized:
            fallback = (self.voice or "default").strip()
            return [fallback or "default"]
        # Keep order while removing duplicates.
        return list(dict.fromkeys(normalized))

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

    def synthesize_waveform(self, text: str, voice: str | None = None, ref_audio: str | None = None) -> np.ndarray:
        # Send the full text directly to VieNeu turbo to avoid chunk boundary artifacts.
        checked_text = self.validate_text(text)
        key = self._make_cache_key(checked_text, voice, ref_audio=ref_audio)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        waveform = self.infer(checked_text, voice=voice, ref_audio=ref_audio)
        if waveform.ndim != 1:
            raise ValueError("Engine infer() must return a mono 1D waveform.")

        output = waveform.astype(np.float32)
        self._cache_put(key, output)
        return output
