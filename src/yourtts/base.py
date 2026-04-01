from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Iterable
import hashlib

import numpy as np
import soundfile as sf

from yourtts.utils.audio import crossfade_concat
from yourtts.utils.text import split_text_chunks


class BaseEngine(ABC):
    """Shared base behavior for all engines."""

    def __init__(
        self,
        sample_rate: int = 22050,
        output_dir: str = "outputs",
        voice: str = "default",
        device: str = "cpu",
        model_name: str = "standard-sine-mvp",
        cache_size: int = 128,
    ) -> None:
        self.sample_rate = sample_rate
        self.output_dir = Path(output_dir)
        self.voice = voice
        self.device = device
        self.model_name = model_name
        self.cache_size = max(0, int(cache_size))
        self._wave_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def infer(self, text: str, voice: str | None = None, ref_audio: str | None = None) -> np.ndarray:
        """Return mono audio waveform as float32 array in range [-1, 1]."""

    def list_voices(self) -> list[str]:
        default_voice = (self.voice or "default").strip()
        return [default_voice or "default"]

    def validate_text(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Input text must not be empty.")
        return cleaned

    def _make_cache_key(self, text: str, voice: str | None, ref_audio: str | None = None) -> str:
        resolved_voice = (voice or self.voice).strip().lower()
        resolved_ref = (ref_audio or "").strip()
        payload = f"{self.model_name}|{self.sample_rate}|{resolved_voice}|{resolved_ref}|{text}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> np.ndarray | None:
        if key not in self._wave_cache:
            self.cache_misses += 1
            return None
        self._wave_cache.move_to_end(key)
        self.cache_hits += 1
        return self._wave_cache[key]

    def _cache_put(self, key: str, waveform: np.ndarray) -> None:
        if self.cache_size <= 0:
            return
        self._wave_cache[key] = waveform
        self._wave_cache.move_to_end(key)
        while len(self._wave_cache) > self.cache_size:
            self._wave_cache.popitem(last=False)

    def cache_stats(self) -> dict:
        return {
            "size": len(self._wave_cache),
            "capacity": self.cache_size,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
        }

    def synthesize_waveform(self, text: str, voice: str | None = None, ref_audio: str | None = None) -> np.ndarray:
        checked_text = self.validate_text(text)
        key = self._make_cache_key(checked_text, voice, ref_audio=ref_audio)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        chunks = split_text_chunks(checked_text)
        if not chunks:
            raise ValueError("No valid text chunks were generated from input.")

        wave_chunks = [self.infer(chunk, voice=voice, ref_audio=ref_audio) for chunk in chunks]
        waveform = crossfade_concat(wave_chunks, sample_rate=self.sample_rate)
        if waveform.ndim != 1:
            raise ValueError("Engine infer() must return a mono 1D waveform.")

        output = waveform.astype(np.float32)
        self._cache_put(key, output)
        return output

    def warmup(self) -> dict:
        self.synthesize_waveform("Warmup synthesis path for yourtts.", voice=self.voice)
        return self.cache_stats()

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        voice: str | None = None,
        ref_audio: str | None = None,
    ) -> str:
        waveform = self.synthesize_waveform(text=text, voice=voice, ref_audio=ref_audio)

        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, waveform, self.sample_rate)
        return str(target)

    def synthesize_batch_to_files(
        self,
        texts: Iterable[str],
        output_dir: str | None = None,
        voice: str | None = None,
        ref_audio: str | None = None,
        prefix: str = "batch",
    ) -> list[str]:
        target_dir = Path(output_dir) if output_dir else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        produced: list[str] = []
        for idx, text in enumerate(texts, start=1):
            name = f"{prefix}_{idx:03d}.wav"
            out = target_dir / name
            produced.append(self.synthesize_to_file(text=text, output_path=str(out), voice=voice, ref_audio=ref_audio))
        return produced
