from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np


@dataclass(slots=True)
class SRTSegment:
    index: int
    start_ms: int
    end_ms: int
    text: str


_TIMECODE_RE = re.compile(
    r"^\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*$"
)


def _timecode_to_ms(hours: str, minutes: str, seconds: str, millis: str) -> int:
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1_000
        + int(millis)
    )


def decode_srt_bytes(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "cp1258", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_srt_text(raw_text: str) -> list[SRTSegment]:
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    segments: list[SRTSegment] = []
    blocks = re.split(r"\n\s*\n", normalized)
    for block in blocks:
        lines = [line.strip("\ufeff") for line in block.split("\n")]
        lines = [line for line in lines if line.strip()]
        if len(lines) < 2:
            continue

        cursor = 0
        try:
            seg_index = int(lines[0].strip())
            cursor = 1
        except ValueError:
            seg_index = len(segments) + 1

        if cursor >= len(lines):
            continue

        match = _TIMECODE_RE.match(lines[cursor].strip())
        if not match:
            continue

        start_ms = _timecode_to_ms(*match.groups()[:4])
        end_ms = _timecode_to_ms(*match.groups()[4:])
        text = " ".join(line.strip() for line in lines[cursor + 1 :] if line.strip())
        if not text:
            continue

        segments.append(
            SRTSegment(
                index=seg_index,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
            )
        )

    return segments


def fit_waveform_to_duration(
    waveform: np.ndarray,
    sample_rate: int,
    target_ms: int,
) -> tuple[np.ndarray, float]:
    wave = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if wave.size == 0 or sample_rate <= 0 or target_ms <= 0:
        return wave, 1.0

    target_samples = max(1, int(round(sample_rate * target_ms / 1000.0)))
    if wave.size <= target_samples:
        return wave, 1.0

    src_x = np.linspace(0.0, 1.0, num=wave.size, endpoint=True)
    dst_x = np.linspace(0.0, 1.0, num=target_samples, endpoint=True)
    stretched = np.interp(dst_x, src_x, wave).astype(np.float32)
    speed_factor = float(wave.size) / float(target_samples)
    return stretched, speed_factor


def speed_up_waveform(
    waveform: np.ndarray,
    speed_factor: float,
) -> np.ndarray:
    wave = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if wave.size == 0 or speed_factor <= 1.0:
        return wave

    target_samples = max(1, int(round(wave.size / float(speed_factor))))
    src_x = np.linspace(0.0, 1.0, num=wave.size, endpoint=True)
    dst_x = np.linspace(0.0, 1.0, num=target_samples, endpoint=True)
    return np.interp(dst_x, src_x, wave).astype(np.float32)
