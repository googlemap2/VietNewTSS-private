from __future__ import annotations

import re


def split_text_chunks(text: str, max_chars: int = 120) -> list[str]:
    """Split text into sentence-like chunks bounded by max_chars."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    candidates = re.split(r"(?<=[.!?])\s+", cleaned)
    chunks: list[str] = []
    current = ""

    for sentence in candidates:
        if not sentence:
            continue
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(sentence) <= max_chars:
            current = sentence
            continue

        # Hard split very long sentence into max_chars windows.
        for idx in range(0, len(sentence), max_chars):
            part = sentence[idx : idx + max_chars].strip()
            if part:
                chunks.append(part)
        current = ""

    if current:
        chunks.append(current)

    return chunks
