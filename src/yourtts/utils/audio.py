from __future__ import annotations

import numpy as np


def crossfade_concat(chunks: list[np.ndarray], sample_rate: int, crossfade_ms: int = 25) -> np.ndarray:
    """Join mono chunks with a simple linear crossfade."""
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0].astype(np.float32)

    fade_samples = max(1, int(sample_rate * crossfade_ms / 1000.0))
    out = chunks[0].astype(np.float32)

    for chunk in chunks[1:]:
        next_chunk = chunk.astype(np.float32)
        local_fade = min(fade_samples, len(out), len(next_chunk))
        if local_fade <= 0:
            out = np.concatenate([out, next_chunk])
            continue

        fade_out = np.linspace(1.0, 0.0, local_fade, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, local_fade, dtype=np.float32)

        blended = out[-local_fade:] * fade_out + next_chunk[:local_fade] * fade_in
        out = np.concatenate([out[:-local_fade], blended, next_chunk[local_fade:]])

    return out.astype(np.float32)
