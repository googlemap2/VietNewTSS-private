import numpy as np

from yourtts.utils.audio import crossfade_concat
from yourtts.utils.text import split_text_chunks
from yourtts.utils.voices import load_voices, voice_names


def test_split_text_chunks_respects_length() -> None:
    text = "Sentence one. Sentence two is slightly longer. Sentence three ends."
    chunks = split_text_chunks(text, max_chars=24)
    assert chunks
    assert all(len(chunk) <= 24 for chunk in chunks)


def test_crossfade_concat_non_empty() -> None:
    a = np.ones(200, dtype=np.float32)
    b = np.ones(200, dtype=np.float32) * 0.5
    out = crossfade_concat([a, b], sample_rate=20000, crossfade_ms=10)
    assert out.ndim == 1
    assert out.dtype == np.float32
    assert len(out) < len(a) + len(b)


def test_voice_loading() -> None:
    voices = load_voices()
    names = voice_names()
    assert "default" in voices
    assert "default" in names
