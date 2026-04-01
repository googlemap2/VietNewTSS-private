from pathlib import Path

import soundfile as sf

from yourtts.factory import create_engine


def test_standard_engine_synthesize_to_file(tmp_path: Path) -> None:
    engine = create_engine(mode="standard", sample_rate=22050, output_dir=str(tmp_path), voice="default")
    out_file = tmp_path / "engine_path.wav"

    generated = engine.synthesize_to_file(
        text="This is a phase three engine path test. It should produce a non-empty wav file.",
        output_path=str(out_file),
        voice="warm",
    )

    assert Path(generated).exists()
    data, sr = sf.read(generated)
    assert sr == 22050
    assert data.size > 0


def test_standard_engine_batch_and_cache(tmp_path: Path) -> None:
    engine = create_engine(mode="standard", sample_rate=22050, output_dir=str(tmp_path), voice="default", cache_size=8)

    out = engine.synthesize_batch_to_files(
        texts=["line one", "line two", "line three"],
        output_dir=str(tmp_path / "batch_out"),
        voice="bright",
        prefix="test_batch",
    )
    assert len(out) == 3
    assert all(Path(p).exists() for p in out)

    stats_before = engine.cache_stats()
    engine.synthesize_waveform("cache me", voice="default")
    engine.synthesize_waveform("cache me", voice="default")
    stats_after = engine.cache_stats()

    assert stats_after["hits"] >= stats_before["hits"] + 1


def test_standard_engine_warmup_populates_cache(tmp_path: Path) -> None:
    engine = create_engine(mode="standard", output_dir=str(tmp_path), cache_size=4)
    stats = engine.warmup()
    assert stats["size"] >= 1
