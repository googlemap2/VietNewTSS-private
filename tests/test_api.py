import io
from pathlib import Path

import apps.api as api
import numpy as np
from yourtts.factory import create_engine
from yourtts.utils.srt import decode_srt_bytes, fit_waveform_to_duration, parse_srt_text, speed_up_waveform


def _use_standard_engine(tmp_path: Path | None = None) -> None:
    output_dir = str(tmp_path) if tmp_path else str(api.config.get("output_dir", "outputs"))
    api.engine = create_engine(
        mode="standard",
        sample_rate=int(api.config.get("sample_rate", 22050)),
        output_dir=output_dir,
        voice=str(api.config.get("voice", "default")),
    )
    if tmp_path:
        api.config["output_dir"] = str(tmp_path)


def test_api_health_and_voices() -> None:
    _use_standard_engine()
    client = api.app.test_client()

    health = client.get("/health")
    assert health.status_code == 200
    body = health.get_json()
    assert body["status"] == "ok"

    voices = client.get("/voices")
    assert voices.status_code == 200
    payload = voices.get_json()
    assert "voices" in payload


def test_api_synthesize_and_batch(tmp_path: Path) -> None:
    _use_standard_engine(tmp_path)

    client = api.app.test_client()

    single = client.post("/synthesize", json={"text": "hello api", "voice": "default"})
    assert single.status_code == 200
    single_payload = single.get_json()
    assert Path(single_payload["output_path"]).exists()

    batch = client.post(
        "/synthesize_batch",
        json={"texts": ["a", "b", "c"], "voice": "warm"},
    )
    assert batch.status_code == 200
    batch_payload = batch.get_json()
    assert batch_payload["count"] == 3
    assert all(Path(p).exists() for p in batch_payload["output_paths"])


def test_api_stream_and_warmup() -> None:
    _use_standard_engine()
    client = api.app.test_client()

    warmup = client.post("/warmup")
    assert warmup.status_code == 200
    assert warmup.get_json()["status"] == "ok"

    stream = client.post("/synthesize_stream", json={"text": "stream me", "voice": "bright"})
    assert stream.status_code == 200
    assert stream.content_type.startswith("audio/wav")
    assert len(stream.data) > 44


def test_srt_utils_and_file_endpoint(tmp_path: Path) -> None:
    _use_standard_engine(tmp_path)
    client = api.app.test_client()

    srt_path = Path("nhioa.srt")
    raw_text = decode_srt_bytes(srt_path.read_bytes())
    segments = parse_srt_text(raw_text)

    assert len(segments) >= 10
    assert all(seg.text.strip() for seg in segments)

    response = client.post(
        "/synthesize_srt_file",
        data={"srt_file": (io.BytesIO(srt_path.read_bytes()), srt_path.name), "voice": "default"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["segment_count"] == len(segments)
    assert Path(payload["output_path"]).exists()
    assert "sped_up_segments" in payload
    assert "max_speed_factor" in payload
    assert payload["fast_mode"] is False
    assert payload["sped_up_segments"] == 0
    assert payload["max_speed_factor"] == 1.0


def test_fit_waveform_to_duration_speeds_up_when_needed() -> None:
    wave = np.linspace(-1.0, 1.0, num=2000, dtype=np.float32)
    fitted, speed = fit_waveform_to_duration(wave, sample_rate=1000, target_ms=1000)
    assert fitted.size == 1000
    assert speed == 2.0


def test_speed_up_waveform_respects_factor() -> None:
    wave = np.linspace(-1.0, 1.0, num=1000, dtype=np.float32)
    faster = speed_up_waveform(wave, 1.25)
    assert faster.size == 800


def test_srt_file_endpoint_fast_mode_is_manual(tmp_path: Path) -> None:
    _use_standard_engine(tmp_path)
    client = api.app.test_client()

    srt_path = Path("nhioa.srt")
    response = client.post(
        "/synthesize_srt_file",
        data={
            "srt_file": (io.BytesIO(srt_path.read_bytes()), srt_path.name),
            "voice": "default",
            "fast_mode": "true",
            "fast_speed": "1.25",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["fast_mode"] is True
    assert payload["fast_speed"] == 1.25
    assert payload["sped_up_segments"] > 0
