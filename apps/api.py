from __future__ import annotations

from datetime import datetime
import io
from pathlib import Path

from yourtts.utils.env import load_dotenv

load_dotenv()

from flask import Flask, Response, jsonify, request
import numpy as np
import soundfile as sf
import yaml

from yourtts.factory import create_engine
from yourtts.utils.srt import decode_srt_bytes, parse_srt_text, speed_up_waveform


def load_config() -> dict:
    path = Path("config.yaml")
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


config = load_config()
engine = create_engine(
    mode=config.get("engine_mode", "standard"),
    sample_rate=int(config.get("sample_rate", 22050)),
    output_dir=config.get("output_dir", "outputs"),
    voice=config.get("voice", "default"),
    device=config.get("device", "cpu"),
    model_name=config.get("model_name", "pnnbao-ump/VieNeu-TTS-0.3B-q8-gguf"),
    cache_size=int(config.get("cache_size", 128)),
)
available_voices = engine.list_voices()

app = Flask(__name__)


def _make_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _synthesize_srt_segments(
    segments,
    voice: str | None,
    ref_audio: str | None = None,
    fast_mode: bool = False,
    fast_speed: float = 1.15,
) -> tuple[np.ndarray, dict]:
    timeline = np.zeros(0, dtype=np.float32)
    sped_up_segments = 0
    max_speed = 1.0
    configured_speed = max(1.0, float(fast_speed))
    for seg in segments:
        wave = np.asarray(
            engine.synthesize_waveform(text=seg.text, voice=voice, ref_audio=ref_audio),
            dtype=np.float32,
        ).reshape(-1)
        if wave.size == 0:
            continue

        if fast_mode and configured_speed > 1.0:
            wave = speed_up_waveform(wave, configured_speed)
            sped_up_segments += 1
            max_speed = max(max_speed, configured_speed)

        start_sample = max(0, int(round(seg.start_ms * engine.sample_rate / 1000.0)))
        if timeline.size < start_sample:
            timeline = np.concatenate([timeline, np.zeros(start_sample - timeline.size, dtype=np.float32)])

        end_sample = start_sample + wave.size
        if timeline.size < end_sample:
            timeline = np.concatenate([timeline, np.zeros(end_sample - timeline.size, dtype=np.float32)])

        timeline[start_sample:end_sample] += wave

    if timeline.size:
        peak = float(np.max(np.abs(timeline)))
        if peak > 1.0:
            timeline = timeline / peak
    return timeline.astype(np.float32), {
        "fast_mode": bool(fast_mode),
        "fast_speed": round(configured_speed, 3),
        "sped_up_segments": sped_up_segments,
        "max_speed_factor": round(max_speed, 3),
    }


@app.post("/synthesize")
def synthesize() -> tuple:
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    voice = payload.get("voice")
    ref_audio = payload.get("ref_audio")

    if not text:
        return jsonify({"error": "text is required"}), 400

    stamp = _make_timestamp()
    output_path = Path(config.get("output_dir", "outputs")) / f"api_{stamp}.wav"
    produced = engine.synthesize_to_file(text=text, output_path=str(output_path), voice=voice, ref_audio=ref_audio)

    return jsonify({"status": "ok", "output_path": produced}), 200


@app.post("/synthesize_batch")
def synthesize_batch() -> tuple:
    payload = request.get_json(silent=True) or {}
    texts = payload.get("texts") or []
    voice = payload.get("voice")
    ref_audio = payload.get("ref_audio")

    if not isinstance(texts, list) or not texts:
        return jsonify({"error": "texts must be a non-empty list"}), 400

    cleaned = [str(item).strip() for item in texts if str(item).strip()]
    if not cleaned:
        return jsonify({"error": "texts must contain at least one non-empty item"}), 400

    stamp = _make_timestamp()
    out_dir = Path(config.get("output_dir", "outputs")) / f"batch_{stamp}"
    produced = engine.synthesize_batch_to_files(
        cleaned,
        output_dir=str(out_dir),
        voice=voice,
        ref_audio=ref_audio,
        prefix="api_batch",
    )
    return jsonify({"status": "ok", "count": len(produced), "output_paths": produced}), 200


@app.post("/synthesize_stream")
def synthesize_stream() -> Response:
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    voice = payload.get("voice")
    ref_audio = payload.get("ref_audio")
    if not text:
        return Response("text is required", status=400, mimetype="text/plain")

    waveform = engine.synthesize_waveform(text=text, voice=voice, ref_audio=ref_audio)
    buffer = io.BytesIO()
    sf.write(buffer, waveform, engine.sample_rate, format="WAV")
    wav_bytes = buffer.getvalue()

    def _stream() -> bytes:
        chunk_size = 4096
        for idx in range(0, len(wav_bytes), chunk_size):
            yield wav_bytes[idx : idx + chunk_size]

    return Response(_stream(), mimetype="audio/wav")


@app.post("/synthesize_clone")
def synthesize_clone() -> tuple:
    text = str(request.form.get("text", "")).strip()
    voice = request.form.get("voice")
    uploaded = request.files.get("ref_audio")

    if not text:
        return jsonify({"error": "text is required"}), 400
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "ref_audio file is required"}), 400

    stamp = _make_timestamp()
    clone_dir = Path(config.get("output_dir", "outputs")) / "clones"
    clone_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(uploaded.filename).suffix or ".wav"
    ref_path = clone_dir / f"ref_{stamp}{suffix}"
    uploaded.save(ref_path)

    out_path = clone_dir / f"clone_{stamp}.wav"
    produced = engine.synthesize_to_file(text=text, output_path=str(out_path), voice=voice, ref_audio=str(ref_path))
    return jsonify({"status": "ok", "output_path": produced, "ref_audio_path": str(ref_path)}), 200


@app.post("/synthesize_srt")
def synthesize_srt() -> tuple:
    payload = request.get_json(silent=True) or {}
    raw_text = str(payload.get("srt_text", "")).strip()
    voice = payload.get("voice")
    ref_audio = payload.get("ref_audio")
    fast_mode = _parse_bool(payload.get("fast_mode", False))
    fast_speed = float(payload.get("fast_speed", 1.15))

    if not raw_text:
        return jsonify({"error": "srt_text is required"}), 400

    segments = parse_srt_text(raw_text)
    if not segments:
        return jsonify({"error": "No valid subtitle segments found in SRT"}), 400

    waveform, render_meta = _synthesize_srt_segments(
        segments,
        voice=voice,
        ref_audio=ref_audio,
        fast_mode=fast_mode,
        fast_speed=fast_speed,
    )
    if waveform.size == 0:
        return jsonify({"error": "Could not render audio from SRT segments"}), 400

    stamp = _make_timestamp()
    output_path = Path(config.get("output_dir", "outputs")) / f"srt_{stamp}.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, waveform, engine.sample_rate)

    return (
        jsonify(
            {
                "status": "ok",
                "output_path": str(output_path),
                "segment_count": len(segments),
                "duration_sec": round(float(waveform.size) / float(engine.sample_rate), 3),
                **render_meta,
            }
        ),
        200,
    )


@app.post("/synthesize_srt_file")
def synthesize_srt_file() -> tuple:
    voice = request.form.get("voice")
    ref_audio = request.form.get("ref_audio")
    uploaded = request.files.get("srt_file")
    fast_mode = _parse_bool(request.form.get("fast_mode", False))
    fast_speed = float(request.form.get("fast_speed", 1.15))

    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "srt_file is required"}), 400
    if not uploaded.filename.lower().endswith(".srt"):
        return jsonify({"error": "srt_file must have .srt extension"}), 400

    raw_text = decode_srt_bytes(uploaded.read()).strip()
    if not raw_text:
        return jsonify({"error": "srt_file is empty"}), 400

    segments = parse_srt_text(raw_text)
    if not segments:
        return jsonify({"error": "No valid subtitle segments found in SRT"}), 400

    waveform, render_meta = _synthesize_srt_segments(
        segments,
        voice=voice,
        ref_audio=ref_audio,
        fast_mode=fast_mode,
        fast_speed=fast_speed,
    )
    if waveform.size == 0:
        return jsonify({"error": "Could not render audio from SRT segments"}), 400

    stamp = _make_timestamp()
    output_path = Path(config.get("output_dir", "outputs")) / f"srt_file_{stamp}.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, waveform, engine.sample_rate)

    return (
        jsonify(
            {
                "status": "ok",
                "output_path": str(output_path),
                "segment_count": len(segments),
                "duration_sec": round(float(waveform.size) / float(engine.sample_rate), 3),
                **render_meta,
            }
        ),
        200,
    )


@app.post("/warmup")
def warmup() -> tuple:
    stats = engine.warmup()
    return jsonify({"status": "ok", "cache": stats}), 200


@app.get("/health")
def health() -> tuple:
    return (
        jsonify(
            {
                "status": "ok",
                "engine_mode": config.get("engine_mode", "standard"),
                "model_name": engine.model_name,
                "device": engine.device,
                "engine_class": engine.__class__.__name__,
                "cache": engine.cache_stats(),
            }
        ),
        200,
    )


@app.get("/voices")
def voices() -> tuple:
    return jsonify({"voices": available_voices}), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
