from __future__ import annotations

from datetime import datetime
from pathlib import Path

from yourtts.utils.env import load_dotenv

load_dotenv()

import gradio as gr
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
default_temperature = float(config.get("temperature", 0.05))
default_top_k = int(config.get("top_k", 1))
configured_voice = str(config.get("voice", "default"))
default_voice = configured_voice if configured_voice in available_voices else (available_voices[0] if available_voices else "default")


def _runtime_options(temperature: float, top_k: int) -> dict:
    return {"temperature": float(temperature), "top_k": int(top_k)}


def _synthesize_srt_segments(
    segments,
    voice: str | None,
    temperature: float,
    top_k: int,
    fast_mode: bool = False,
    fast_speed: float = 1.15,
) -> tuple[np.ndarray, dict]:
    timeline = np.zeros(0, dtype=np.float32)
    runtime_options = _runtime_options(temperature, top_k)
    sped_up_segments = 0
    max_speed = 1.0
    configured_speed = max(1.0, float(fast_speed))
    for seg in segments:
        wave = np.asarray(
            engine.synthesize_waveform(text=seg.text, voice=voice, **runtime_options),
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


def synthesize(text: str, voice: str, temperature: float, top_k: int) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    output_path = Path(config.get("output_dir", "outputs")) / f"web_{stamp}.wav"
    return engine.synthesize_to_file(
        text=text,
        output_path=str(output_path),
        voice=voice,
        **_runtime_options(temperature, top_k),
    )


def synthesize_clone(text: str, voice: str, ref_audio: str | None, temperature: float, top_k: int) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    output_path = Path(config.get("output_dir", "outputs")) / f"clone_{stamp}.wav"
    return engine.synthesize_to_file(
        text=text,
        output_path=str(output_path),
        voice=voice,
        ref_audio=ref_audio,
        **_runtime_options(temperature, top_k),
    )


def synthesize_batch(text_blob: str, voice: str, temperature: float, top_k: int) -> list[str]:
    rows = [line.strip() for line in text_blob.splitlines() if line.strip()]
    if not rows:
        return []
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = Path(config.get("output_dir", "outputs")) / f"web_batch_{stamp}"
    return engine.synthesize_batch_to_files(
        rows,
        output_dir=str(out_dir),
        voice=voice,
        prefix="web_batch",
        **_runtime_options(temperature, top_k),
    )


def synthesize_srt_file(
    srt_file: str | None,
    voice: str,
    temperature: float,
    top_k: int,
    fast_mode: bool,
    fast_speed: float,
) -> tuple[str | None, dict]:
    if not srt_file:
        return None, {"status": "error", "detail": "Please upload an .srt file."}

    source = Path(srt_file)
    if not source.exists():
        return None, {"status": "error", "detail": "Uploaded SRT file was not found on disk."}

    raw_text = decode_srt_bytes(source.read_bytes()).strip()
    if not raw_text:
        return None, {"status": "error", "detail": "Uploaded SRT file is empty."}

    segments = parse_srt_text(raw_text)
    if not segments:
        return None, {"status": "error", "detail": "No valid subtitle segments found in SRT."}

    waveform, render_meta = _synthesize_srt_segments(
        segments,
        voice=voice,
        temperature=temperature,
        top_k=top_k,
        fast_mode=fast_mode,
        fast_speed=fast_speed,
    )
    if waveform.size == 0:
        return None, {"status": "error", "detail": "Could not render audio from SRT segments."}

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    output_path = Path(config.get("output_dir", "outputs")) / f"web_srt_{stamp}.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, waveform, engine.sample_rate)
    return str(output_path), {
        "status": "ok",
        "output_path": str(output_path),
        "segment_count": len(segments),
        "duration_sec": round(float(waveform.size) / float(engine.sample_rate), 3),
        **render_meta,
    }


def run_warmup() -> dict:
    return engine.warmup()


with gr.Blocks(title="yourtts") as demo:
    gr.Markdown("# yourtts\nSingle and batch synthesis demo.")
    with gr.Accordion("Emotion Settings", open=False):
        temperature_input = gr.Slider(
            label="Temperature (higher = more expressive, less stable)",
            minimum=0.01,
            maximum=1.0,
            step=0.05,
            value=default_temperature,
        )
        top_k_input = gr.Slider(
            label="Top-k (higher = more variation)",
            minimum=1,
            maximum=100,
            step=1,
            value=default_top_k,
        )

    with gr.Tab("Single"):
        text_input = gr.Textbox(label="Text", lines=4, placeholder="Type sentence to synthesize...")
        voice_input = gr.Dropdown(
            label="Voice",
            choices=available_voices,
            value=default_voice,
        )
        run_single = gr.Button("Synthesize")
        single_audio = gr.Audio(type="filepath", label="Generated Audio")
        run_single.click(fn=synthesize, inputs=[text_input, voice_input, temperature_input, top_k_input], outputs=single_audio)

    with gr.Tab("Batch"):
        batch_input = gr.Textbox(
            label="Batch Text (one line = one output)",
            lines=10,
            placeholder="Line 1\nLine 2\nLine 3",
        )
        batch_voice = gr.Dropdown(
            label="Voice",
            choices=available_voices,
            value=default_voice,
        )
        run_batch = gr.Button("Synthesize Batch")
        batch_files = gr.Files(label="Generated Files")
        run_batch.click(
            fn=synthesize_batch,
            inputs=[batch_input, batch_voice, temperature_input, top_k_input],
            outputs=batch_files,
        )

    with gr.Tab("Clone"):
        clone_text = gr.Textbox(label="Text", lines=4, placeholder="Text to synthesize with cloned voice")
        clone_voice = gr.Dropdown(
            label="Fallback Voice",
            choices=available_voices,
            value=default_voice,
        )
        clone_ref = gr.Audio(label="Reference Audio (3-10s)", type="filepath")
        run_clone = gr.Button("Synthesize Clone")
        clone_audio = gr.Audio(type="filepath", label="Cloned Output")
        run_clone.click(
            fn=synthesize_clone,
            inputs=[clone_text, clone_voice, clone_ref, temperature_input, top_k_input],
            outputs=clone_audio,
        )

    with gr.Tab("SRT"):
        srt_file = gr.File(label="Subtitle File (.srt)", file_types=[".srt"], type="filepath")
        srt_voice = gr.Dropdown(
            label="Voice",
            choices=available_voices,
            value=default_voice,
        )
        srt_fast_mode = gr.Checkbox(label="Fast Read Mode", value=False)
        srt_fast_speed = gr.Slider(label="Fast Speed", minimum=1.0, maximum=2.0, step=0.05, value=1.15)
        run_srt = gr.Button("Synthesize From SRT")
        srt_audio = gr.Audio(type="filepath", label="Timeline Audio")
        srt_meta = gr.JSON(label="SRT Result")
        run_srt.click(
            fn=synthesize_srt_file,
            inputs=[srt_file, srt_voice, temperature_input, top_k_input, srt_fast_mode, srt_fast_speed],
            outputs=[srt_audio, srt_meta],
        )

    with gr.Tab("Engine"):
        warmup_button = gr.Button("Warmup Engine")
        cache_stats = gr.JSON(label="Cache Stats")
        warmup_button.click(fn=run_warmup, inputs=None, outputs=cache_stats)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
