from __future__ import annotations

from datetime import datetime
from pathlib import Path

import gradio as gr
import yaml

from yourtts.factory import create_engine


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
)


def synthesize(text: str, voice: str) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    output_path = Path(config.get("output_dir", "outputs")) / f"web_{stamp}.wav"
    return engine.synthesize_to_file(text=text, output_path=str(output_path), voice=voice)


demo = gr.Interface(
    fn=synthesize,
    inputs=[
        gr.Textbox(label="Text", lines=4, placeholder="Type sentence to synthesize..."),
        gr.Dropdown(label="Voice", choices=["default"], value="default"),
    ],
    outputs=gr.Audio(type="filepath", label="Generated Audio"),
    title="yourtts MVP",
    description="Phase 1 demo UI for local synthesis",
)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
