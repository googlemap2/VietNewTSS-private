from __future__ import annotations

from pathlib import Path

import yaml

from yourtts.factory import create_engine
from yourtts.utils.env import load_dotenv


def main() -> None:
    load_dotenv()
    config_path = Path("config.yaml")
    config = {}
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    mode = config.get("engine_mode", "standard")
    sample_rate = int(config.get("sample_rate", 22050))
    output_dir = config.get("output_dir", "outputs")
    voice = config.get("voice", "default")

    engine = create_engine(
        mode=mode,
        sample_rate=sample_rate,
        output_dir=output_dir,
        voice=voice,
        device=config.get("device", "cpu"),
        model_name=config.get("model_name", "pnnbao-ump/VieNeu-TTS-0.3B-q8-gguf"),
        cache_size=int(config.get("cache_size", 128)),
    )
    out_path = str(Path(output_dir) / "smoke.wav")
    smoke_text = (
        "Hello from yourtts smoke path. "
        "This sentence is intentionally longer so phase two text chunking and audio join are exercised."
    )
    produced = engine.synthesize_to_file(smoke_text, output_path=out_path, voice=voice)
    print(f"Smoke complete. Generated: {produced}")


if __name__ == "__main__":
    main()
