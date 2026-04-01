from __future__ import annotations

from pathlib import Path

import yaml

from yourtts.factory import create_engine


def main() -> None:
    config_path = Path("config.yaml")
    config = {}
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    mode = config.get("engine_mode", "standard")
    sample_rate = int(config.get("sample_rate", 22050))
    output_dir = config.get("output_dir", "outputs")
    voice = config.get("voice", "default")

    engine = create_engine(mode=mode, sample_rate=sample_rate, output_dir=output_dir, voice=voice)
    out_path = str(Path(output_dir) / "smoke.wav")
    produced = engine.synthesize_to_file("Hello from yourtts smoke path.", output_path=out_path, voice=voice)
    print(f"Smoke complete. Generated: {produced}")


if __name__ == "__main__":
    main()
