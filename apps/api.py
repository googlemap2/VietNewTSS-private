from __future__ import annotations

from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request
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

app = Flask(__name__)


@app.post("/synthesize")
def synthesize() -> tuple:
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    voice = payload.get("voice")

    if not text:
        return jsonify({"error": "text is required"}), 400

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    output_path = Path(config.get("output_dir", "outputs")) / f"api_{stamp}.wav"
    produced = engine.synthesize_to_file(text=text, output_path=str(output_path), voice=voice)

    return jsonify({"status": "ok", "output_path": produced}), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
