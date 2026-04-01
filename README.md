# yourtts

Personal TTS project inspired by VieNeu-style architecture.

## Features (Current)

- Standard engine synthesis to WAV
- Preset voices from `assets/voices.json`
- Text chunking + crossfade audio join
- Gradio web UI
- Flask API with `/synthesize` and `/voices`
- Batch synthesis (`/synthesize_batch`)
- Warmup + cache stats (`/warmup`, `/health`)
- WAV streaming response (`/synthesize_stream`)

## Quick start

1. Create and activate a virtual environment.
```bash
python -m venv .venv
```
```bash
.\.venv\Scripts\Activate.ps1
```
```bash
deactivate
```

2. Install package in editable mode:

```bash
python -m pip install -e .[dev]
```

Recommended for VieNeu-style quality (Turbo GGUF):

```bash
python -m pip install -e .[vieneu]
```

Then set:
- `engine_mode: turbo`
- `model_name: pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf`

3. Run smoke path:

```bash
python -m yourtts.smoke
```

4. Run API:

```bash
python apps/api.py
```

5. Run web UI:

```bash
python apps/web_ui.py
```

## API quick check

Get voices:

```bash
curl http://127.0.0.1:8000/voices
```

Warmup:

```bash
curl -X POST http://127.0.0.1:8000/warmup
```

Health:

```bash
curl http://127.0.0.1:8000/health
```

`/health` returns `engine_class` so you can verify runtime engine (for target setup it should be `VieneuTurboEngine`).

Synthesize:

```bash
curl -X POST http://127.0.0.1:8000/synthesize -H "Content-Type: application/json" -d "{\"text\":\"Xin chao\",\"voice\":\"warm\"}"
```

Batch synthesize:

```bash
curl -X POST http://127.0.0.1:8000/synthesize_batch -H "Content-Type: application/json" -d "{\"texts\":[\"xin chao\",\"ban khoe khong\"],\"voice\":\"default\"}"
```

Clone with reference audio (multipart):

```bash
curl -X POST http://127.0.0.1:8000/synthesize_clone -F "text=Xin chao tu giong clone" -F "voice=default" -F "ref_audio=@examples/ref.wav"
```

## Docker run path

Build image:

```bash
docker build -t yourtts:dev .
```

Run API container:

```bash
docker run --rm -p 8000:8000 yourtts:dev python apps/api.py
```

Run Web UI container:

```bash
docker run --rm -p 7860:7860 yourtts:dev python apps/web_ui.py
```
