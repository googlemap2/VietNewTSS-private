# yourtts

Personal TTS project inspired by VieNeu-style architecture.

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
