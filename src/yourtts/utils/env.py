from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str = ".env") -> None:
    source = Path(path)
    if not source.exists():
        return

    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip("'").strip('"')
        if not key:
            continue
        os.environ.setdefault(key, value)
