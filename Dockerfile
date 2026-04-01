FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md config.yaml /app/
COPY src /app/src
COPY apps /app/apps
COPY assets /app/assets

RUN pip install --no-cache-dir -e .

EXPOSE 8000 7860

CMD ["python", "apps/api.py"]
