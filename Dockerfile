FROM python:3.10-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev

COPY . .

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app
WORKDIR /app

ENV PYTHONPATH=/app \
    SPEAKER_MODEL_PATH=/app/src/resoursces/models/iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k \
    INPUT_DATA_PATH=/app/src/resoursces/test/input_Data

EXPOSE 8017

CMD ["/app/.venv/bin/uvicorn", "src.v1.api:app", "--host", "0.0.0.0", "--port", "8017"]
