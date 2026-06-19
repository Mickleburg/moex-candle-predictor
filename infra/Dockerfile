# Reproducible Linux image for the MOEX autonomous agent (scheduler daemon).
# Build context is the repo root:  docker build -f infra/Dockerfile -t moex-agent .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Europe/Moscow

# tzdata so APScheduler's Europe/Moscow cron fires at the right wall-clock time.
RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && ln -fs /usr/share/zoneinfo/Europe/Moscow /etc/localtime \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# App code. data/ (state, logs, parquet) is a mounted volume in compose, not baked in.
COPY . .

# Non-root runtime user; it must own the mounted data dir.
RUN useradd --create-home --uid 10001 agent \
    && mkdir -p /app/data && chown -R agent:agent /app
USER agent

# Default: the long-lived scheduler (EOD + pre-open + dead-man's-switch). TZ Europe/Moscow.
ENTRYPOINT ["python", "-m", "agent.src.cli"]
CMD ["scheduler"]
