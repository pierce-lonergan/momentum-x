# ╔══════════════════════════════════════════════════════════════╗
# ║              MOMENTUM-X Docker Image                        ║
# ║  Multi-stage build: build deps → slim runtime               ║
# ╚══════════════════════════════════════════════════════════════╝

FROM python:3.12-slim AS base

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Stage 1: Install dependencies ──────────────────────────
FROM base AS deps

# Install build essentials for any compiled deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir ".[dev]"

# ── Stage 2: Runtime ───────────────────────────────────────
FROM base AS runtime

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Non-root user for security
RUN useradd --create-home --shell /bin/bash trader
USER trader

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "from config.settings import Settings; Settings()" || exit 1

# Default: paper trading mode
ENTRYPOINT ["python", "-m", "main"]
CMD ["paper"]
