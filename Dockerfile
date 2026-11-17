# ---- build stage ----
FROM python:3.12-slim AS builder

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock* requirements.txt* ./

# Repository dependencies
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- runtime stage ----
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /install /usr/local

COPY . .

# No-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "genai_api.main:app", "--host", "0.0.0.0", "--port", "8000"]