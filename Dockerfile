FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from backend/pyproject.toml (single source of truth)
COPY backend/pyproject.toml ./backend/
WORKDIR /app/backend
RUN pip install --upgrade pip && pip install --no-cache-dir -e .

# Copy application code
WORKDIR /app
COPY . .

# Set working directory to backend for runtime
WORKDIR /app/backend

# Start command (run migrations first, use PORT env var for Railway compatibility)
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
