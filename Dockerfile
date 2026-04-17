FROM python:3.12-slim

WORKDIR /app

# Build timestamp: 2026-04-17_22-23 (force rebuild)
ENV BUILD_TIME="2026-04-17_22:23"
RUN echo "Cache invalidation: $(date)" && rm -rf /usr/local/lib/python3.12/dist-packages/*

# Install dependencies (using deps.txt to bypass Railway caching)
COPY deps.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r deps.txt

# Copy application code
COPY . .

# Set working directory to backend
WORKDIR /app/backend

# Start command (run migrations first, use PORT env var for Railway compatibility)
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
