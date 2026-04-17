FROM python:3.12-slim

WORKDIR /app

# Build timestamp: 2026-04-17_22-21
ENV BUILD_TIME="2026-04-17_22:21"

# Install dependencies (using deps.txt to bypass Railway caching)
COPY deps.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r deps.txt

# Copy application code
COPY . .

# Set working directory to backend
WORKDIR /app/backend

# Start command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
