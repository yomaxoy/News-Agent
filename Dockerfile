FROM python:3.12-slim

WORKDIR /app

# Build timestamp for cache invalidation: 2026-04-17_22-14
ENV BUILD_TIME="2026-04-17_22:14"

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set working directory to backend
WORKDIR /app/backend

# Start command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
