# News Agent Backend

Modern news aggregation platform backend built with FastAPI.

## Setup

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- PostgreSQL
- Redis

### Local Development

1. **Start Services**
```bash
docker-compose up -d
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup Environment**
```bash
cp .env.example .env.local
# Edit .env.local with your settings
export $(cat .env.local | xargs)
```

5. **Run Migrations**
```bash
alembic upgrade head
```

6. **Start Server**
```bash
uvicorn app.main:app --reload
```

Server will be available at http://localhost:8000

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Database Migrations

Create a new migration:
```bash
alembic revision --autogenerate -m "Description of changes"
```

Apply migrations:
```bash
alembic upgrade head
```

Downgrade migrations:
```bash
alembic downgrade -1
```

## Testing

Run tests:
```bash
pytest
```

With coverage:
```bash
pytest --cov=app
```

## Project Structure

```
backend/
├── app/
│   ├── api/              # API routes
│   ├── core/             # Core utilities (security, config)
│   ├── db/               # Database models and session
│   ├── integrations/     # External API integrations
│   ├── services/         # Business logic
│   ├── tasks/            # Celery tasks
│   ├── schemas/          # Pydantic models
│   ├── config.py         # Settings
│   └── main.py           # FastAPI app
├── migrations/           # Alembic migrations
├── tests/                # Test suite
├── requirements.txt      # Python dependencies
└── Dockerfile            # Docker image
```

## Environment Variables

See `.env.example` for all available settings.

## Deployment

Build Docker image:
```bash
docker build -t news-agent-backend:latest .
```

## License

MIT
