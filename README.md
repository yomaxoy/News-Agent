# News Agent 📰

A modern web-scraping news aggregation platform that automatically fetches, summarizes, and distributes personalized news digests via Discord, Email, and more.

## Features ✨

- **Multi-Source Aggregation**: Add unlimited RSS feeds and sources
- **AI-Powered Summaries**: Groq API integration for intelligent article summarization
- **Flexible Scheduling**: Cron-based digest generation (daily, weekly, custom)
- **Multi-Channel Delivery**:
  - 💬 Discord webhooks
  - 📧 Email via SendGrid
  - (Coming soon) Slack, Telegram, Podcast
- **Smart Deduplication**: Automatic duplicate article detection
- **User Isolation**: Secure multi-tenant architecture
- **Dashboard UI**: Modern Next.js dashboard for managing sources and schedules

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL 14+
- Redis 6+ (optional, for caching)

### Setup

1. **Clone repository**
   ```bash
   git clone https://github.com/yomaxoy/News-Agent.git
   cd News-Agent
   ```

2. **Setup Backend**
   ```bash
   cd backend
   
   # Create .env file
   cp .env.example .env
   # Edit .env with your settings (Groq API key, SendGrid token, etc)
   
   # Install dependencies
   pip install -e .
   
   # Create database
   python -c "from app.db.database import Base, engine; Base.metadata.create_all(bind=engine)"
   
   # Run migrations (if using Alembic)
   alembic upgrade head
   ```

3. **Setup Frontend**
   ```bash
   cd ../frontend
   
   # Copy environment template
   cp .env.local.example .env.local
   # Edit .env.local with your API URL
   
   # Install dependencies
   npm install
   ```

4. **Start Services**
   
   Terminal 1 - Backend:
   ```bash
   cd backend
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   
   Terminal 2 - Frontend:
   ```bash
   cd frontend
   npm run dev
   ```
   
   Terminal 3 - Celery Worker (for scheduled tasks):
   ```bash
   cd backend
   celery -A app.tasks worker --loglevel=info
   ```

5. **Access the app**
   - Frontend: http://localhost:3000
   - API Docs: http://localhost:8000/docs

## Configuration

### Environment Variables

**Backend** (`.env`)
```env
# Database
DATABASE_URL=postgresql://user:password@localhost/newsagent

# JWT Security
SECRET_KEY=your-secret-key-change-in-production
ALGORITHM=HS256

# Groq API (for summarization)
GROQ_API_KEY=your-groq-api-key

# SendGrid (for email delivery)
SENDGRID_API_KEY=your-sendgrid-api-key
SENDGRID_FROM_EMAIL=noreply@newsagent.io

# Redis Cache
REDIS_URL=redis://localhost:6379

# Environment
ENVIRONMENT=development
```

**Frontend** (`.env.local`)
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-secret-key-generate-with-openssl
```

## Project Structure

```
News-Agent/
├── backend/
│   ├── app/
│   │   ├── api/v1/              # REST API endpoints
│   │   ├── core/                # Security, config
│   │   ├── db/                  # Database models
│   │   ├── integrations/        # External API clients (Groq, etc)
│   │   ├── schemas/             # Pydantic request/response models
│   │   ├── services/            # Business logic
│   │   ├── tasks/               # Celery tasks
│   │   └── main.py              # FastAPI app
│   ├── tests/                   # Unit & integration tests
│   ├── migrations/              # Alembic DB migrations
│   └── pyproject.toml           # Dependencies
│
├── frontend/
│   ├── app/
│   │   ├── api/                 # NextAuth API routes
│   │   ├── auth/                # Login, register pages
│   │   ├── dashboard/           # Protected dashboard
│   │   │   ├── sources/         # Source management
│   │   │   ├── schedules/       # Schedule management
│   │   │   ├── digests/         # Digest preview
│   │   │   └── settings/        # User settings
│   │   └── layout.tsx           # Root layout
│   ├── components/              # Reusable UI components
│   ├── lib/                     # API client, schemas
│   ├── middleware.ts            # Protected routes
│   └── package.json
│
└── docs/
    ├── ARCHITECTURE.md          # System design
    └── GETTING_STARTED.md       # Installation guide
```

## API Examples

### Register
```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "SecurePass123"}'
```

### Login
```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "SecurePass123"}'
# Returns: {"id": 1, "email": "...", "token": "eyJ..."}
```

### Create Source
```bash
curl -X POST http://localhost:8000/api/sources \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "BBC News",
    "url": "https://feeds.bbc.co.uk/news/rss.xml",
    "category": "News"
  }'
```

### Create Schedule
```bash
curl -X POST http://localhost:8000/api/schedules \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Daily Digest",
    "cron_expression": "0 6 * * *",
    "timezone": "UTC",
    "source_ids": [1, 2, 3],
    "max_articles": 7
  }'
```

## Testing

### Run Backend Tests
```bash
cd backend
pytest tests/ -v                          # Run all tests
pytest tests/test_auth.py -v              # Run specific test file
pytest --cov=app --cov-report=html tests/ # Generate coverage report
```

**Current Coverage**: 83% (75 tests passing)

### Run Frontend Tests (Coming Soon)
```bash
cd frontend
npm test
```

## Deployment

### Docker
```bash
# Build
docker build -t news-agent-backend ./backend
docker build -t news-agent-frontend ./frontend

# Run
docker run -p 8000:8000 news-agent-backend
docker run -p 3000:3000 news-agent-frontend
```

### Cloud Deployment (Railway/Render)
See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed instructions.

## Architecture

![Architecture Diagram](docs/architecture.svg)

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design, database schema, and technical decisions.

## Roadmap

### MVP (GATE 1-6) ✅
- [x] Authentication system
- [x] RSS source management
- [x] Digest generation with Groq
- [x] Cron-based scheduling
- [x] Discord & Email delivery
- [x] Next.js dashboard UI

### Phase 2 (GATE 7-9) 🚧
- [x] Frontend source/schedule management
- [x] Digest preview & history
- [x] User settings page
- [ ] GitHub OAuth
- [ ] Slack integration
- [ ] Advanced analytics

### Phase 3 (GATE 10-12) 📋
- [ ] Podcast generation (Google Cloud TTS)
- [ ] Newsletter HTML templates
- [ ] Web scraping support
- [ ] Premium tier

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch: `git checkout -b feature/name`
3. Write tests for your changes
4. Submit pull request with description

## License

MIT License - See LICENSE file for details

## Support

- 📧 Email: support@newsagent.io
- 🐛 Issues: GitHub Issues
- 💬 Discussions: GitHub Discussions

## Acknowledgments

- **Groq** for fast LLM API
- **SendGrid** for email delivery
- **FastAPI** and **Next.js** communities
- **Open source** projects: feedparser, croniter, SQLAlchemy, etc

---

Made with ❤️ for the news aggregation community
