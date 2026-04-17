# Getting Started Guide

This guide walks you through setting up News Agent for local development or self-hosted deployment.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Backend Configuration](#backend-configuration)
4. [Frontend Configuration](#frontend-configuration)
5. [Running Services](#running-services)
6. [First Time Setup](#first-time-setup)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required
- Python 3.11 or higher (`python --version`)
- Node.js 18 or higher (`node --version`)
- Git (`git --version`)

### Optional (for production/full features)
- PostgreSQL 14+ (for persistent database)
- Redis 6+ (for caching and Celery broker)
- Docker & Docker Compose (for containerized deployment)

### API Keys Required
- **Groq API Key**: Get from https://console.groq.com
- **SendGrid API Key**: Get from https://sendgrid.com (free tier available)

---

## Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/yomaxoy/News-Agent.git
cd News-Agent
```

### 2. Create Python Virtual Environment

**Linux/macOS:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

Verify:
```bash
python --version  # Should be 3.11+
which python      # Should point to venv/bin/python
```

---

## Backend Configuration

### Step 1: Install Dependencies
```bash
cd backend
pip install -e .
```

Verify installation:
```bash
python -c "import fastapi; print(fastapi.__version__)"
```

### Step 2: Create Environment File
```bash
cp .env.example .env
```

Edit `.env` with your settings:
```env
# Database (SQLite for development, PostgreSQL for production)
DATABASE_URL=sqlite:///./news_agent.db
# Or for PostgreSQL:
# DATABASE_URL=postgresql://user:password@localhost/newsagent

# JWT Secret (generate with: python -c "import secrets; print(secrets.token_urlsafe(32))")
SECRET_KEY=your-random-secret-key-here
ALGORITHM=HS256

# Groq API (required for digest summarization)
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# SendGrid API (required for email delivery)
SENDGRID_API_KEY=SG.your_sendgrid_api_key_here
SENDGRID_FROM_EMAIL=noreply@yourdomain.com

# Redis (optional, for caching and Celery)
REDIS_URL=redis://localhost:6379

# Environment
ENVIRONMENT=development
```

### Step 3: Initialize Database
```bash
# SQLite (default for development)
python -c "from app.db.database import Base, engine; Base.metadata.create_all(bind=engine)"

# PostgreSQL (if using)
# 1. Create database: createdb newsagent
# 2. Run migrations: alembic upgrade head
```

### Step 4: Run Backend
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

Visit http://localhost:8000/docs for interactive API documentation.

---

## Frontend Configuration

### Step 1: Install Dependencies
```bash
cd ../frontend
npm install
```

Verify:
```bash
npm list react  # Should show React version
```

### Step 2: Create Environment File
```bash
cp .env.local.example .env.local
```

Edit `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-random-secret-key-generate-with-openssl
```

Generate a secure secret:
```bash
openssl rand -base64 32
```

### Step 3: Run Frontend
```bash
npm run dev
```

You should see:
```
> ready - started server on 0.0.0.0:3000, url: http://localhost:3000
```

Visit http://localhost:3000 in your browser.

---

## Running Services

### Option 1: Multiple Terminals (Recommended for Development)

**Terminal 1 - Backend API:**
```bash
cd backend
source venv/bin/activate  # or: venv\Scripts\activate on Windows
python -m uvicorn app.main:app --reload --port 8000
```

**Terminal 2 - Frontend Dev Server:**
```bash
cd frontend
npm run dev
```

**Terminal 3 - Celery Worker (Optional, for scheduled tasks):**
```bash
cd backend
source venv/bin/activate
celery -A app.tasks worker --loglevel=info
```

**Terminal 4 - Celery Beat (Optional, for scheduler):**
```bash
cd backend
source venv/bin/activate
celery -A app.tasks beat --loglevel=info
```

### Option 2: Docker Compose (All Services)
```bash
docker-compose up
```

This starts:
- Backend API (port 8000)
- Frontend (port 3000)
- PostgreSQL (port 5432)
- Redis (port 6379)

---

## First Time Setup

### 1. Create User Account
Visit http://localhost:3000/auth/register

Fill in:
- Email: `user@example.com`
- Password: `SecurePassword123` (must have uppercase + number)
- Confirm Password

### 2. Verify Email (In Development)
In development mode, email verification is automatic. In production, check your email.

### 3. Login
Go to http://localhost:3000/auth/login with your credentials.

### 4. Add Your First Source
1. Go to Dashboard → Sources
2. Click "+ Add Source"
3. Fill in:
   - Name: `BBC News`
   - URL: `https://feeds.bbc.co.uk/news/rss.xml`
   - Category: `News`
4. Click "Test Feed" to verify
5. Click "Save Source"

### 5. Create Your First Schedule
1. Go to Dashboard → Schedules
2. Click "+ Create Schedule"
3. Fill in:
   - Name: `Morning Digest`
   - When to run: `Every Day at 6 AM` (or custom cron)
   - Max Articles: `7`
   - Select sources: Check "BBC News"
4. Click "Save Schedule"

### 6. Setup Delivery Channel
1. In schedule list, click "Edit" on your schedule
2. Go to Delivery section
3. Add Discord webhook OR Email:
   - **Discord**: Get webhook URL from server settings
   - **Email**: Enter recipient email
4. Save

### 7. Test Digest Generation (Manual)
Backend endpoint to trigger manually:
```bash
curl -X POST http://localhost:8000/api/digests/generate \
  -H "Authorization: Bearer <your-token>" \
  -H "Content-Type: application/json" \
  -d '{"schedule_id": 1}'
```

---

## Running Tests

### Backend Tests
```bash
cd backend

# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_auth.py -v

# Run specific test
pytest tests/test_auth.py::TestRegister::test_register_success -v
```

**Current Status**: 75 tests passing, 83% code coverage ✅

### Frontend Tests (Coming Soon)
```bash
cd frontend
npm test
```

---

## Development Workflow

### Making Changes
1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes in editor
3. Run tests: `pytest tests/ -v`
4. Format code: `black app/` (optional)
5. Lint code: `ruff check app/` (optional)

### Committing
```bash
git add .
git commit -m "feat: add new feature description"
git push origin feature/my-feature
```

Then create a pull request on GitHub.

---

## Troubleshooting

### Backend Issues

**"ModuleNotFoundError: No module named 'fastapi'"**
```bash
# Activate virtual environment
source venv/bin/activate
# Then install
pip install -e .
```

**"postgresql error: could not connect to server"**
```bash
# Use SQLite instead (default)
# Edit .env: DATABASE_URL=sqlite:///./news_agent.db
```

**"GROQ_API_KEY not set"**
```bash
# Get key from https://console.groq.com
# Add to .env: GROQ_API_KEY=gsk_...
# Restart backend
```

**"ConnectionRefusedError: Cannot connect to Redis"**
```bash
# Redis is optional. If not needed, remove from .env or set:
REDIS_URL=  # Leave empty to disable caching
```

### Frontend Issues

**"Module not found: next-auth"**
```bash
cd frontend
npm install next-auth
```

**"localhost:3000 refused to connect"**
- Check if frontend is running: `npm run dev`
- Check if port 3000 is available: `lsof -i :3000`
- Try different port: `PORT=3001 npm run dev`

**"API calls returning 401 Unauthorized"**
- Check .env.local has correct `NEXT_PUBLIC_API_URL`
- Check backend is running: `http://localhost:8000/docs`
- Check token is stored in cookies: Browser DevTools → Application → Cookies

### Common Issues

**"Port 8000 already in use"**
```bash
# Find what's using it
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
uvicorn app.main:app --port 8001
```

**"Database locked (SQLite)"**
```bash
# SQLite locks with multiple processes
# Use PostgreSQL for production/multiple workers
# Or ensure only one backend instance is running
```

---

## Next Steps

After setup is complete:

1. **Read Documentation**
   - [ARCHITECTURE.md](../ARCHITECTURE.md) - System design
   - [API Documentation](http://localhost:8000/docs) - Interactive API docs

2. **Explore Features**
   - Add multiple sources
   - Create different schedules
   - Test digest generation
   - Configure delivery channels

3. **For Production**
   - Use PostgreSQL instead of SQLite
   - Setup proper environment variables
   - Enable HTTPS
   - Configure proper CORS
   - Setup monitoring (Sentry)

4. **Contributing**
   - Fork repository
   - Make changes
   - Run tests
   - Submit pull request

---

## Support

Having issues? Check:
1. [GitHub Issues](https://github.com/yomaxoy/News-Agent/issues) - Known issues
2. [Discussions](https://github.com/yomaxoy/News-Agent/discussions) - Community help
3. [ARCHITECTURE.md](../ARCHITECTURE.md) - Technical details

---

**Last Updated**: April 2026
