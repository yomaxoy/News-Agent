# Deployment Guide

This guide explains how to deploy News Agent to production using Railway (backend) and Vercel (frontend).

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Backend Deployment (Railway)](#backend-deployment-railway)
3. [Frontend Deployment (Vercel)](#frontend-deployment-vercel)
4. [GitHub Actions CI/CD](#github-actions-cicd)
5. [Environment Variables](#environment-variables)
6. [Database Migrations](#database-migrations)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Accounts
- GitHub account (push code)
- Railway account (https://railway.app) - for backend
- Vercel account (https://vercel.com) - for frontend
- Groq account (https://console.groq.com) - for API key
- SendGrid account (https://sendgrid.com) - for email

### Required Tools
- Git
- GitHub CLI (optional): `gh auth login`
- Railway CLI (optional): `npm install -g @railway/cli`
- Vercel CLI (optional): `npm install -g vercel`

## Backend Deployment (Railway)

### Step 1: Create Railway Project

1. Go to https://railway.app/dashboard
2. Click "New Project"
3. Select "GitHub Repo"
4. Connect your GitHub account (if not already connected)
5. Select the `News-Agent` repository

### Step 2: Add PostgreSQL Database

In Railway dashboard:
1. Click "Add"
2. Select "PostgreSQL"
3. Railway automatically creates a database

### Step 3: Add Redis Cache (Optional but Recommended)

1. Click "Add"
2. Select "Redis"
3. Railway automatically configures it

### Step 4: Configure Backend Service

Railway should auto-detect the backend Dockerfile. Configure:

1. **Build Settings**
   - Dockerfile: `./backend/Dockerfile`
   - Build Command: (empty - Docker handles it)
   - Start Command: (empty - Dockerfile CMD)

2. **Environment Variables**
   - Click "Raw Editor" and add:
   ```
   DATABASE_URL=$DATABASE_URL
   REDIS_URL=$REDIS_URL
   GROQ_API_KEY=your-groq-key
   SENDGRID_API_KEY=your-sendgrid-key
   SECRET_KEY=<generate-secure-key>
   ENVIRONMENT=production
   ```

3. **Port Configuration**
   - Set to 8000 (default)

4. **Domain**
   - Click "Settings" → "Domains"
   - Add custom domain or use Railway subdomain

### Step 5: Deploy

Railway automatically deploys on every push to `main` branch.

Monitor deployment:
```bash
# Using Railway CLI
railway logs -f

# Or in dashboard → Logs tab
```

Verify deployment:
```bash
curl https://your-railway-backend.railway.app/api/health
# Should return: {"status": "ok", "environment": "production"}
```

---

## Frontend Deployment (Vercel)

### Step 1: Create Vercel Project

1. Go to https://vercel.com/dashboard
2. Click "Add New..."
3. Select "Project"
4. Import GitHub repository (`News-Agent`)
5. Select Framework: `Next.js`

### Step 2: Configure Build Settings

Vercel auto-detects Next.js settings:
- **Root Directory**: `frontend`
- **Build Command**: `npm run build`
- **Start Command**: `npm start`
- **Install Command**: `npm ci`

### Step 3: Set Environment Variables

In Vercel project settings:

1. Click "Settings" → "Environment Variables"
2. Add:
   ```
   NEXT_PUBLIC_API_URL=https://your-railway-backend.railway.app
   NEXTAUTH_URL=https://your-vercel-frontend.vercel.app
   NEXTAUTH_SECRET=<generate-secure-key>
   ```

3. Click "Save"

### Step 4: Deploy

Option A: Automatic (Recommended)
- Vercel auto-deploys on every push to `main`

Option B: Manual
```bash
vercel --prod
```

### Step 5: Verify

Visit your Vercel domain and test:
- Register new account
- Add RSS source
- Create schedule
- Verify API connectivity

---

## GitHub Actions CI/CD

### Setup Secrets

In GitHub repository settings (`Settings` → `Secrets and variables` → `Actions`):

Add secrets:
```
DOCKER_USERNAME=your-docker-hub-username
DOCKER_PASSWORD=your-docker-hub-password
RAILWAY_API_TOKEN=your-railway-token
VERCEL_TOKEN=your-vercel-token
VERCEL_ORG_ID=your-vercel-org-id
VERCEL_PROJECT_ID=your-vercel-project-id
API_URL=https://your-railway-backend.railway.app
```

**Get these values:**

- **Docker Hub**: Create account at https://hub.docker.com
- **Railway Token**: In Railway → Account Settings → API Tokens
- **Vercel Token**: In Vercel → Account Settings → Tokens
- **Vercel Project ID**: In Vercel project → Settings → General

### How CI/CD Works

On every push to `main`:

1. **Tests** (always runs)
   - Backend: `pytest tests/ --cov=app`
   - Frontend: `npm run build`

2. **Build & Push** (if tests pass)
   - Build Docker image for backend
   - Push to Docker Hub

3. **Deploy** (if builds succeed)
   - Railway auto-deploys from Docker Hub
   - Vercel auto-deploys from GitHub

---

## Environment Variables

### Backend Production (.env)

```env
# Database
DATABASE_URL=postgresql://user:password@host/newsagent

# Security
SECRET_KEY=<generate-with-openssl-rand-base64-32>
ALGORITHM=HS256

# APIs
GROQ_API_KEY=gsk_...
SENDGRID_API_KEY=SG....

# Cache
REDIS_URL=redis://host:6379

# Environment
ENVIRONMENT=production
LOG_LEVEL=info
```

### Frontend Production (.env.production)

```env
NEXT_PUBLIC_API_URL=https://api.newsagent.io
NEXTAUTH_URL=https://newsagent.io
NEXTAUTH_SECRET=<generate-with-openssl>
```

Generate secrets:
```bash
# OpenSSL
openssl rand -base64 32

# Python
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## Database Migrations

### Initial Setup

Railway auto-creates database, but you need to initialize schema:

```bash
# Option 1: Via Railway Dashboard
# SSH into container and run:
python -c "from app.db.database import Base, engine; Base.metadata.create_all(bind=engine)"

# Option 2: Using Alembic
# In Railway container:
alembic upgrade head
```

### After Schema Changes

1. Create migration:
   ```bash
   alembic revision --autogenerate -m "Description of changes"
   ```

2. Review generated migration file

3. Push to GitHub
   - GitHub Actions will apply migration on deployment

---

## Monitoring

### Health Checks

Railway and Vercel have built-in health checks. Configure:

**Backend:**
- Endpoint: `/api/health`
- Interval: 30s
- Timeout: 5s

**Frontend:**
- Endpoint: `/api/auth/signin`
- Interval: 30s
- Timeout: 5s

### Logs

**Backend (Railway):**
```bash
railway logs -f
```

Or via Railway dashboard:
- Project → Logs tab

**Frontend (Vercel):**
- In Vercel dashboard → Deployments → Logs

### Error Tracking (Future)

Setup Sentry for error monitoring:

```python
import sentry_sdk
sentry_sdk.init(
    dsn="your-sentry-dsn",
    environment="production",
)
```

---

## Troubleshooting

### Backend Issues

**502 Bad Gateway**
```bash
# Check logs
railway logs -f

# Common causes:
# 1. DATABASE_URL not set
# 2. GROQ_API_KEY invalid
# 3. Application crash during startup
```

**Database connection failed**
```bash
# Verify DATABASE_URL
echo $DATABASE_URL

# Test connection
psql $DATABASE_URL -c "SELECT 1"

# Check PostgreSQL is running
railway services list
```

**API returns 500**
```bash
# Check backend logs
railway logs -f

# Verify environment variables
echo $SECRET_KEY
echo $GROQ_API_KEY
```

### Frontend Issues

**API calls fail with 401**
- Check `NEXT_PUBLIC_API_URL` is correct
- Verify backend is running and accessible
- Check CORS configuration

**NextAuth errors**
- Verify `NEXTAUTH_SECRET` is set
- Check `NEXTAUTH_URL` matches deployment domain
- Clear browser cookies and try again

**Build fails in Vercel**
```bash
# Check build logs in Vercel dashboard
# Common issues:
# 1. Missing environment variables
# 2. Type errors (check tsconfig.json)
# 3. Module not found (check imports)

# Test locally:
cd frontend
npm run build
```

### Deployment Issues

**GitHub Actions fails**
1. Check Secrets are set correctly
2. View workflow run logs in GitHub
3. Verify Docker Hub credentials
4. Check Railway/Vercel tokens are valid

**Auto-deploy not working**
- Verify webhook is set up
- Check branch is `main`
- View deployment logs in Railway/Vercel

---

## Performance Optimization

### Backend
- Enable Redis caching
- Use connection pooling
- Implement rate limiting
- Monitor slow queries

### Frontend
- Enable ISR (Incremental Static Regeneration)
- Optimize images
- Lazy load components
- Enable Vercel Analytics

---

## Security Checklist

- [x] All secrets in GitHub Secrets (not .env)
- [x] HTTPS enforced
- [x] CORS restricted to frontend origin
- [x] Database has strong password
- [x] API keys rotated regularly
- [x] Logs don't contain sensitive data
- [x] Database backups enabled
- [ ] DDoS protection (Vercel, Railway)
- [ ] SSL certificate auto-renewal (automatic)

---

## Next Steps

1. **Test deployment thoroughly**
   - Create account
   - Add sources
   - Generate digests
   - Test delivery channels

2. **Setup monitoring**
   - Configure Sentry for error tracking
   - Setup uptime monitoring
   - Enable analytics

3. **Prepare for launch**
   - Write announcement post
   - Prepare marketing materials
   - Setup email list for beta users

---

**Last Updated**: April 2026
