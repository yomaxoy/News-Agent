# News Agent - Architecture Documentation

## Project Overview

News Agent is a modern web-scraping news aggregation platform that automatically fetches, summarizes, and distributes news digests via multiple channels (Discord, Email, Slack, Telegram).

## Technology Stack

### Backend
- **Framework**: FastAPI 0.104.1 (async REST API)
- **Language**: Python 3.11+
- **Database**: PostgreSQL with SQLAlchemy 2.0 ORM
- **Job Queue**: Celery 5.3 with Redis broker
- **Authentication**: JWT tokens with Argon2 password hashing
- **LLM Integration**: Groq API (Llama 3.3 70B) for digest summarization
- **Feed Parsing**: feedparser for RSS validation
- **Email**: SendGrid API for email delivery
- **Testing**: pytest with 83% code coverage

### Frontend
- **Framework**: Next.js 14 with TypeScript
- **Styling**: Tailwind CSS
- **Auth**: NextAuth.js with credentials provider
- **Forms**: React Hook Form + Zod for validation
- **API Client**: Axios with interceptors for JWT injection
- **Build**: Turbopack (Next.js 16)

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Browser                          │
│              (Next.js React Frontend)                    │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/REST API
                     │
┌────────────────────▼────────────────────────────────────┐
│                  FastAPI Backend                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  API Endpoints (Auth, Sources, Schedules, etc)   │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Services (Auth, Digest, Delivery, etc)          │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Database (PostgreSQL) & Cache (Redis)           │  │
│  └──────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
         │                        │
         ▼                        ▼
    ┌─────────┐            ┌────────────┐
    │ Celery  │            │ External   │
    │ Tasks   │            │ APIs:      │
    │         │            │ - Groq     │
    │ Worker  │            │ - SendGrid │
    │ Process │            │ - Discord  │
    └─────────┘            └────────────┘
```

## Database Schema

### Core Tables
- **users**: User accounts with email verification and password hashing
- **sources**: RSS feed URLs per user with categories
- **schedules**: Cron-based digest generation schedules per user
- **schedule_sources**: M2M relationship between schedules and sources
- **articles**: Parsed RSS articles with deduplication
- **digests**: Generated digest content with status tracking
- **delivery_channels**: Multi-channel delivery config (Discord, Email, etc)
- **jobs**: Celery task execution history and status

### Key Relationships
```
User (1) ──→ (M) Source
User (1) ──→ (M) Schedule
Schedule (1) ──→ (M) Source (via schedule_sources)
Schedule (1) ──→ (M) DeliveryChannel
Source (1) ──→ (M) Article
Schedule (1) ──→ (M) Digest
```

## API Endpoints

### Authentication (`/api/auth`)
- `POST /register` - Create new user account
- `POST /login` - Get JWT access token
- `POST /verify-email` - Verify email address
- `POST /password-reset` - Initiate password reset
- `POST /password-change` - Change user password
- `GET /users/me` - Get current user info

### Sources (`/api/sources`)
- `GET /sources` - List user's RSS sources
- `POST /sources` - Create new source with validation
- `GET /sources/{id}` - Get source details
- `PUT /sources/{id}` - Update source
- `DELETE /sources/{id}` - Delete source
- `POST /sources/test` - Validate RSS feed URL

### Schedules (`/api/schedules`)
- `GET /schedules` - List user's schedules
- `POST /schedules` - Create schedule with cron expression
- `GET /schedules/{id}` - Get schedule details
- `PUT /schedules/{id}` - Update schedule
- `DELETE /schedules/{id}` - Delete schedule

### Delivery Channels (`/api/schedules/{schedule_id}/channels`)
- `GET /channels` - List delivery channels for schedule
- `POST /channels` - Create Discord/Email channel with validation
- `PUT /channels/{channel_id}` - Enable/disable channel
- `DELETE /channels/{channel_id}` - Remove channel

## Service Layer Architecture

### AuthService
- User registration with email verification
- JWT token generation and validation
- Password hashing (Argon2) and reset flow
- Session management with Redis

### SourceService
- CRUD operations on RSS sources
- Feed validation with feedparser
- Deduplication detection
- Category-based organization

### DigestService
- Article fetching from multiple RSS sources
- Intelligent deduplication (hash + word overlap)
- LLM-powered summarization via Groq API
- Circuit breaker pattern for API failures
- Response caching in Redis (12h TTL)
- Fallback digest generation on API errors
- Multiple output formats (Markdown, HTML, Plain text)

### ScheduleService
- Cron expression parsing with croniter
- Next run time calculation with timezone support
- Due schedule detection for job execution
- Execution tracking and history

### DeliveryService
- Discord webhook delivery with chunking (2000 char limit)
- Email delivery via SendGrid API
- URL validation and webhook testing
- Error handling for rate limits and timeouts
- Delivery status tracking

## Authentication & Authorization

### Flow
1. User registers with email/password
2. Password hashed with Argon2 and stored
3. Email verification token sent
4. After verification, user can login
5. Backend returns JWT access token
6. Frontend stores token in NextAuth session
7. Subsequent API requests include `Authorization: Bearer <token>` header

### User Isolation
- All endpoints verify `user_id` from JWT token
- Users can only access their own sources/schedules/digests
- Database queries filtered by user_id for security

## Digest Generation Pipeline

```
1. Scheduler detects due schedule (cron match)
2. Fetch all articles from schedule's sources
3. Deduplicate articles (hash-based then word overlap)
4. Prepare article batch for summarization
5. Call Groq API with article list
6. Parse LLM response into digest summary
7. Save digest to database
8. Queue delivery tasks for all channels
9. Deliver via Discord webhooks / SendGrid email
10. Update delivery status
```

### Error Handling
- **Groq API failure**: Fallback to generic bullet-point summary
- **Rate limiting (429)**: Retry with exponential backoff
- **Connection timeout**: Log error and mark digest as failed
- **Discord webhook invalid**: Validate before storing channel
- **Email delivery failed**: Queue for retry later

## Frontend Architecture

### Page Structure
```
/app/layout.tsx
├── /auth/login
├── /auth/register
└── /dashboard (protected)
    ├── /page.tsx (home with quick stats)
    ├── /sources
    │   ├── /page.tsx (list)
    │   ├── /new/page.tsx (create)
    │   └── /[id]/page.tsx (edit)
    ├── /schedules
    │   ├── /page.tsx (list)
    │   ├── /new/page.tsx (create)
    │   └── /[id]/page.tsx (edit)
    ├── /digests/page.tsx (preview history)
    └── /settings/page.tsx (profile & password)
```

### Components
- **Navbar**: Top navigation with user menu
- **Sidebar**: Left navigation with active indicator
- **SourceForm**: Form with URL validation and test feed button
- **SourceCard**: List item with edit/delete actions
- **ScheduleForm**: Cron presets and timezone selector
- **ScheduleCard**: Schedule details with next run time

### API Integration
- Axios client with JWT token injection via interceptors
- Error handling with 401 redirect to login
- Request/response validation with Zod
- React Hook Form for complex form state

## Deployment Architecture

### Local Development
```bash
# Terminal 1: Backend
cd backend && python -m uvicorn app.main:app --reload

# Terminal 2: Frontend  
cd frontend && npm run dev

# Terminal 3: Celery Worker (optional)
cd backend && celery -A app.tasks worker --loglevel=info
```

### Production Deployment
- Backend: Docker container on Railway/Render
- Frontend: Deployed to Vercel with automatic builds
- Database: PostgreSQL managed service
- Cache: Redis managed service
- Environment variables in GitHub Secrets for CI/CD

## Testing Strategy

### Backend Testing (83% coverage)
- **Unit Tests**: Service methods with mocked dependencies
- **Integration Tests**: API endpoints with test database
- **Database Isolation**: SQLite in-memory with StaticPool
- **External API Mocking**: Groq, Discord, SendGrid mocked
- **Fixtures**: Reusable test data (users, sources, schedules)

### Frontend Testing (Todo: Add Jest + React Testing Library)
- Component unit tests
- Integration tests for user flows
- API mocking with MSW (Mock Service Worker)

## Security Considerations

### Authentication & Authorization
- JWT tokens with 24-hour expiration
- Argon2 password hashing (resistant to GPU attacks)
- User isolation: queries filtered by user_id
- HTTPS required in production

### Input Validation
- Pydantic schema validation on all inputs
- URL validation with Pydantic HttpUrl
- Email format validation with email-validator
- Cron expression validation with croniter

### API Security
- CORS restricted to frontend origin
- Rate limiting via Redis (per schedule: 10 digests/hour)
- Timeout handling for external API calls
- Circuit breaker for failing services

### Data Protection
- Database queries use parameterized statements (SQLAlchemy)
- No sensitive data in logs
- Environment variables for API keys
- GDPR compliance: 90-day article retention policy

## Performance Optimizations

### Backend
- Async/await for I/O operations (FastAPI, SQLAlchemy async)
- Response caching in Redis (digest generation: 12h TTL)
- Connection pooling for database
- Article deduplication before LLM call (reduce costs)
- Discord message chunking (2000 char limit)

### Frontend
- Next.js static pre-rendering where possible
- Code splitting and lazy loading
- Tailwind CSS for small bundle size
- Image optimization via Next.js Image component

## Monitoring & Observability

### Logging
- Structured logging to stdout
- Service-level error tracking (future: Sentry)
- Celery task logs in worker output

### Metrics (Future)
- API response times
- Digest generation duration
- Delivery success rates
- Cache hit ratios

## Future Enhancements (Phase 2-3)

### Phase 2
- GitHub OAuth integration
- Slack webhook delivery
- Telegram bot integration
- Web scraping for non-RSS websites
- Advanced scheduling (custom times per timezone)

### Phase 3
- Podcast generation (Google Cloud TTS)
- Newsletter HTML templates
- Analytics dashboard
- Premium tier with limits

## Development Workflow

1. Create feature branch: `git checkout -b feature/name`
2. Make changes with tests
3. Run test suite: `pytest tests/ --cov=app`
4. Commit: `git commit -m "feat: description"`
5. Push to origin and create pull request
6. After merge, GitHub Actions deploys automatically

---

**Last Updated**: April 2026
**Maintainer**: News Agent Team
