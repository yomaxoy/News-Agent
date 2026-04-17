# Sentry Error Tracking Setup

This guide explains how to setup Sentry for error monitoring in production.

## What is Sentry?

Sentry is an error tracking platform that:
- Captures unhandled exceptions in real-time
- Sends alerts for critical errors
- Tracks error trends and patterns
- Helps debug production issues
- Free tier: 100 events/month, perfect for MVP

## Setup

### 1. Create Sentry Account

1. Go to https://sentry.io/signup
2. Sign up with email
3. Create organization (e.g., "News Agent")
4. Create project: Select "Python" for backend, "JavaScript" for frontend

### 2. Backend Integration (FastAPI)

#### Install SDK
```bash
pip install sentry-sdk
```

#### Configure in `backend/app/main.py`

Add at the top of the file:

```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlAlchemyIntegration
import os

# Initialize Sentry
if os.getenv("ENVIRONMENT") == "production":
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),
        integrations=[
            FastApiIntegration(),
            SqlAlchemyIntegration(),
        ],
        # Performance monitoring (25% of transactions)
        traces_sample_rate=0.25,
        # Error sampling (100%)
        sample_rate=1.0,
        environment=os.getenv("ENVIRONMENT", "development"),
        # Attach stack traces
        attach_stacktrace=True,
    )
```

#### Set Environment Variable

In Railway (or your production environment):
```
SENTRY_DSN=https://your-sentry-dsn-here@sentry.io/project-id
```

Get this from Sentry project settings.

### 3. Frontend Integration (Next.js)

#### Install SDK
```bash
npm install @sentry/nextjs
```

#### Configure in `frontend/next.config.ts`

```typescript
import { withSentryConfig } from "@sentry/nextjs";

const nextConfig = {
  // your Next.js config
};

export default withSentryConfig(nextConfig, {
  org: "your-org-slug",
  project: "your-project-name",
  authToken: process.env.SENTRY_AUTH_TOKEN,
});
```

#### Set Environment Variables

In Vercel:
```
NEXT_PUBLIC_SENTRY_DSN=https://your-sentry-dsn-here@sentry.io/project-id
SENTRY_AUTH_TOKEN=your-auth-token
```

### 4. Configure Alerts

In Sentry Dashboard:

1. Go to Alerts
2. Create Alert Rule:
   - **Condition**: Error rate > 5% in 1 hour
   - **Action**: Send email + Slack notification
   
3. Another Rule:
   - **Condition**: New error (first occurrence)
   - **Action**: Send email immediately

### 5. Test Integration

#### Backend Test
```bash
# Add this endpoint temporarily to test
@app.get("/api/test-error")
async def test_error():
    """Test error tracking"""
    raise Exception("This is a test error")

# Then trigger it:
curl http://localhost:8000/api/test-error
```

#### Frontend Test
```typescript
// Add to frontend page
throw new Error("Test Sentry integration");
```

Check Sentry dashboard - errors should appear within seconds.

## Sentry Dashboard

### Key Features

1. **Issues Dashboard**
   - See all errors grouped by type
   - Frequency and last occurrence
   - Stack traces
   - Affected users
   - Browser/OS info

2. **Performance Monitoring**
   - Slow API endpoints
   - Slow database queries
   - Page load performance

3. **Release Tracking**
   - Track which version introduced errors
   - Compare performance between releases
   - Regression detection

4. **Team & Alerts**
   - Assign issues to team members
   - Resolve/ignore/reopen issues
   - Email/Slack notifications

## Best Practices

### Do's
- [x] Set environment variables for DSN
- [x] Capture meaningful context (user ID, request ID)
- [x] Monitor error trends over time
- [x] Set up alerts for critical errors
- [x] Review Sentry dashboard weekly
- [x] Link errors to commits in GitHub

### Don'ts
- [ ] Don't put DSN in version control
- [ ] Don't log sensitive data (passwords, tokens)
- [ ] Don't monitor every event (use sampling)
- [ ] Don't ignore high-error-rate warnings
- [ ] Don't let errors accumulate without fixing

## Cost Optimization

### Sentry Pricing
- **Free Tier**: 100 events/month
- **Pay-as-you-go**: $0.50 per 10K events
- **Team Plan**: $29/month (50K events)

For MVP with 100 users:
- Estimate: 100-500 errors/month
- Free tier is sufficient

### Reduce Events Captured
```python
# Only capture errors in production
if os.getenv("ENVIRONMENT") != "production":
    sentry_sdk.init(enabled=False)

# Ignore specific errors
def before_send(event, hint):
    if "DeprecationWarning" in str(hint.get("exc_info")):
        return None
    return event

sentry_sdk.init(before_send=before_send)
```

## Troubleshooting

### Errors not appearing in Sentry
1. Check `SENTRY_DSN` is set correctly
2. Verify environment is "production"
3. Check error is actually occurring (check logs)
4. Verify network connectivity to Sentry
5. Check Sentry quota (Free tier: 100 events/month)

### Too many errors
1. Implement sampling (see code above)
2. Add `before_send` filter to exclude noise
3. Fix bugs causing errors
4. Increase Sentry plan

### Performance Issues
1. Check `traces_sample_rate` (default: 100%)
2. Reduce to 10-25% for high-traffic apps
3. Focus on slow endpoints/queries

---

**Last Updated**: April 2026
