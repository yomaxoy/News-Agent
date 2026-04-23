from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from app.api.v1 import auth, sources, schedules, delivery, digests
from app.config import settings
import logging

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="News Agent API",
    description="Modern web-scraping news aggregation platform",
    version="0.1.0"
)

# Validate critical settings on startup
@app.on_event("startup")
async def startup_checks():
    """Validate required configuration before starting"""
    if settings.ENVIRONMENT == "production":
        if not settings.SECRET_KEY:
            logger.error("CRITICAL: SECRET_KEY environment variable is not set in production")
            raise RuntimeError("SECRET_KEY must be configured in production environment")
        if settings.SECRET_KEY.startswith("your-"):
            logger.error("CRITICAL: Using placeholder SECRET_KEY in production is not allowed")
            raise RuntimeError("SECRET_KEY cannot use the default placeholder in production")

# CORS configuration
# Browsers reject credentialed requests when allow_origins=["*"], so we use
# allow_origin_regex=".*" instead which echoes the request origin back.
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

cors_kwargs = {
    "allow_credentials": True,
    "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
}
if allowed_origins:
    cors_kwargs["allow_origins"] = allowed_origins
else:
    cors_kwargs["allow_origin_regex"] = ".*"

app.add_middleware(CORSMiddleware, **cors_kwargs)

# Include routers
app.include_router(auth.router)
app.include_router(sources.router)
app.include_router(schedules.router)
app.include_router(delivery.router)
app.include_router(digests.router)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "is_production": settings.is_production,
        "cookie_samesite": "none" if settings.is_production else "lax",
        "cookie_secure": settings.is_production,
        "cors_allow_origins": allowed_origins if allowed_origins else "regex:.*",
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "News Agent API - See /docs for API documentation"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
