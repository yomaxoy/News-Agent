from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from app.api.v1 import auth, sources, schedules, delivery, digests
from app.core.config import settings
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
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000"
).split(",")

if settings.ENVIRONMENT != "production":
    # In development, allow all origins for easier testing
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

# Include routers
app.include_router(auth.router)
app.include_router(sources.router)
app.include_router(schedules.router)
app.include_router(delivery.router)
app.include_router(digests.router)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "environment": os.getenv("ENVIRONMENT", "development")}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "News Agent API - See /docs for API documentation"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
