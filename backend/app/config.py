import os
from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env")

    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/news_agent_db"
    SQLALCHEMY_ECHO: bool = False

    # Security
    SECRET_KEY: str = ""
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # External APIs
    GROQ_API_KEY: str = ""
    SENDGRID_API_KEY: str = ""
    GOOGLE_APPLICATION_CREDENTIALS: str = ""

    # App
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    @property
    def is_production(self) -> bool:
        """Detect production via explicit ENVIRONMENT or Railway auto-set vars."""
        return (
            self.ENVIRONMENT == "production"
            or bool(os.getenv("RAILWAY_ENVIRONMENT"))
            or bool(os.getenv("RAILWAY_ENVIRONMENT_NAME"))
            or bool(os.getenv("RAILWAY_PROJECT_ID"))
        )

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
