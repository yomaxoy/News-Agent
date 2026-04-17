"""Source schemas for RSS feed management"""
from pydantic import BaseModel, HttpUrl, field_validator
from typing import Optional
from datetime import datetime

AVAILABLE_CATEGORIES = [
    "Tech",
    "Business",
    "News",
    "Entertainment",
    "Science",
    "Health",
    "Sport"
]

class SourceBase(BaseModel):
    name: str
    url: str
    category: Optional[str] = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        try:
            HttpUrl(v)
        except Exception:
            raise ValueError("Invalid URL format")
        return v

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: Optional[str]) -> Optional[str]:
        if v and v not in AVAILABLE_CATEGORIES:
            raise ValueError(f"Category must be one of: {', '.join(AVAILABLE_CATEGORIES)}")
        return v

class SourceCreate(SourceBase):
    pass

class SourceUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    is_active: Optional[bool] = None

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: Optional[str]) -> Optional[str]:
        if v and v not in AVAILABLE_CATEGORIES:
            raise ValueError(f"Category must be one of: {', '.join(AVAILABLE_CATEGORIES)}")
        return v

class SourceResponse(SourceBase):
    id: int
    user_id: int
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}

class FeedTestRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        try:
            HttpUrl(v)
        except Exception:
            raise ValueError("Invalid URL format")
        return v

class FeedTestResponse(BaseModel):
    valid: bool
    title: Optional[str] = None
    entries: int = 0
    error: Optional[str] = None

class CategoriesResponse(BaseModel):
    categories: list[str]
