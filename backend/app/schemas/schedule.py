"""Schedule schemas"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime
from croniter import croniter

class ScheduleBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    cron_expression: str = Field(..., min_length=5, max_length=255)
    timezone: Optional[str] = "UTC"
    max_articles: Optional[int] = 7
    profile: Optional[str] = "Allgemeine Nachrichten"
    language: Optional[str] = "Deutsch"

    @field_validator("cron_expression")
    @classmethod
    def validate_cron(cls, v: str) -> str:
        if not croniter.is_valid(v):
            raise ValueError("Invalid cron expression")
        return v

class ScheduleCreate(ScheduleBase):
    source_ids: Optional[List[int]] = []

class ScheduleUpdate(BaseModel):
    name: Optional[str] = None
    cron_expression: Optional[str] = None
    timezone: Optional[str] = None
    max_articles: Optional[int] = None
    is_active: Optional[bool] = None
    profile: Optional[str] = None
    language: Optional[str] = None

    @field_validator("cron_expression")
    @classmethod
    def validate_cron(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not croniter.is_valid(v):
            raise ValueError("Invalid cron expression")
        return v

class ScheduleResponse(ScheduleBase):
    id: int
    user_id: int
    is_active: bool
    next_run_at: Optional[datetime] = None
    last_run_at: Optional[datetime] = None
    created_at: datetime
    profile: str
    language: str

    model_config = {"from_attributes": True}

class ScheduleWithSources(ScheduleResponse):
    source_ids: List[int] = []
