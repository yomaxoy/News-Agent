"""Schedule management endpoints"""
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.core.security import get_current_user
from app.schemas.schedule import (
    ScheduleCreate,
    ScheduleUpdate,
    ScheduleResponse,
    ScheduleWithSources
)
from app.services.schedule import ScheduleService

router = APIRouter(prefix="/api/schedules", tags=["schedules"])

@router.get("", response_model=list[ScheduleResponse])
async def list_schedules(
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    """List all schedules for authenticated user"""
    schedules = ScheduleService.list_schedules(db, user_id, skip, limit)
    return schedules

@router.post("", response_model=ScheduleResponse, status_code=status.HTTP_201_CREATED)
async def create_schedule(
    schedule_create: ScheduleCreate,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new schedule"""
    schedule = ScheduleService.create_schedule(
        db,
        user_id,
        schedule_create.source_ids or [],
        schedule_create
    )
    return schedule

@router.get("/{schedule_id}", response_model=ScheduleResponse)
async def get_schedule(
    schedule_id: int,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get schedule details"""
    schedule = ScheduleService.get_schedule(db, user_id, schedule_id)
    return schedule

@router.put("/{schedule_id}", response_model=ScheduleResponse)
async def update_schedule(
    schedule_id: int,
    schedule_update: ScheduleUpdate,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update schedule details"""
    schedule = ScheduleService.update_schedule(db, user_id, schedule_id, schedule_update)
    return schedule

@router.delete("/{schedule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_schedule(
    schedule_id: int,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a schedule"""
    ScheduleService.delete_schedule(db, user_id, schedule_id)
    return None
