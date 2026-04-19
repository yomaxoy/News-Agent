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
from app.services.digest import DigestService
from app.db.models import ScheduleSource

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

@router.post("/{schedule_id}/run", status_code=status.HTTP_200_OK)
async def run_schedule_now(
    schedule_id: int,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Run a schedule immediately (generate digest now)"""
    schedule = ScheduleService.get_schedule(db, user_id, schedule_id)

    # Get all sources for this schedule
    schedule_sources = db.query(ScheduleSource).filter(
        ScheduleSource.schedule_id == schedule_id
    ).all()
    source_ids = [ss.source_id for ss in schedule_sources]

    if not source_ids:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Schedule has no sources configured"
        )

    # Fetch and process articles
    articles = DigestService.fetch_articles_from_sources(
        db, source_ids, schedule.max_articles
    )

    if not articles:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No articles found from configured sources"
        )

    articles = DigestService.deduplicate_articles(articles)

    # Generate digest
    digest_content = DigestService.generate_digest(articles, max_articles=schedule.max_articles)

    # Save digest
    digest = DigestService.save_digest(db, schedule_id, digest_content)

    # Update schedule last_run_at
    ScheduleService.mark_schedule_run(db, schedule_id)

    return {
        "status": "success",
        "digest_id": digest.id,
        "articles_processed": len(articles),
        "message": f"Digest generated with {len(articles)} articles"
    }
