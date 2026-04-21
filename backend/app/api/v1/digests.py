"""Digest management endpoints"""
from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.db.database import get_db
from app.core.security import get_current_user
from app.db.models import Digest, Schedule

router = APIRouter(prefix="/api/digests", tags=["digests"])


@router.get("", response_model=list[dict])
async def list_digests(
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 30
):
    """List all digests for authenticated user's schedules"""
    # Get all schedules for this user
    schedules = db.query(Schedule).filter(Schedule.user_id == user_id).all()
    schedule_ids = [s.id for s in schedules]

    if not schedule_ids:
        return []

    # Get digests for these schedules, ordered by newest first
    digests = db.query(Digest).filter(
        Digest.schedule_id.in_(schedule_ids)
    ).order_by(desc(Digest.created_at)).offset(skip).limit(limit).all()

    result = []
    for digest in digests:
        result.append({
            "id": digest.id,
            "schedule_id": digest.schedule_id,
            "schedule_name": digest.schedule.name,
            "content_text": digest.content_text,
            "status": digest.status,
            "created_at": digest.created_at.isoformat() + "Z"
        })

    return result


@router.get("/{digest_id}", response_model=dict)
async def get_digest(
    digest_id: int,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get digest details"""
    # Ownership check BEFORE fetching content (prevents timing attacks)
    digest = db.query(Digest).join(Schedule).filter(
        Digest.id == digest_id,
        Schedule.user_id == user_id
    ).first()

    if not digest:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Digest not found"
        )

    schedule = digest.schedule

    return {
        "id": digest.id,
        "schedule_id": digest.schedule_id,
        "schedule_name": schedule.name,
        "content_text": digest.content_text,
        "status": digest.status,
        "created_at": digest.created_at.isoformat() + "Z"
    }
