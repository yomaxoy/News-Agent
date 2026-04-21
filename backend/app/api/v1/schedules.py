"""Schedule management endpoints"""
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from datetime import datetime
import json
import logging
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
from app.services.delivery import DeliveryService
from app.db.models import ScheduleSource

logger = logging.getLogger(__name__)

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

    # Fetch and process articles with fallback logic
    articles = DigestService.fetch_articles_from_sources(
        db, source_ids, schedule.max_articles, max_age_hours=24
    )

    fallback_message = ""

    if not articles:
        logger.info(f"No articles found in last 24 hours for schedule {schedule_id}, trying last 7 days...")
        fallback_message = "⚠️ Keine Artikel in den vergangenen 24 Stunden gefunden. Suche nach Artikeln der letzten 7 Tage...\n\n"

        articles = DigestService.fetch_articles_from_sources(
            db, source_ids, schedule.max_articles, max_age_hours=168  # 7 days
        )

    if not articles:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Es gibt keine aktuellen Artikel in den konfigurierten Quellen (weder in den letzten 24 Stunden noch in den letzten 7 Tagen)"
        )

    articles = DigestService.deduplicate_articles(articles)

    # Generate digest with user preferences
    digest_content = DigestService.generate_digest(
        articles,
        profile=schedule.profile,
        language=schedule.language,
        max_articles=schedule.max_articles
    )

    # Prepend fallback message if articles were fetched from older time period
    if fallback_message:
        digest_content = fallback_message + digest_content

    # Ensure content is never None or empty
    if not digest_content or not digest_content.strip():
        digest_content = f"Daily News Digest – {datetime.now().strftime('%Y-%m-%d')}\n\nNo content could be generated. Please check your sources."

    # Save digest
    digest = DigestService.save_digest(db, schedule_id, digest_content)

    # Update schedule last_run_at
    ScheduleService.mark_schedule_run(db, schedule_id)

    # Send digest to configured delivery channels
    delivery_results = {}
    for channel in schedule.delivery_channels:
        if not channel.is_enabled:
            continue

        try:
            channel_config = json.loads(channel.config) if isinstance(channel.config, str) else channel.config

            if channel.type == "discord":
                success = DeliveryService.deliver_via_discord(
                    webhook_url=channel_config.get("webhook_url"),
                    content=digest_content
                )
                delivery_results["discord"] = "sent" if success else "failed"

            elif channel.type == "email":
                success = DeliveryService.deliver_via_email(
                    email=channel_config.get("email"),
                    subject=f"News Digest – {datetime.now().strftime('%Y-%m-%d')}",
                    html_content=digest_content,
                    sendgrid_api_key=None  # Uses env var from config
                )
                delivery_results["email"] = "sent" if success else "failed"
            else:
                logger.warning(f"Unknown delivery channel type: {channel.type}")

        except Exception as e:
            logger.error(f"Error sending digest via {channel.type}: {e}")
            delivery_results[channel.type] = f"error: {str(e)[:50]}"

    return {
        "status": "success",
        "digest_id": digest.id,
        "articles_processed": len(articles),
        "message": f"Digest generated with {len(articles)} articles",
        "deliveries": delivery_results
    }
