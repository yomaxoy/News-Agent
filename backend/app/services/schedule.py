"""Schedule management service"""
from sqlalchemy.orm import Session
from sqlalchemy import and_
from app.db.models import Schedule, Source, ScheduleSource
from app.schemas.schedule import ScheduleCreate, ScheduleUpdate
from fastapi import HTTPException, status
from croniter import croniter
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class ScheduleService:
    @staticmethod
    def create_schedule(
        db: Session,
        user_id: int,
        source_ids: list,
        schedule_create: ScheduleCreate
    ) -> Schedule:
        """Create a new schedule with sources"""
        # Validate cron expression
        if not croniter.is_valid(schedule_create.cron_expression):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid cron expression"
            )

        # Calculate next run time
        cron = croniter(schedule_create.cron_expression, datetime.now(timezone.utc))
        next_run = cron.get_next(datetime)

        schedule = Schedule(
            user_id=user_id,
            name=schedule_create.name,
            cron_expression=schedule_create.cron_expression,
            timezone=schedule_create.timezone or "UTC",
            max_articles=schedule_create.max_articles or 7,
            next_run_at=next_run,
            is_active=True
        )
        db.add(schedule)
        db.commit()
        db.refresh(schedule)

        # Add sources to schedule
        for source_id in source_ids:
            # Verify source exists and belongs to user
            source = db.query(Source).filter(
                and_(Source.id == source_id, Source.user_id == user_id)
            ).first()

            if not source:
                db.delete(schedule)
                db.commit()
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Source {source_id} not found"
                )

            schedule_source = ScheduleSource(
                schedule_id=schedule.id,
                source_id=source_id
            )
            db.add(schedule_source)

        db.commit()
        return schedule

    @staticmethod
    def get_schedule(db: Session, user_id: int, schedule_id: int) -> Schedule:
        """Get a schedule (only if owned by user)"""
        schedule = db.query(Schedule).filter(
            and_(Schedule.id == schedule_id, Schedule.user_id == user_id)
        ).first()

        if not schedule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Schedule not found"
            )

        return schedule

    @staticmethod
    def list_schedules(db: Session, user_id: int, skip: int = 0, limit: int = 100):
        """List all schedules for a user"""
        return db.query(Schedule).filter(
            Schedule.user_id == user_id
        ).offset(skip).limit(limit).all()

    @staticmethod
    def update_schedule(
        db: Session,
        user_id: int,
        schedule_id: int,
        schedule_update: ScheduleUpdate
    ) -> Schedule:
        """Update a schedule"""
        schedule = ScheduleService.get_schedule(db, user_id, schedule_id)

        # Validate cron expression if updated
        if schedule_update.cron_expression:
            if not croniter.is_valid(schedule_update.cron_expression):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid cron expression"
                )
            schedule.cron_expression = schedule_update.cron_expression
            # Recalculate next run
            cron = croniter(schedule.cron_expression, datetime.now(timezone.utc))
            schedule.next_run_at = cron.get_next(datetime)

        if schedule_update.name is not None:
            schedule.name = schedule_update.name

        if schedule_update.timezone is not None:
            schedule.timezone = schedule_update.timezone

        if schedule_update.max_articles is not None:
            schedule.max_articles = schedule_update.max_articles

        if schedule_update.is_active is not None:
            schedule.is_active = schedule_update.is_active

        db.commit()
        db.refresh(schedule)
        return schedule

    @staticmethod
    def delete_schedule(db: Session, user_id: int, schedule_id: int) -> None:
        """Delete a schedule"""
        schedule = ScheduleService.get_schedule(db, user_id, schedule_id)
        db.delete(schedule)
        db.commit()

    @staticmethod
    def calculate_next_run(cron_expression: str) -> datetime:
        """Calculate next run time from cron expression"""
        try:
            cron = croniter(cron_expression, datetime.now(timezone.utc))
            return cron.get_next(datetime)
        except Exception as e:
            logger.error(f"Error calculating next run: {e}")
            return None

    @staticmethod
    def get_due_schedules(db: Session) -> list:
        """Get all schedules that are due to run"""
        now = datetime.now(timezone.utc)
        due_schedules = db.query(Schedule).filter(
            and_(
                Schedule.is_active == True,
                Schedule.next_run_at <= now
            )
        ).all()

        return due_schedules

    @staticmethod
    def mark_schedule_run(db: Session, schedule_id: int) -> Schedule:
        """Mark schedule as run and calculate next run time"""
        schedule = db.query(Schedule).filter(Schedule.id == schedule_id).first()

        if not schedule:
            return None

        schedule.last_run_at = datetime.now(timezone.utc)

        # Calculate next run
        cron = croniter(schedule.cron_expression, datetime.now(timezone.utc))
        schedule.next_run_at = cron.get_next(datetime)

        db.commit()
        db.refresh(schedule)
        return schedule
