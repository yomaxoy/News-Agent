from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, Index, UniqueConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(128), unique=True, index=True, nullable=True)
    password_hash = Column(String(255), nullable=False)
    email_verified = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    sources = relationship("Source", back_populates="user", cascade="all, delete-orphan")
    schedules = relationship("Schedule", back_populates="user", cascade="all, delete-orphan")

class Source(Base):
    __tablename__ = "sources"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    url = Column(Text, nullable=False)
    category = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="sources")
    articles = relationship("Article", back_populates="source", cascade="all, delete-orphan")
    schedules = relationship("Schedule", secondary="schedule_sources", back_populates="sources")

    __table_args__ = (
        Index('idx_user_created', 'user_id', 'created_at'),
        Index('idx_sources_active_user', 'is_active', 'user_id'),
        UniqueConstraint('user_id', 'url', name='uq_source_user_url'),
    )

class Schedule(Base):
    __tablename__ = "schedules"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=True)
    cron_expression = Column(String(255), nullable=False)
    timezone = Column(String(100), default="UTC")
    is_active = Column(Boolean, default=True, nullable=False)
    next_run_at = Column(DateTime(timezone=True), nullable=True)
    last_run_at = Column(DateTime(timezone=True), nullable=True)
    max_articles = Column(Integer, default=7)
    profile = Column(String(500), nullable=False, server_default="Allgemeine Nachrichten")
    language = Column(String(50), nullable=False, server_default="Deutsch")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="schedules")
    sources = relationship("Source", secondary="schedule_sources", back_populates="schedules")
    digests = relationship("Digest", back_populates="schedule", cascade="all, delete-orphan")
    delivery_channels = relationship("DeliveryChannel", back_populates="schedule", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_user_next_run', 'user_id', 'next_run_at'),
    )

class ScheduleSource(Base):
    __tablename__ = "schedule_sources"

    id = Column(Integer, primary_key=True, index=True)
    schedule_id = Column(Integer, ForeignKey("schedules.id", ondelete="CASCADE"), nullable=False)
    source_id = Column(Integer, ForeignKey("sources.id", ondelete="CASCADE"), nullable=False)

    __table_args__ = (
        Index('idx_schedule_sources_source_id', 'source_id'),
    )

class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("sources.id", ondelete="CASCADE"), nullable=False)
    external_id = Column(String(255), nullable=True)
    title = Column(String(500), nullable=False)
    summary = Column(Text, nullable=True)
    url = Column(Text, nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    source = relationship("Source", back_populates="articles")

    __table_args__ = (
        Index('idx_source_published', 'source_id', 'published_at'),
        UniqueConstraint('source_id', 'url', name='uq_article_source_url'),
    )

class Digest(Base):
    __tablename__ = "digests"

    id = Column(Integer, primary_key=True, index=True)
    schedule_id = Column(Integer, ForeignKey("schedules.id", ondelete="CASCADE"), nullable=False)
    content_text = Column(Text, nullable=True)
    content_html = Column(Text, nullable=True)
    status = Column(String(50), default="generated")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    schedule = relationship("Schedule", back_populates="digests")

    __table_args__ = (
        Index('idx_schedule_created', 'schedule_id', 'created_at'),
    )

class DeliveryChannel(Base):
    __tablename__ = "delivery_channels"

    id = Column(Integer, primary_key=True, index=True)
    schedule_id = Column(Integer, ForeignKey("schedules.id", ondelete="CASCADE"), nullable=False)
    type = Column(String(50), nullable=False)  # discord, email, slack, telegram
    config = Column(Text, nullable=False)  # JSON-encoded
    is_enabled = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    schedule = relationship("Schedule", back_populates="delivery_channels")

class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    schedule_id = Column(Integer, ForeignKey("schedules.id", ondelete="SET NULL"), nullable=True)
    status = Column(String(50), default="queued")  # queued, running, completed, failed
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index('idx_schedule_status', 'schedule_id', 'status'),
    )
