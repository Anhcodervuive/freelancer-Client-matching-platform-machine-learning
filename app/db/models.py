# app/db/models.py
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (
    Column,
    String,
    DateTime,
    JSON,
    Numeric,
    Integer,
    Float,
    Boolean,
)
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class Embedding(Base):
    __tablename__ = "embedding"
    id = Column(String, primary_key=True)
    entity_type = Column(String)
    entity_id = Column(String)
    kind = Column(String)
    model = Column(String)
    version = Column(String, nullable=True)
    vector = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class MatchFeature(Base):
    __tablename__ = "match_feature"

    id = Column(String, primary_key=True)
    job_id = Column(String, nullable=False)
    freelancer_id = Column(String, nullable=False)

    # ----- CORE SIMILARITY / GAP -----
    similarity_score = Column(Float, nullable=True)
    budget_gap = Column(Numeric(12, 2), nullable=True)
    timezone_gap_hours = Column(Integer, nullable=True)
    level_gap = Column(Integer, nullable=True)

    # ----- JOB FEATURES -----
    job_experience_level_num = Column(Integer, nullable=True)
    job_required_skill_count = Column(Integer, nullable=True)
    job_screening_question_count = Column(Integer, nullable=True)
    job_stats_applies = Column(Integer, nullable=True)
    job_stats_offers = Column(Integer, nullable=True)
    job_stats_accepts = Column(Integer, nullable=True)

    # ----- FREELANCER FEATURES -----
    freelancer_skill_count = Column(Integer, nullable=True)
    freelancer_stats_applies = Column(Integer, nullable=True)
    freelancer_stats_offers = Column(Integer, nullable=True)
    freelancer_stats_accepts = Column(Integer, nullable=True)
    freelancer_invite_accept_rate = Column(Float, nullable=True)
    freelancer_region = Column(String, nullable=True)

    # ----- PAIRWISE FEATURES -----
    skill_overlap_count = Column(Integer, nullable=True)
    skill_overlap_ratio = Column(Float, nullable=True)
    has_past_collaboration = Column(Boolean, nullable=True)
    past_collaboration_count = Column(Integer, nullable=True)
    has_viewed_job = Column(Boolean, nullable=True)

    # ----- ML OUTPUTS -----
    p_match = Column(Float, nullable=True)
    p_freelancer_accept = Column(Float, nullable=True)
    p_client_accept = Column(Float, nullable=True)

    last_interaction_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )
    updated_at = Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )


class JobPost(Base):
    __tablename__ = "job_post"

    id = Column(String, primary_key=True)
    specialty_id = Column(String)
    title = Column(String)
    description = Column(String)
    visibility = Column(String)
    status = Column(String)
    is_deleted = Column(Integer)


class JobInvitation(Base):
    __tablename__ = "job_invitation"

    id = Column(String, primary_key=True)
    job_id = Column(String)
    client_id = Column(String)
    freelancer_id = Column(String)
    status = Column(String)  # "SENT" | "ACCEPTED" | "DECLINED" | "EXPIRED" ...
    sent_at = Column(DateTime(timezone=True))
    responded_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True))
