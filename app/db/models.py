# app/db/models.py
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, DateTime, JSON, Numeric, Integer, Float
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
    updated_at = Column(DateTime(timezone=True),server_default=func.now() , onupdate=func.now())

class MatchFeature(Base):
    __tablename__ = "match_feature"
    id = Column(String, primary_key=True)
    job_id = Column(String)
    freelancer_id = Column(String)
    similarityScore = Column("similarity_score", Float, nullable=True)
    budget_gap = Column(Numeric(12, 2), nullable=True)
    rate_gap = Column(Numeric(12, 2), nullable=True)
    timezone_gap_hours = Column(Integer, nullable=True)
    level_gap = Column(Integer, nullable=True)
    p_match = Column(Float, nullable=True)
    p_freelancer_accept = Column(Float, nullable=True)
    p_client_accept = Column(Float, nullable=True)
    last_interaction_at = Column(DateTime(timezone=True), nullable=True)
    # ❌ bỏ server_default / onupdate ở đây
    created_at = Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,      # default client-side (tùy, có cũng được)
    )
    updated_at = Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,      # sẽ bị override bởi crud.py khi cần
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
    # thêm các field bạn thật sự cần

    # ví dụ nếu bạn cần join bảng required skill:
    # relationships nếu cần (không bắt buộc)
