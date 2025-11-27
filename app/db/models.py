# app/db/models.py
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, DateTime, JSON, Numeric, Integer, Float
from sqlalchemy.sql import func

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
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class MatchFeature(Base):
    __tablename__ = "match_feature"
    id = Column(String, primary_key=True)
    job_id = Column(String)
    freelancer_id = Column(String)
    similarityScore = Column(Float, nullable=True)
    budget_gap = Column(Numeric(12, 2), nullable=True)
    rate_gap = Column(Numeric(12, 2), nullable=True)
    timezone_gap_hours = Column(Integer, nullable=True)
    level_gap = Column(Integer, nullable=True)
    p_match = Column(String)
    p_freelancer_accept = Column(String)
    p_client_accept = Column(String)
    last_interaction_at = Column(DateTime(timezone=True), nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
