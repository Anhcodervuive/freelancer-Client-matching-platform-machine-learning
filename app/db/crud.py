# app/db/crud.py
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from app.db.models import Embedding, MatchFeature
from app.config import DATABASE_URL
from typing import List, Optional
import uuid
from datetime import datetime, timezone

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def save_embedding(
    session: AsyncSession,
    entity_type: str,
    entity_id: str,
    kind: str,
    model: str,
    vector: List[float],
):
    """
    UPSERT embedding theo (entity_type, entity_id, kind, model)
    và tự set created_at / updated_at bằng tay.
    """
    now = datetime.now(timezone.utc)

    # 1. Tìm row sẵn có
    stmt = (
      select(Embedding)
      .where(Embedding.entity_type == entity_type)
      .where(Embedding.entity_id == entity_id)
      .where(Embedding.kind == kind)
      .where(Embedding.model == model)
    )
    result = await session.execute(stmt)
    obj: Optional[Embedding] = result.scalars().first()

    if obj is None:
        # 2. Chưa có → tạo mới
        obj = Embedding(
            id=str(uuid.uuid4()),
            entity_type=entity_type,
            entity_id=entity_id,
            kind=kind,
            model=model,
            vector=vector,
            created_at=now,   # ✅ tự set
            updated_at=now,   # ✅ tự set
        )
        session.add(obj)
    else:
        # 3. Đã có → update vector + updated_at
        obj.vector = vector
        obj.updated_at = now   # ✅ tự set

    await session.commit()
    await session.refresh(obj)
    return obj


async def upsert_match_feature(
    session: AsyncSession,
    *,
    job_id: str,
    freelancer_id: str,

    # ----- CORE SIMILARITY / GAP -----
    similarity_score: float | None = None,
    budget_gap: float | None = None,
    timezone_gap_hours: int | None = None,
    level_gap: int | None = None,

    # ----- JOB FEATURES -----
    job_experience_level_num: int | None = None,
    job_required_skill_count: int | None = None,
    job_screening_question_count: int | None = None,
    job_stats_applies: int | None = None,
    job_stats_offers: int | None = None,
    job_stats_accepts: int | None = None,

    # ----- FREELANCER FEATURES -----
    freelancer_skill_count: int | None = None,
    freelancer_stats_applies: int | None = None,
    freelancer_stats_offers: int | None = None,
    freelancer_stats_accepts: int | None = None,
    freelancer_invite_accept_rate: float | None = None,
    freelancer_region: str | None = None,

    # ----- PAIRWISE FEATURES -----
    skill_overlap_count: int | None = None,
    skill_overlap_ratio: float | None = None,
    has_past_collaboration: bool | None = None,
    past_collaboration_count: int | None = None,
    has_viewed_job: bool | None = None,

    # ----- ML OUTPUT -----
    p_match: float | None = None,
    p_freelancer_accept: float | None = None,
    p_client_accept: float | None = None,
):
    """
    Full upsert match_feature cho ML.
    Chỉ ghi đè field nếu param != None.
    """
    mf_id = f"{job_id}-{freelancer_id}"
    mf = await session.get(MatchFeature, mf_id)
    now = datetime.now(timezone.utc)

    if not mf:
        mf = MatchFeature(
            id=mf_id,
            job_id=job_id,
            freelancer_id=freelancer_id,
            created_at=now,
            updated_at=now,
        )
        session.add(mf)

    # cập nhật field nếu param != None
    for field, value in locals().items():
        if field in ["session", "job_id", "freelancer_id", "mf", "mf_id", "now"]:
            continue
        if value is not None:
            setattr(mf, field, value)

    mf.updated_at = now

    await session.commit()
    return mf
