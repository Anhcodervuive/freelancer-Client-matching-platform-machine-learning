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
    similarity_score: float | None = None,
    budget_gap: float | None = None,
    rate_gap: float | None = None,
    timezone_gap_hours: int | None = None,
    level_gap: int | None = None,
    p_match: float | None = None,
    p_freelancer_accept: float | None = None,
    p_client_accept: float | None = None,
):
    """
    UPSERT 1 record match_feature cho (job, freelancer).
    Mặc định không ghi đè field nào nếu param = None.
    """
    mf_id = f"{job_id}-{freelancer_id}"
    mf = await session.get(MatchFeature, mf_id)

    now = datetime.now(timezone.utc)

    # nếu đã có record
    if mf:
        # update các field
        if rate_gap is not None:
            mf.rate_gap = rate_gap
        if timezone_gap_hours is not None:
            mf.timezone_gap_hours = timezone_gap_hours
        if level_gap is not None:
            mf.level_gap = level_gap
        if p_match is not None:
            mf.p_match = p_match
        if p_freelancer_accept is not None:
            mf.p_freelancer_accept = p_freelancer_accept
        if p_client_accept is not None:
            mf.p_client_accept = p_client_accept

        mf.updated_at = now

    else:
        # create mới
        mf = MatchFeature(
            id=mf_id,
            job_id=job_id,
            freelancer_id=freelancer_id,
            similarity_score=similarity_score,
            budget_gap=budget_gap,
            rate_gap=rate_gap,
            timezone_gap_hours=timezone_gap_hours,
            level_gap=level_gap,
            p_match=p_match,
            p_freelancer_accept=p_freelancer_accept,
            p_client_accept=p_client_accept,
            last_interaction_at=None,

            # ✅ set tay khi insert
            created_at=now,
            updated_at=now,
        )
        session.add(mf)

    await session.commit()
    await session.refresh(mf)
    return mf
