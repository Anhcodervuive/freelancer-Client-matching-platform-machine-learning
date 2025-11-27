# app/db/crud.py
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from app.db.models import Embedding, MatchFeature
from app.config import DATABASE_URL
import asyncio

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def save_embedding(session: AsyncSession, entity_type, entity_id, kind, model, vector):
    obj = Embedding(entity_type=entity_type, entity_id=entity_id, kind=kind, model=model, vector=vector)
    session.add(obj)
    await session.commit()
    return obj

async def update_match_feature(session: AsyncSession, job_id, freelancer_id, p_match=None, p_accept=None):
    mf = await session.get(MatchFeature, f"{job_id}-{freelancer_id}")
    if not mf:
        mf = MatchFeature(id=f"{job_id}-{freelancer_id}", job_id=job_id, freelancer_id=freelancer_id)
    if p_match:
        mf.p_match = p_match
    if p_accept:
        mf.p_freelancer_accept = p_accept
    session.add(mf)
    await session.commit()
    return mf
