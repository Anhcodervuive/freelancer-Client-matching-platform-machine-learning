# app/workers/tasks.py (tiếp)

from multiprocessing import Process
import asyncio

from app.db.session import async_session
from app.db.models import MatchFeature
from app.features.similarity import multi_embedding_similarity
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Embedding
from typing import Dict, List

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


async def get_entity_embeddings(
    session: AsyncSession,
    entity_type: str,       # "JOB" | "FREELANCER"
    entity_id: str,
) -> Dict[str, List[float]]:
    """
    Lấy tất cả embedding của 1 entity, key theo kind:
    return kiểu: {"FULL": [...], "SKILLS": [...], "DOMAIN": [...]}
    """
    stmt = (
        select(Embedding)
        .where(Embedding.entity_type == entity_type)
        .where(Embedding.entity_id == entity_id)
        .where(Embedding.model == DEFAULT_MODEL)
    )
    rows = (await session.execute(stmt)).scalars().all()

    result: Dict[str, List[float]] = {}
    for row in rows:
        # row.kind = "FULL" | "SKILLS" | "DOMAIN"
        result[row.kind] = row.vector
        print(row.id)
    return result


async def recompute_matches_for_job(job_id: str, top_n: int = 200):
    async with async_session() as session:
        # 1) Lấy tất cả embedding (FULL/SKILLS/DOMAIN) của job
        job_embs = await get_entity_embeddings(session, "JOB", job_id)
        if not job_embs:
            return

        # 2) Lấy list freelancer_id mà mình muốn xét (ví dụ: tất cả freelancer)
        stmt_fids = select(MatchFeature.freelancer_id).distinct()
        freelancer_ids = [row[0] for row in (await session.execute(stmt_fids)).all()]

        scored: list[tuple[str, float]] = []

        for fid in freelancer_ids:
            fr_embs = await get_entity_embeddings(session, "FREELANCER", fid)
            if not fr_embs:
                continue

            score = multi_embedding_similarity(job_embs, fr_embs)
            if score is None:
                continue

            scored.append((fid, score))

        # 3) Lấy top_n theo similarity
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_n]

        # 4) Upsert vào bảng match_feature
        for freelancer_id, sim in top:
            mf_id = f"{job_id}-{freelancer_id}"
            mf = await session.get(MatchFeature, mf_id)
            if not mf:
                mf = MatchFeature(
                    id=mf_id,
                    job_id=job_id,
                    freelancer_id=freelancer_id,
                )
            mf.similarityScore = sim  # cột similarity_score
            session.add(mf)

        await session.commit()


async def recompute_matches_for_freelancer(freelancer_id: str, top_n: int = 200):
    async with async_session() as session:
        fr_embs = await get_entity_embeddings(session, "FREELANCER", freelancer_id)
        if not fr_embs:
            return

        # Lấy list job_id muốn xét (ví dụ: tất cả job đang PUBLISHED)
        from app.db.models import JobPost  # nếu bạn đã map

        stmt_jobs = select(JobPost.id).where(JobPost.is_deleted == False)
        job_ids = [row[0] for row in (await session.execute(stmt_jobs)).all()]

        scored: list[tuple[str, float]] = []

        for job_id in job_ids:
            job_embs = await get_entity_embeddings(session, "JOB", job_id)
            if not job_embs:
                continue

            score = multi_embedding_similarity(job_embs, fr_embs)
            if score is None:
                continue

            scored.append((job_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_n]

        for job_id, sim in top:
            mf_id = f"{job_id}-{freelancer_id}"
            mf = await session.get(MatchFeature, mf_id)
            if not mf:
                mf = MatchFeature(
                    id=mf_id,
                    job_id=job_id,
                    freelancer_id=freelancer_id,
                )
            mf.similarityScore = sim
            session.add(mf)

        await session.commit()

def _job_worker(job_id: str, top_n: int) -> None:
    asyncio.run(recompute_matches_for_job(job_id, top_n))


def _freelancer_worker(freelancer_id: str, top_n: int) -> None:
    asyncio.run(recompute_matches_for_freelancer(freelancer_id, top_n))


def schedule_recompute_for_job(job_id: str, top_n: int = 200) -> None:
    """
    Spawn 1 process nền để recompute match cho job.
    Dùng được trên Windows (spawn) vì không gọi khi import module.
    """
    p = Process(target=_job_worker, args=(job_id, top_n), daemon=True)
    p.start()


def schedule_recompute_for_freelancer(
    freelancer_id: str,
    top_n: int = 200,
) -> None:
    """
    Spawn 1 process nền để recompute match cho freelancer.
    """
    p = Process(target=_freelancer_worker, args=(freelancer_id, top_n), daemon=True)
    p.start()