# app/workers/tasks.py

from multiprocessing import Process
import asyncio
from typing import Dict, List, Tuple

from sqlalchemy import select

from app.db.session import async_session
from app.db.models import Embedding, MatchFeature, JobPost
from app.db.crud import upsert_match_feature
from app.features.similarity import (
    DEFAULT_SIMILARITY_WEIGHTS,
    multi_embedding_similarity,
)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


async def get_entity_embeddings(
    session,
    entity_type: str,   # "JOB" | "FREELANCER"
    entity_id: str,
) -> Dict[str, List[float]]:
    """
    Lấy tất cả embedding của 1 entity, key theo kind:
      {"FULL": [...], "SKILLS": [...], "DOMAIN": [...]}
    Chỉ lấy cho 1 model (DEFAULT_MODEL).
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
        # row.kind: "FULL" | "SKILLS" | "DOMAIN"
        result[row.kind] = row.vector
    return result


async def recompute_matches_for_job(job_id: str, top_n: int = 200):
    """
    Recompute similarity cho 1 job với các freelancer hiện có trong match_feature
    (hoặc sau này bạn có thể thay bằng bảng freelancer).
    """
    async with async_session() as session:
        job_embs = await get_entity_embeddings(session, "JOB", job_id)
        if not job_embs:
            print(f"[recompute_matches_for_job] No embeddings for JOB {job_id}")
            return

        # Lấy list freelancer_id hiện có (có thể là toàn bộ freelancer trong hệ thống sau này)
        stmt_fids = select(MatchFeature.freelancer_id).distinct()
        freelancer_ids = [row[0] for row in (await session.execute(stmt_fids)).all()]

        scored: List[Tuple[str, float]] = []

        for fid in freelancer_ids:
            fr_embs = await get_entity_embeddings(session, "FREELANCER", fid)
            if not fr_embs:
                continue

            sim = multi_embedding_similarity(job_embs, fr_embs, weights=DEFAULT_SIMILARITY_WEIGHTS)
            if sim is None:
                continue

            scored.append((fid, sim))

        # sort lấy top_n
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_n]

        # upsert match_feature
        for freelancer_id, sim in top:
            await upsert_match_feature(
                session,
                job_id=job_id,
                freelancer_id=freelancer_id,
                similarity_score=sim,
                # các feature khác (budget_gap, level_gap, ...) để sau cũng được
            )

        print(f"[recompute_matches_for_job] JOB {job_id} => updated {len(top)} pairs.")


async def recompute_matches_for_freelancer(freelancer_id: str, top_n: int = 200):
    """
    Recompute similarity cho 1 freelancer với các job còn sống (is_deleted = 0).
    """
    async with async_session() as session:
        fr_embs = await get_entity_embeddings(session, "FREELANCER", freelancer_id)
        if not fr_embs:
            print(f"[recompute_matches_for_freelancer] No embeddings for FREELANCER {freelancer_id}")
            return

        # Lấy list job_id muốn xét (job còn active, chưa xoá)
        stmt_jobs = select(JobPost.id).where(JobPost.is_deleted == 0)
        job_ids = [row[0] for row in (await session.execute(stmt_jobs)).all()]

        scored: List[Tuple[str, float]] = []

        for job_id in job_ids:
            job_embs = await get_entity_embeddings(session, "JOB", job_id)
            if not job_embs:
                continue

            sim = multi_embedding_similarity(job_embs, fr_embs, weights=DEFAULT_SIMILARITY_WEIGHTS)
            if sim is None:
                continue

            scored.append((job_id, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_n]

        for job_id, sim in top:
            await upsert_match_feature(
                session,
                job_id=job_id,
                freelancer_id=freelancer_id,
                similarity_score=sim,
            )

        print(f"[recompute_matches_for_freelancer] FREELANCER {freelancer_id} => updated {len(top)} pairs.")


def _job_worker(job_id: str, top_n: int) -> None:
    asyncio.run(recompute_matches_for_job(job_id, top_n))


def _freelancer_worker(freelancer_id: str, top_n: int) -> None:
    asyncio.run(recompute_matches_for_freelancer(freelancer_id, top_n))


def schedule_recompute_for_job(job_id: str, top_n: int = 200) -> None:
    """
    Spawn 1 process nền để recompute match cho job.
    Dùng được trên Windows (spawn).
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
