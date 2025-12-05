"""
CLI script: lấy dữ liệu đã seed trong DB, tính embedding cho Job/Freelancer
và tính điểm match (cosine weighted) giữa từng cặp.

Cách chạy (từ project root):
    python -m app.test.test_db_match_embeddings
hoặc
    python app/test/test_db_match_embeddings.py

Yêu cầu:
    - .env chứa DATABASE_URL (MySQL DSN)
    - pip install sentence-transformers aiomysql

Kết quả in ra:
    - Số job, freelancer lấy được + số skill mỗi bản ghi
    - Điểm similarity FULL / SKILLS / DOMAIN (nếu có) + weighted tổng
    - Lưu embedding + match_feature vào DB (UPSERT)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Dict, List

from sqlalchemy import text

# Cho phép chạy trực tiếp từ thư mục app/test
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import AsyncSessionLocal  # noqa: E402
from app.db.crud import save_embedding, upsert_match_feature  # noqa: E402
from app.features.skill_processing import aggregate_skill_embedding, normalize_skill_list  # noqa: E402
from app.features.similarity import multi_embedding_similarity  # noqa: E402
from app.models.ml_models import embed_text  # noqa: E402

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


async def fetch_jobs(session) -> List[dict]:
    """Trả về danh sách job với title/description/domain + skill names."""

    sql_jobs = text(
        """
        SELECT jp.id, jp.title, jp.description, sp.name AS specialty_name, cat.name AS category_name
        FROM job_post jp
        LEFT JOIN specialty sp ON sp.id = jp.specialty_id
        LEFT JOIN category cat ON cat.id = sp.category_id
        WHERE jp.is_deleted = 0
        """
    )
    sql_job_skills = text(
        """
        SELECT jrs.job_id, s.name AS skill_name
        FROM job_required_skill jrs
        JOIN skill s ON s.id = jrs.skill_id
        """
    )

    jobs = (await session.execute(sql_jobs)).mappings().all()
    skill_rows = (await session.execute(sql_job_skills)).mappings().all()

    skills_by_job: Dict[str, List[str]] = {}
    for row in skill_rows:
        skills_by_job.setdefault(row["job_id"], []).append(row["skill_name"])

    enriched = []
    for job in jobs:
        enriched.append(
            {
                "id": job["id"],
                "title": job["title"] or "",
                "description": job["description"] or "",
                "specialty": job["specialty_name"] or "",
                "category": job["category_name"] or "",
                "skills": skills_by_job.get(job["id"], []),
            }
        )

    return enriched


async def fetch_freelancers(session) -> List[dict]:
    """Trả về danh sách freelancer với title/bio + skill names."""

    sql_fre = text(
        """
        SELECT f.user_id, f.title, f.bio
        FROM freelancer f
        WHERE f.user_id IS NOT NULL
        """
    )
    sql_fre_skills = text(
        """
        SELECT fss.userId AS user_id, s.name AS skill_name
        FROM freelancer_skill_selection fss
        JOIN skill s ON s.id = fss.skillId
        WHERE fss.is_deleted = 0
        """
    )

    freelancers = (await session.execute(sql_fre)).mappings().all()
    skill_rows = (await session.execute(sql_fre_skills)).mappings().all()

    skills_by_fr: Dict[str, List[str]] = {}
    for row in skill_rows:
        skills_by_fr.setdefault(row["user_id"], []).append(row["skill_name"])

    enriched = []
    for fr in freelancers:
        enriched.append(
            {
                "id": fr["user_id"],
                "title": fr["title"] or "",
                "bio": fr["bio"] or "",
                "skills": skills_by_fr.get(fr["user_id"], []),
            }
        )

    return enriched


def build_domain_text(category: str, specialty: str) -> str:
    parts = [p for p in [category, specialty] if p]
    return " / ".join(parts)


async def compute_job_embeddings(job: dict, model_name: str) -> Dict[str, List[float]]:
    """Tính embedding FULL/SKILLS/DOMAIN cho 1 job."""

    embs: Dict[str, List[float]] = {}

    full_text = (job.get("title") or "") + "\n" + (job.get("description") or "")
    embs["FULL"] = embed_text(full_text, model_name=model_name, normalize=True)

    skill_emb = aggregate_skill_embedding(job.get("skills", []), model_name=model_name, normalize=True)
    if skill_emb:
        embs["SKILLS"] = skill_emb

    domain_text = build_domain_text(job.get("category", ""), job.get("specialty", ""))
    if domain_text:
        embs["DOMAIN"] = embed_text(domain_text, model_name=model_name, normalize=True)

    return embs


async def compute_freelancer_embeddings(fr: dict, model_name: str) -> Dict[str, List[float]]:
    embs: Dict[str, List[float]] = {}

    full_text = (fr.get("title") or "") + "\n" + (fr.get("bio") or "")
    embs["FULL"] = embed_text(full_text, model_name=model_name, normalize=True)

    skill_emb = aggregate_skill_embedding(fr.get("skills", []), model_name=model_name, normalize=True)
    if skill_emb:
        embs["SKILLS"] = skill_emb

    return embs


async def persist_embeddings(session, entity_type: str, entity_id: str, embs: Dict[str, List[float]], model_name: str):
    for kind, vec in embs.items():
        await save_embedding(
            session,
            entity_type=entity_type,
            entity_id=entity_id,
            kind=kind,
            model=model_name,
            vector=[float(v) for v in vec],
        )


async def main(model_name: str = DEFAULT_MODEL, top_k: int = 5):
    async with AsyncSessionLocal() as session:
        jobs = await fetch_jobs(session)
        freelancers = await fetch_freelancers(session)

        print(f"Loaded {len(jobs)} job(s) & {len(freelancers)} freelancer(s)")

        job_embs_map: Dict[str, Dict[str, List[float]]] = {}
        for job in jobs:
            embs = await compute_job_embeddings(job, model_name)
            job_embs_map[job["id"]] = embs
            await persist_embeddings(session, "JOB", job["id"], embs, model_name)

        fr_embs_map: Dict[str, Dict[str, List[float]]] = {}
        for fr in freelancers:
            embs = await compute_freelancer_embeddings(fr, model_name)
            fr_embs_map[fr["id"]] = embs
            await persist_embeddings(session, "FREELANCER", fr["id"], embs, model_name)

        print("Embeddings saved. Computing matches...")

        for job in jobs:
            job_id = job["id"]
            scored: List[tuple[str, float]] = []
            for fr in freelancers:
                fr_id = fr["id"]
                sim = multi_embedding_similarity(job_embs_map[job_id], fr_embs_map[fr_id])
                if sim is None:
                    continue
                scored.append((fr_id, sim))

                await upsert_match_feature(
                    session,
                    job_id=job_id,
                    freelancer_id=fr_id,
                    similarity_score=sim,
                    p_match=sim,
                    p_freelancer_accept=sim * 0.9,
                    p_client_accept=sim * 0.9,
                )

            scored.sort(key=lambda x: x[1], reverse=True)
            print("\n=== Job:", job["title"])
            print("Skills:", normalize_skill_list(job.get("skills", [])))
            for fr_id, score in scored[:top_k]:
                fr = next(f for f in freelancers if f["id"] == fr_id)
                print(f"  - {fr['title'] or fr_id}: {score:.4f} | skills={normalize_skill_list(fr.get('skills', []))}")


if __name__ == "__main__":
    asyncio.run(main())
