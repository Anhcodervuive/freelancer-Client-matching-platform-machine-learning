"""
CLI script: lấy dữ liệu đã seed trong DB, tính embedding cho Job/Freelancer
và tính điểm match (cosine weighted) giữa từng cặp.

Kết quả:
    - Tính và lưu embedding FULL / SKILLS / DOMAIN
    - Tính similarity_score + các feature numeric:
        + budget_gap
        + timezone_gap_hours
        + level_gap
        + job_experience_level_num
        + job_required_skill_count
        + job_screening_question_count (*approx, nếu có bảng)
        + job_stats_offers, job_stats_accepts  (từ job_invitation)
        + freelancer_skill_count
        + freelancer_stats_offers, freelancer_stats_accepts (từ job_invitation)
        + freelancer_invite_accept_rate      (từ job_invitation)
        + skill_overlap_count, skill_overlap_ratio
        + has_past_collaboration, past_collaboration_count (tạm = 0)
        + has_viewed_job (tạm = False)
    - Lưu vào bảng embedding + match_feature
      (KHÔNG đụng tới p_match / p_*_accept, ML sẽ fill sau)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

from sqlalchemy import text

# Cho phép chạy trực tiếp từ thư mục app/test
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import AsyncSessionLocal  # noqa: E402
from app.db.crud import save_embedding, upsert_match_feature  # noqa: E402
from app.features.skill_processing import (  # noqa: E402
    aggregate_skill_embedding,
    normalize_skill_list,
)
from app.features.similarity import (  # noqa: E402
    DEFAULT_SIMILARITY_WEIGHTS,
    multi_embedding_similarity,
)
from app.models.ml_models import embed_text  # noqa: E402

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ----------------- FETCH CORE DATA -----------------


async def fetch_jobs(session) -> List[dict]:
    """Trả về danh sách job + skill + budget + experience_level + screening_question_count (nếu có)."""

    sql_jobs = text(
        """
        SELECT
            jp.id,
            jp.title,
            jp.description,
            jp.budget_amount,
            jp.experience_level,
            sp.name AS specialty_name,
            cat.name AS category_name
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

    # nếu bạn có bảng job_screening_question thì dùng query này;
    # nếu KHÔNG có bảng đó thì có thể xoá block này, job_screening_question_count sẽ = 0
    sql_screening = text(
        """
        SELECT jsq.job_id, COUNT(*) AS cnt
        FROM job_screening_question jsq
        WHERE jsq.is_deleted = 0
        GROUP BY jsq.job_id
        """
    )

    jobs = (await session.execute(sql_jobs)).mappings().all()
    skill_rows = (await session.execute(sql_job_skills)).mappings().all()

    skills_by_job: Dict[str, List[str]] = {}
    for row in skill_rows:
        skills_by_job.setdefault(row["job_id"], []).append(row["skill_name"])

    screening_by_job: Dict[str, int] = {}
    try:
        screening_rows = (await session.execute(sql_screening)).mappings().all()
        for row in screening_rows:
            screening_by_job[row["job_id"]] = int(row["cnt"])
    except Exception:
        # nếu bảng job_screening_question không tồn tại → cho toàn bộ = 0
        screening_by_job = {}

    enriched = []
    for job in jobs:
        raw_skills = skills_by_job.get(job["id"], [])
        norm_skills = normalize_skill_list(raw_skills)

        enriched.append(
            {
                "id": job["id"],
                "title": job["title"] or "",
                "description": job["description"] or "",
                "specialty": job["specialty_name"] or "",
                "category": job["category_name"] or "",
                "skills": raw_skills,
                "skills_norm": norm_skills,
                # numeric
                "budget_amount": float(job["budget_amount"] or 0.0),
                "experience_level": (job["experience_level"] or "").upper(),
                "screening_question_count": screening_by_job.get(job["id"], 0),
            }
        )

    return enriched


async def fetch_freelancers(session) -> List[dict]:
    """Trả về danh sách freelancer với title/bio + skill."""

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
        raw_skills = skills_by_fr.get(fr["user_id"], [])
        norm_skills = normalize_skill_list(raw_skills)

        enriched.append(
            {
                "id": fr["user_id"],
                "title": fr["title"] or "",
                "bio": fr["bio"] or "",
                "skills": raw_skills,
                "skills_norm": norm_skills,
                # region / timezone hiện chưa lấy được → để sau
            }
        )

    return enriched


async def fetch_invitation_stats(session) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Đọc job_invitation để build:
      - job_stats: job_stats_offers, job_stats_accepts
      - freelancer_stats: freelancer_stats_offers, freelancer_stats_accepts, freelancer_invite_accept_rate

    NOTE: đây là approximation, vì hiện tại mình chưa có bảng job_application / contract rõ ràng.
    """

    sql = text(
        """
        SELECT job_id, freelancer_id, status
        FROM job_invitation
        """
    )
    rows = (await session.execute(sql)).mappings().all()

    job_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "offers": 0,
        "accepts": 0,
        "applies": 0,  # hiện chưa có bảng apply → tạm = 0
    })
    fr_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "offers": 0,
        "accepts": 0,
        "applies": 0,  # hiện chưa có bảng apply → tạm = 0
        "total_invites": 0,
        "accepted_invites": 0,
    })

    for r in rows:
        job_id = r["job_id"]
        fid = r["freelancer_id"]
        status = (r["status"] or "").upper()

        job_stats[job_id]["offers"] += 1
        fr_stats[fid]["offers"] += 1
        fr_stats[fid]["total_invites"] += 1

        if status == "ACCEPTED":
            job_stats[job_id]["accepts"] += 1
            fr_stats[fid]["accepts"] += 1
            fr_stats[fid]["accepted_invites"] += 1
        # DECLINED/EXPIRED… không cộng accepts

    # tính invite_accept_rate cho freelancer
    for fid, s in fr_stats.items():
        total = s["total_invites"]
        if total > 0:
            s["invite_accept_rate"] = s["accepted_invites"] / total
        else:
            s["invite_accept_rate"] = None

    return job_stats, fr_stats


def build_domain_text(category: str, specialty: str) -> str:
    parts = [p for p in [category, specialty] if p]
    return " / ".join(parts)


# ----------------- EMBEDDINGS -----------------


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


# ----------------- FEATURE ENGINEERING -----------------


JOB_LEVEL_MAP = {
    "ENTRY": 1,
    "INTERMEDIATE": 2,
    "EXPERT": 3,
}


def estimate_freelancer_level(fr: dict) -> int:
    """
    Ước lượng "level" freelancer từ số skill:
      ≤3 skill  -> 1 (ENTRY)
      4-7 skill -> 2 (INTERMEDIATE)
      >7 skill  -> 3 (EXPERT)
    """
    n = len(fr.get("skills_norm", []))
    if n <= 3:
        return 1
    if n <= 7:
        return 2
    return 3


def compute_job_experience_level_num(job: dict) -> int:
    lv_str = (job.get("experience_level") or "INTERMEDIATE").upper()
    return JOB_LEVEL_MAP.get(lv_str, 2)


def compute_level_gap(job: dict, fr: dict) -> int:
    job_level = compute_job_experience_level_num(job)
    fr_level = estimate_freelancer_level(fr)
    return job_level - fr_level


def compute_budget_gap(job: dict, fr: dict) -> float:
    """
    Chênh lệch ngân sách job - cost ước lượng freelancer.
    Hiện tại không có rate freelancer → đặt 0, sau này update schema thì sửa lại đây.
    """
    budget = float(job.get("budget_amount") or 0.0)
    fr_cost_estimate = 0.0
    return budget - fr_cost_estimate


def compute_timezone_gap_hours(job: dict, fr: dict) -> int:
    """
    Hiện chưa có timezone trong schema → tạm thời cho = 0.
    Sau này lấy từ profile.country / timezone thì sửa lại ở đây.
    """
    return 0


def compute_skill_overlap(job: dict, fr: dict) -> Tuple[int, float]:
    """
    Đếm số skill trùng nhau + ratio so với skill job cần.
    """
    job_skills = set(job.get("skills_norm", []))
    fr_skills = set(fr.get("skills_norm", []))

    overlap = job_skills & fr_skills
    overlap_count = len(overlap)
    job_count = len(job_skills) or 1
    overlap_ratio = overlap_count / job_count
    return overlap_count, overlap_ratio


# ----------------- MAIN PIPELINE -----------------


async def main(model_name: str = DEFAULT_MODEL, top_k: int = 25):
    async with AsyncSessionLocal() as session:
        jobs = await fetch_jobs(session)
        freelancers = await fetch_freelancers(session)
        job_stats, fr_stats = await fetch_invitation_stats(session)

        print(f"Loaded {len(jobs)} job(s) & {len(freelancers)} freelancer(s)")
        print(f"Similarity weights: {DEFAULT_SIMILARITY_WEIGHTS}")

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

        job_top: Dict[str, List[tuple[str, float]]] = {}
        fr_top: Dict[str, List[tuple[str, float]]] = {}
        job_all_scores: Dict[str, List[tuple[str, float]]] = {}
        fr_all_scores: Dict[str, List[tuple[str, float]]] = {}

        # map để tra nhanh
        job_lookup = {j["id"]: j for j in jobs}
        fr_lookup = {f["id"]: f for f in freelancers}

        # 1) tính similarity cho mọi cặp
        for job in jobs:
            job_id = job["id"]
            scored: List[tuple[str, float]] = []
            for fr in freelancers:
                fr_id = fr["id"]
                sim = multi_embedding_similarity(
                    job_embs_map[job_id],
                    fr_embs_map[fr_id],
                    weights=DEFAULT_SIMILARITY_WEIGHTS,
                )
                if sim is None:
                    continue
                scored.append((fr_id, sim))
                fr_top.setdefault(fr_id, []).append((job_id, sim))

            scored.sort(key=lambda x: x[1], reverse=True)
            job_all_scores[job_id] = scored
            job_top[job_id] = scored[:top_k]

        for fr_id, pairs in fr_top.items():
            pairs.sort(key=lambda x: x[1], reverse=True)
            fr_all_scores[fr_id] = pairs
            fr_top[fr_id] = pairs[:top_k]

        # 2) Union của top-N mỗi job và top-N mỗi freelancer
        pairs_to_persist = {
            (job_id, fr_id): score
            for job_id, matches in job_top.items()
            for fr_id, score in matches
        }
        for fr_id, matches in fr_top.items():
            for job_id, score in matches:
                pairs_to_persist.setdefault((job_id, fr_id), score)

        # 3) Lưu vào match_feature
        for (job_id, fr_id), sim in pairs_to_persist.items():
            job = job_lookup[job_id]
            fr = fr_lookup[fr_id]

            budget_gap = compute_budget_gap(job, fr)
            level_gap = compute_level_gap(job, fr)
            tz_gap = compute_timezone_gap_hours(job, fr)
            job_level_num = compute_job_experience_level_num(job)
            job_skill_count = len(job.get("skills_norm", []))
            fr_skill_count = len(fr.get("skills_norm", []))
            overlap_count, overlap_ratio = compute_skill_overlap(job, fr)

            js = job_stats.get(job_id, {})
            fs = fr_stats.get(fr_id, {})

            await upsert_match_feature(
                session,
                job_id=job_id,
                freelancer_id=fr_id,
                # core similarity / gap
                similarity_score=sim,
                budget_gap=budget_gap,
                timezone_gap_hours=tz_gap,
                level_gap=level_gap,
                # job features
                job_experience_level_num=job_level_num,
                job_required_skill_count=job_skill_count,
                job_screening_question_count=job.get("screening_question_count", 0),
                job_stats_applies=js.get("applies"),
                job_stats_offers=js.get("offers"),
                job_stats_accepts=js.get("accepts"),
                # freelancer features
                freelancer_skill_count=fr_skill_count,
                freelancer_stats_applies=fs.get("applies"),
                freelancer_stats_offers=fs.get("offers"),
                freelancer_stats_accepts=fs.get("accepts"),
                freelancer_invite_accept_rate=fs.get("invite_accept_rate"),
                # pairwise features
                skill_overlap_count=overlap_count,
                skill_overlap_ratio=overlap_ratio,
                has_past_collaboration=False,      # chưa có bảng contract → tạm 0/False
                past_collaboration_count=0,
                has_viewed_job=False,              # chưa có bảng job_view → tạm False
            )

        # 4) In debug kết quả
        for job in jobs:
            job_id = job["id"]
            top_matches = job_top.get(job_id, [])
            rest_matches = job_all_scores.get(job_id, [])[top_k:]

            print("\n=== Job:", job["title"])
            print("Skills:", normalize_skill_list(job.get("skills", [])))
            print(f"Top {top_k} freelancers saved to match_feature:")
            for fr_id, score in top_matches:
                fr = fr_lookup.get(fr_id, {})
                print(
                    f"  - Freelancer: {fr.get('title') or fr_id} | score={score:.4f} | "
                    f"skills={normalize_skill_list(fr.get('skills', []))}"
                )

            if rest_matches:
                print("  Not selected:")
                for fr_id, score in rest_matches:
                    fr = fr_lookup.get(fr_id, {})
                    print(
                        f"    * Freelancer: {fr.get('title') or fr_id} | score={score:.4f} | "
                        f"skills={normalize_skill_list(fr.get('skills', []))}"
                    )

        for fr in freelancers:
            fr_id = fr["id"]
            top_jobs = fr_top.get(fr_id, [])
            rest_jobs = fr_all_scores.get(fr_id, [])[top_k:]

            print("\n=== Freelancer:", fr.get("title") or fr_id)
            print("Skills:", normalize_skill_list(fr.get("skills", [])))
            print(f"Top {top_k} job posts saved to match_feature:")
            for job_id, score in top_jobs:
                job = job_lookup.get(job_id, {})
                print(
                    f"  - Job: {job.get('title') or job_id} | score={score:.4f} | "
                    f"skills={normalize_skill_list(job.get('skills', []))}"
                )

            if rest_jobs:
                print("  Not selected:")
                for job_id, score in rest_jobs:
                    job = job_lookup.get(job_id, {})
                    print(
                        f"    * Job: {job.get('title') or job_id} | score={score:.4f} | "
                        f"skills={normalize_skill_list(job.get('skills', []))}"
                    )


if __name__ == "__main__":
    asyncio.run(main())
