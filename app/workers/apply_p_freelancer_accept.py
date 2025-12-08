# app/workers/apply_p_freelancer_accept.py

"""
Dùng model logreg_p_freelancer_accept.pkl để fill cột p_freelancer_accept
cho toàn bộ bảng match_feature.

Feature X (20 chiều, phải khớp với train_p_freelancer_accept.py):

[
    similarity_score,
    budget_gap,
    timezone_gap_hours,
    level_gap,

    job_experience_level_num,
    job_required_skill_count,
    job_screening_question_count,
    job_stats_applies,
    job_stats_offers,
    job_stats_accepts,

    freelancer_skill_count,
    freelancer_stats_applies,
    freelancer_stats_offers,
    freelancer_stats_accepts,
    freelancer_invite_accept_rate,

    skill_overlap_count,
    skill_overlap_ratio,
    has_past_collaboration,
    past_collaboration_count,
    has_viewed_job,
]
"""

import asyncio
from typing import List

from sqlalchemy import select

from app.db.session import async_session
from app.db.models import MatchFeature
from app.models.ml_models import predict_p_freelancer_accept


def _build_features_from_match_feature(mf: MatchFeature) -> list[float]:
    """Chuyển 1 row MatchFeature -> list[float] 20 phần tử."""

    def f(v) -> float:
        return float(v) if v is not None else 0.0

    def b(v) -> float:
        # bool/None -> 0.0/1.0
        return 1.0 if v else 0.0

    return [
        # ---- core similarity / gap ----
        f(mf.similarity_score),
        f(mf.budget_gap),
        f(mf.timezone_gap_hours),
        f(mf.level_gap),

        # ---- job features ----
        f(mf.job_experience_level_num),
        f(mf.job_required_skill_count),
        f(mf.job_screening_question_count),
        f(mf.job_stats_applies),
        f(mf.job_stats_offers),
        f(mf.job_stats_accepts),

        # ---- freelancer features ----
        f(mf.freelancer_skill_count),
        f(mf.freelancer_stats_applies),
        f(mf.freelancer_stats_offers),
        f(mf.freelancer_stats_accepts),
        f(mf.freelancer_invite_accept_rate),

        # ---- pairwise features ----
        f(mf.skill_overlap_count),
        f(mf.skill_overlap_ratio),
        b(mf.has_past_collaboration),
        f(mf.past_collaboration_count),
        b(mf.has_viewed_job),
    ]


async def recompute_p_freelancer_accept_for_all() -> None:
    """
    Load toàn bộ match_feature, tính p_freelancer_accept bằng model ML,
    rồi ghi lại vào DB.
    """
    async with async_session() as session:
        stmt = select(MatchFeature)
        rows: List[MatchFeature] = (await session.execute(stmt)).scalars().all()

        if not rows:
            print("[apply_p_freelancer_accept] Không có row match_feature nào.")
            return

        print(f"[apply_p_freelancer_accept] Found {len(rows)} match_feature rows.")

        updated = 0
        for mf in rows:
            feats = _build_features_from_match_feature(mf)
            prob = predict_p_freelancer_accept(feats)
            mf.p_freelancer_accept = prob
            updated += 1

        await session.commit()
        print(f"[apply_p_freelancer_accept] Updated p_freelancer_accept for {updated} rows.")


if __name__ == "__main__":
    asyncio.run(recompute_p_freelancer_accept_for_all())
