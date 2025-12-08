# app/features/dataset_p_freelancer_accept.py

"""
Dataset builder (in-memory) cho mô hình p_freelancer_accept.

Label:
    1  nếu invitation.status = 'ACCEPTED'
    0  nếu invitation.status ∈ {'DECLINED', 'EXPIRED'}

Feature X đang dùng (9 chiều):

    [
        similarity_score,
        budget_gap,
        timezone_gap_hours,
        level_gap,
        job_experience_level_num,
        job_required_skill_count,
        freelancer_skill_count,
        skill_overlap_count,
        skill_overlap_ratio,
    ]
"""

from typing import List, Tuple
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import JobInvitation, MatchFeature

POSITIVE_STATUSES = {"ACCEPTED"}
NEGATIVE_STATUSES = {"DECLINED", "EXPIRED"}


def _label_from_status(status: str) -> int | None:
    if status in POSITIVE_STATUSES:
        return 1
    if status in NEGATIVE_STATUSES:
        return 0
    return None  # SENT / WITHDRAWN thì bỏ


async def build_p_freelancer_accept_dataset(
    session: AsyncSession,
) -> Tuple[np.ndarray, np.ndarray]:
    stmt = (
        select(JobInvitation, MatchFeature)
        .join(
            MatchFeature,
            (MatchFeature.job_id == JobInvitation.job_id)
            & (MatchFeature.freelancer_id == JobInvitation.freelancer_id),
            isouter=True,
        )
    )
    result = await session.execute(stmt)
    rows: List[tuple[JobInvitation, MatchFeature | None]] = result.all()

    X: list[list[float]] = []
    y: list[int] = []

    for inv, mf in rows:
        label = _label_from_status(inv.status)
        if label is None:
            continue

        if mf is not None:
            feats = [
                float(mf.similarityScore or 0.0),
                float(mf.budget_gap or 0.0),
                float(mf.timezone_gap_hours or 0.0),
                float(mf.level_gap or 0.0),
                float(mf.job_experience_level_num or 0.0),
                float(mf.job_required_skill_count or 0.0),
                float(mf.freelancer_skill_count or 0.0),
                float(mf.skill_overlap_count or 0.0),
                float(mf.skill_overlap_ratio or 0.0),
            ]
        else:
            feats = [0.0] * 9

        X.append(feats)
        y.append(label)

    if not X:
        raise RuntimeError("Không có sample nào cho p_freelancer_accept")

    return np.array(X, dtype=float), np.array(y, dtype=int)
