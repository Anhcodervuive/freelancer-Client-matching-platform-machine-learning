# app/features/similarity.py
from typing import Dict, Sequence, Optional
import math

def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def multi_embedding_similarity(
    job_embs: Dict[str, Sequence[float]],
    fr_embs: Dict[str, Sequence[float]],
    weights: Dict[str, float] | None = None,
) -> Optional[float]:
    """
    Kết hợp FULL / SKILLS / DOMAIN thành 1 similarity_score.

    job_embs, fr_embs: {"FULL": [...], "SKILLS": [...], "DOMAIN": [...]}
    weights:            {"FULL": 0.6, "SKILLS": 0.3, "DOMAIN": 0.1}
    """
    if weights is None:
        weights = {"FULL": 0.2, "SKILLS": 0.6, "DOMAIN": 0.2}

    sims: list[float] = []
    w_used: list[float] = []

    for kind, w in weights.items():
        v_job = job_embs.get(kind)
        v_fr = fr_embs.get(kind)
        if v_job is None or v_fr is None:
            continue
        consinSimilary = cosine_similarity(v_job, v_fr)
        print(kind, consinSimilary)
        sims.append(consinSimilary)
        w_used.append(w)

    if not sims:
        return None

    # weighted average
    total_w = sum(w_used)
    return sum(s * w for s, w in zip(sims, w_used)) / total_w
