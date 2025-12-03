# app/api/routes.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.crud import save_embedding
from app.db.session import get_session
from app.models.ml_models import embed_text
from app.features.skill_processing import aggregate_skill_embedding

from app.workers.tasks import (
    schedule_recompute_for_job,
    schedule_recompute_for_freelancer,
)

router = APIRouter()


class UpdateEmbeddingRequest(BaseModel):
    entity_type: str   # "JOB" | "FREELANCER"
    entity_id: str     # id của job hoặc freelancer từ Prisma
    kind: str          # "FULL" | "SKILLS" | "DOMAIN"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    text: Optional[str] = None      # đoạn text để embed
    skills: Optional[List[str]] = None  # list skill (ưu tiên dùng khi kind="SKILLS")


@router.post("/update_embedding")
async def update_embedding(
    payload: UpdateEmbeddingRequest,
    session: AsyncSession = Depends(get_session),
):
    # 1. Tính embedding
    if payload.kind == "SKILLS" and payload.skills:
        vector = aggregate_skill_embedding(payload.skills, model_name=payload.model)
        if vector is None:
            raise HTTPException(status_code=400, detail="Skill list trống sau khi chuẩn hoá")
    elif payload.text:
        vector = embed_text(payload.text, model_name=payload.model)
    else:
        raise HTTPException(status_code=400, detail="Thiếu text hoặc skills để embed")

    # 2. Lưu vào DB (y như cũ)
    obj = await save_embedding(
        session,
        entity_type=payload.entity_type,
        entity_id=payload.entity_id,
        kind=payload.kind,
        model=payload.model,
        vector=vector,
    )

    # 3. 🔥 Sau khi lưu xong thì bắn background job update MatchFeature
    #    - chỉ dùng embedding kind="FULL" để match (đúng ý mình bàn)
    if payload.entity_type == "JOB":
        # job này vừa được update embedding -> tính lại 100–200 freelancer top
        schedule_recompute_for_job(payload.entity_id, top_n=200)
    elif payload.entity_type == "FREELANCER":
        # freelancer này vừa được update embedding -> tính lại 100–200 job top
        schedule_recompute_for_freelancer(payload.entity_id, top_n=200)

    return {"status": "ok", "id": obj.id}
