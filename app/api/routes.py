# app/api/routes.py
from fastapi import APIRouter, Depends, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.crud import save_embedding
from app.db.session import get_session
from app.models.ml_models import embed_text

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
    text: str          # đoạn text để embed


@router.post("/update_embedding")
async def update_embedding(
    payload: UpdateEmbeddingRequest,
    session: AsyncSession = Depends(get_session),
):
    # 1. Tính embedding từ text (y như cũ)
    vector = embed_text(payload.text, model_name=payload.model)

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
