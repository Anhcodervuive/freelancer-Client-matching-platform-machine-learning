# app/api/routes.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.crud import save_embedding
from app.db.session import get_session
from app.models.ml_models import embed_text

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
    # 1. Tính embedding từ text
    vector = embed_text(payload.text, model_name=payload.model)

    # 2. Lưu vào DB
    obj = await save_embedding(
        session,
        entity_type=payload.entity_type,
        entity_id=payload.entity_id,
        kind=payload.kind,
        model=payload.model,
        vector=vector,
    )

    return {"status": "ok", "id": obj.id}
