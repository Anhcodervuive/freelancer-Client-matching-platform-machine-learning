# app/api/routes.py
from fastapi import APIRouter
from app.db.crud import save_embedding
from app.db.models import Embedding
from app.db.crud import async_session

router = APIRouter()

@router.post("/update_embedding")
async def update_embedding(entity_type: str, entity_id: str, kind: str, model: str, vector: list):
    async with async_session() as session:
        obj = await save_embedding(session, entity_type, entity_id, kind, model, vector)
        return {"status": "ok", "id": obj.id}
