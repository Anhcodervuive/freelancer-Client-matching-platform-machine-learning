# app/workers/tasks.py
from celery import Celery
from app.db.crud import async_session, update_match_feature
from app.models.ml_models import predict_p_match, predict_p_accept
from app.db.models import Embedding
from app.config import CELERY_BROKER_URL
import asyncio

celery_app = Celery("ml_tasks", broker=CELERY_BROKER_URL)

@celery_app.task
def compute_match_features(job_ids: list, freelancer_ids: list):
    async def run_batch():
        async with async_session() as session:
            # Dummy: lấy embedding từ DB
            for job_id in job_ids:
                for freelancer_id in freelancer_ids:
                    # Giả sử lấy embedding list
                    job_emb = [0.1,0.2,0.3]
                    freelancer_emb = [0.1,0.2,0.3]
                    p_match = predict_p_match(job_emb, freelancer_emb)
                    p_accept = predict_p_accept(job_emb, freelancer_emb)
                    await update_match_feature(session, job_id, freelancer_id, p_match, p_accept)
    asyncio.run(run_batch())
