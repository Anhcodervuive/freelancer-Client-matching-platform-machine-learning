# app/main.py
from fastapi import FastAPI
from app.api.routes import router as api_router

app = FastAPI(title="ML Service")

app.include_router(api_router, prefix="/api")

@app.get("/health")
def health_check():
    return {"status": "ok"}
