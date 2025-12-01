# app/main.py
from fastapi import FastAPI
from app.api.routes import router as api_router
from app.db.session import engine 

app = FastAPI(title="ML Service")

app.include_router(api_router, prefix="/api")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.on_event("shutdown")
async def on_shutdown() -> None:
    # Đảm bảo đóng connection pool của aiomysql trước khi loop đóng
    await engine.dispose()