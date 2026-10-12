from fastapi import APIRouter

from app.routers import health, predict

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(predict.router)
