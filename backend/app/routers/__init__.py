from fastapi import APIRouter

from app.routers import chat, health, predict

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(predict.router)
api_router.include_router(chat.router)
