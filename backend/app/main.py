from fastapi import FastAPI

from app.routers import api_router

app = FastAPI(title="Algae RAG Platform API")
app.include_router(api_router)
