from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_db

router = APIRouter()


@router.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    """
    서버 상태를 확인하는 엔드포인트.

    Docker, 모니터링 도구, 프론트엔드에서 서버 상태 확인에 사용한다.
    """
    return {"status": "ok"}


@router.get("/health/db", tags=["System"])
def db_health_check(db: Annotated[Session, Depends(get_db)]) -> dict[str, str]:
    """DB 연결 상태를 확인하는 엔드포인트."""
    db.execute(text("SELECT 1"))
    return {"status": "ok"}
