from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    """
    서버 상태를 확인하는 엔드포인트.

    Docker, 모니터링 도구, 프론트엔드에서 서버 상태 확인에 사용한다.
    """
    return {"status": "ok"}
