from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.ml.locations import LOCATIONS
from app.ml.observations import Observations, get_observations
from app.ml.predictor import Predictor, get_predictor, to_cells
from app.rag.mock import build_answer
from app.routers.predict import Quantiles, to_quantiles

router = APIRouter(tags=["Chat"])


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # 실험에서는 시나리오의 지점과 기준일을 직접 넘긴다. 생략하면 질문에서 찾는다.
    location: str | None = None
    base_date: date | None = None


class CurrentStatus(BaseModel):
    date: date
    cells_per_ml: float


class ForecastStatus(BaseModel):
    target_date: date
    input_until: date
    cells_per_ml: Quantiles
    stage: int


class Evidence(BaseModel):
    id: str
    document: str
    publisher: str
    page: int
    section: str


class ChatResponse(BaseModel):
    """필드 이름은 실험 설계안의 출력 형식을 따른다 (DEV_PLAN.md 13장)."""

    question: str
    location: str | None
    current_status: CurrentStatus | None
    forecast_status: ForecastStatus | None
    risk_level: int | None
    management_actions: list[str]
    evidence: list[Evidence]
    limitations: str | None
    answer: str
    llm: str
    predict_backend: str


@router.post("/chat")
def chat(
    request: ChatRequest,
    predictor: Annotated[Predictor, Depends(get_predictor)],
    observations: Annotated[Observations, Depends(get_observations)],
) -> ChatResponse:
    """질문에 답한다. 지금은 LLM_PROVIDER=mock만 동작한다."""
    if settings.llm_provider != "mock":
        raise HTTPException(
            status_code=501,
            detail=f"LLM_PROVIDER={settings.llm_provider}는 아직 구현 전입니다.",
        )

    location = None
    if request.location is not None:
        location = LOCATIONS.get(request.location)
        if location is None:
            raise HTTPException(
                status_code=404, detail=f"모르는 지점입니다: {request.location}"
            )

    result = build_answer(
        request.question, location, request.base_date, predictor, observations
    )

    return ChatResponse(
        question=request.question,
        location=result.location.code,
        current_status=(
            None
            if result.observed is None
            else CurrentStatus(
                date=result.base_date,
                cells_per_ml=round(to_cells(result.observed)),
            )
        ),
        forecast_status=ForecastStatus(
            target_date=result.forecast.target_date,
            input_until=result.forecast.input_until,
            cells_per_ml=to_quantiles(result.forecast),
            stage=result.stage,
        ),
        risk_level=result.stage,
        management_actions=result.actions,
        evidence=[Evidence(**vars(item)) for item in result.evidence],
        limitations=result.limitations,
        answer=result.answer,
        llm=settings.llm_provider,
        predict_backend=predictor.name,
    )
