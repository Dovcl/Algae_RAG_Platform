from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import ChatLog
from app.db.session import get_db
from app.ml.locations import LOCATIONS, Location
from app.ml.observations import Observations, get_observations
from app.ml.predictor import Predictor, get_predictor, to_cells
from app.rag.answer import LlmError
from app.rag.answer import build_answer as build_rag_answer
from app.rag.mock import build_answer, find_date, find_location
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


def _ollama_chat(
    request: ChatRequest,
    location: Location | None,
    predictor: Predictor,
    observations: Observations,
    session: Session,
) -> ChatResponse:
    question = request.question
    try:
        result = build_rag_answer(
            session,
            question,
            location or find_location(question),
            request.base_date or find_date(question),
            predictor,
            observations,
        )
    except LlmError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    forecast = result.forecast
    response = ChatResponse(
        question=question,
        location=None if result.location is None else result.location.code,
        current_status=(
            None
            if result.observed is None or result.base_date is None
            else CurrentStatus(
                date=result.base_date,
                cells_per_ml=round(to_cells(result.observed)),
            )
        ),
        forecast_status=(
            None
            if forecast is None or result.stage is None
            else ForecastStatus(
                target_date=forecast.target_date,
                input_until=forecast.input_until,
                cells_per_ml=to_quantiles(forecast),
                stage=result.stage,
            )
        ),
        risk_level=result.stage,
        management_actions=result.actions,
        evidence=[
            Evidence(
                id=number,
                document=hit.document,
                publisher=hit.publisher,
                page=hit.page or hit.pdf_page,
                section=hit.table_id or hit.section or "",
            )
            for number, hit in zip(result.evidence_ids, result.evidence, strict=True)
        ],
        limitations=result.limitations,
        answer=result.answer,
        llm=settings.llm_provider,
        predict_backend=predictor.name,
    )
    session.add(
        ChatLog(
            question=question,
            answer=result.answer,
            llm=settings.llm_provider,
            llm_model=settings.llm_model,
            predict_backend=predictor.name,
            prompt=result.prompt,
            raw_output=result.raw_output,
            final_response=response.model_dump(mode="json"),
        )
    )
    session.commit()
    return response


@router.post("/chat")
def chat(
    request: ChatRequest,
    predictor: Annotated[Predictor, Depends(get_predictor)],
    observations: Annotated[Observations, Depends(get_observations)],
    session: Annotated[Session, Depends(get_db)],
) -> ChatResponse:
    """질문에 답한다. LLM_PROVIDER=ollama면 문서를 검색해 실제 모델로 답한다."""
    location = None
    if request.location is not None:
        location = LOCATIONS.get(request.location)
        if location is None:
            raise HTTPException(
                status_code=404, detail=f"모르는 지점입니다: {request.location}"
            )

    if settings.llm_provider == "ollama":
        return _ollama_chat(request, location, predictor, observations, session)

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
        forecast_status=(
            None
            if result.forecast is None or result.stage is None
            else ForecastStatus(
                target_date=result.forecast.target_date,
                input_until=result.forecast.input_until,
                cells_per_ml=to_quantiles(result.forecast),
                stage=result.stage,
            )
        ),
        risk_level=result.stage,
        management_actions=result.actions,
        evidence=[Evidence(**vars(item)) for item in result.evidence],
        limitations=result.limitations,
        answer=result.answer,
        llm=settings.llm_provider,
        predict_backend=predictor.name,
    )
