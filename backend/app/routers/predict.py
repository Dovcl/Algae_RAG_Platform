from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.ml.locations import LOCATIONS, RIVER_NAMES
from app.ml.predictor import Predictor, classify_stage, get_predictor, to_cells

router = APIRouter(tags=["Predict"])


class LocationOut(BaseModel):
    code: str
    river: str
    river_name: str


class PredictRequest(BaseModel):
    location: str
    target_date: date | None = None


class Quantiles(BaseModel):
    q20: float
    q50: float
    q80: float


class PredictResponse(BaseModel):
    location: str
    target_date: date
    input_until: date
    cells_per_ml: Quantiles
    stage: int
    backend: str


@router.get("/locations")
def list_locations() -> list[LocationOut]:
    """예측 모델이 있는 지점 목록."""
    return [
        LocationOut(code=loc.code, river=loc.river, river_name=RIVER_NAMES[loc.river])
        for loc in LOCATIONS.values()
    ]


@router.post("/predict")
def predict(
    request: PredictRequest,
    predictor: Annotated[Predictor, Depends(get_predictor)],
) -> PredictResponse:
    """지점의 남조류 세포수를 예측한다. target_date를 생략하면 자료의 마지막 주."""
    location = LOCATIONS.get(request.location)
    if location is None:
        raise HTTPException(
            status_code=404, detail=f"모르는 지점입니다: {request.location}"
        )

    result = predictor.predict(location, request.target_date)

    return PredictResponse(
        location=location.code,
        target_date=result.target_date,
        input_until=result.input_until,
        cells_per_ml=Quantiles(
            q20=round(to_cells(result.q20), 1),
            q50=round(to_cells(result.q50), 1),
            q80=round(to_cells(result.q80), 1),
        ),
        stage=classify_stage(result.q50),
        backend=predictor.name,
    )
