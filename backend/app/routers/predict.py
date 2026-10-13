from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.ml.locations import LOCATIONS, RIVER_NAMES, Location
from app.ml.observations import Observations, get_observations
from app.ml.predictor import (
    Prediction,
    Predictor,
    classify_stage,
    get_predictor,
    to_cells,
)

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


class TimeseriesPoint(BaseModel):
    date: date
    observed: float | None
    predicted: Quantiles
    stage: int


class TimeseriesResponse(BaseModel):
    location: str
    backend: str
    points: list[TimeseriesPoint]


def _find_location(code: str) -> Location:
    location = LOCATIONS.get(code)
    if location is None:
        raise HTTPException(status_code=404, detail=f"모르는 지점입니다: {code}")
    return location


def _to_quantiles(result: Prediction) -> Quantiles:
    return Quantiles(
        q20=round(to_cells(result.q20), 1),
        q50=round(to_cells(result.q50), 1),
        q80=round(to_cells(result.q80), 1),
    )


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
    location = _find_location(request.location)
    result = predictor.predict(location, request.target_date)

    return PredictResponse(
        location=location.code,
        target_date=result.target_date,
        input_until=result.input_until,
        cells_per_ml=_to_quantiles(result),
        stage=classify_stage(result.q50),
        backend=predictor.name,
    )


@router.get("/timeseries")
def timeseries(
    location: str,
    predictor: Annotated[Predictor, Depends(get_predictor)],
    observations: Annotated[Observations, Depends(get_observations)],
    start: date | None = None,
    end: date | None = None,
    weeks: Annotated[int, Query(ge=1, le=520)] = 12,
) -> TimeseriesResponse:
    """지점의 주별 관측값과 그 주에 대한 예측값.

    start를 생략하면 end(생략 시 자료의 마지막 주)에서 거슬러 weeks주를 돌려준다.
    """
    found = _find_location(location)
    if start is not None and end is not None and start > end:
        raise HTTPException(status_code=400, detail="start가 end보다 늦습니다.")

    points = []
    for week in observations.weeks(found, start, end, weeks):
        observed = observations.cyanocell(found, week)
        result = predictor.predict(found, week)
        points.append(
            TimeseriesPoint(
                date=week,
                observed=None if observed is None else round(to_cells(observed), 1),
                predicted=_to_quantiles(result),
                stage=classify_stage(result.q50),
            )
        )

    return TimeseriesResponse(
        location=found.code, backend=predictor.name, points=points
    )
