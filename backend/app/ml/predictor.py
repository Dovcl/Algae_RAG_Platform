from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Protocol

from app.core.config import settings
from app.ml.locations import Location


@dataclass(frozen=True)
class Prediction:
    """모델 출력 그대로의 값. 분위수는 log10(1 + cells/mL)."""

    target_date: date
    input_until: date
    q20: float
    q50: float
    q80: float


class Predictor(Protocol):
    name: str

    def predict(self, location: Location, target_date: date | None) -> Prediction: ...


def classify_stage(log_value: float) -> int:
    """단계 분류. 기준은 DEV_PLAN.md 5장 (바꾸지 않는다)."""
    if log_value < 3:
        return 0
    if log_value < 4:
        return 1
    if log_value < 5:
        return 2
    if log_value < 6:
        return 3
    return 4


def to_cells(log_value: float) -> float:
    """log10(1 + x) 값을 cells/mL로 되돌린다. 상한 8로 자른다."""
    return 10 ** min(log_value, 8.0) - 1


@lru_cache
def get_predictor() -> Predictor:
    if settings.predict_backend == "mock":
        from app.ml.mock import MockPredictor

        return MockPredictor()
    raise NotImplementedError("PREDICT_BACKEND=tft는 아직 구현 전입니다.")
