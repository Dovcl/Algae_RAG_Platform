from datetime import date
from functools import lru_cache
from typing import Protocol

from app.core.config import settings
from app.ml.locations import Location


class Observations(Protocol):
    def weeks(
        self, location: Location, start: date | None, end: date | None, count: int
    ) -> list[date]: ...

    def cyanocell(self, location: Location, week: date) -> float | None: ...


@lru_cache
def get_observations() -> Observations:
    if settings.predict_backend == "mock":
        from app.ml.mock import MockObservations

        return MockObservations()
    raise NotImplementedError("실제 관측 자료 조회는 아직 구현 전입니다.")
