import math
import random
from datetime import date, timedelta

from app.ml.locations import Location
from app.ml.predictor import Prediction

LATEST_DATE = date(2023, 12, 25)  # 지점별 CSV의 마지막 주
WEEK = timedelta(weeks=1)
DEFAULT_WEEKS = 12


def _seasonal_log(location: Location, week: date) -> float:
    """여름에 높고 겨울에 낮은 곡선. 지점마다 높이와 정점 시기가 조금씩 다르다."""
    rng = random.Random(location.code)
    base = rng.uniform(1.5, 2.5)
    amplitude = rng.uniform(2.0, 3.5)
    peak_day = rng.uniform(210, 240)  # 7월 말 ~ 8월 말

    day = week.timetuple().tm_yday
    return base + amplitude * math.exp(-(((day - peak_day) / 45) ** 2))


class MockPredictor:
    name = "mock"

    def predict(self, location: Location, target_date: date | None) -> Prediction:
        target = target_date or LATEST_DATE
        # 같은 지점·같은 주에는 항상 같은 숫자가 나오게 한다
        rng = random.Random(f"{location.code}-{target.isoformat()}")

        q50 = _seasonal_log(location, target) + rng.uniform(-0.25, 0.25)
        spread = rng.uniform(0.2, 0.6)

        return Prediction(
            target_date=target,
            input_until=target - WEEK,
            q20=max(q50 - spread, 0.0),
            q50=max(q50, 0.0),
            q80=min(q50 + spread, 8.0),
        )


class MockObservations:
    def weeks(
        self, location: Location, start: date | None, end: date | None
    ) -> list[date]:
        last = min(end or LATEST_DATE, LATEST_DATE)
        first = start or last - WEEK * (DEFAULT_WEEKS - 1)

        week = LATEST_DATE
        while week > last:
            week -= WEEK

        weeks = []
        while week >= first:
            weeks.append(week)
            week -= WEEK
        return weeks[::-1]

    def cyanocell(self, location: Location, week: date) -> float | None:
        if week > LATEST_DATE:
            return None
        rng = random.Random(f"obs-{location.code}-{week.isoformat()}")
        return max(_seasonal_log(location, week) + rng.uniform(-0.3, 0.3), 0.0)
