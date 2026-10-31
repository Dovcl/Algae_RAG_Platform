"""연구실에서 받은 CSV로 관측값과 저장된 예측값을 읽는다.

모델을 다시 돌리지 않는다. 지금 자료로 재추론하면 당시 저장값과 달라지므로(재현성 전달 문서)
2023년 저장값을 그대로 쓴다. 저장값은 중앙값뿐이라 q20, q80은 None이다.
"""

import csv
import math
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

from app.ml.locations import Location
from app.ml.predictor import Prediction

WEEK = timedelta(weeks=1)


@lru_cache
def _read_observed(path: Path, column: str) -> dict[date, float | None]:
    """주별 남조류 세포수. 값은 log10(1 + cells/mL), 빈 칸은 보간하지 않고 None으로 둔다."""
    with path.open(encoding="cp949", newline="") as file:
        return {
            date.fromisoformat(row["Date"]): (
                math.log10(1 + float(row[column])) if row[column] else None
            )
            for row in csv.DictReader(file)
        }


@lru_cache
def _read_forecast(directory: Path) -> dict[date, float]:
    """예측 대상 주별 저장 예측 중앙값. Date는 입력 시점이 아니라 대상 주다."""
    path = next(directory.glob("*_values.csv"))
    with path.open(encoding="utf-8-sig", newline="") as file:
        return {
            date.fromisoformat(row["Date"]): float(row["Forecasted"])
            for row in csv.DictReader(file)
        }


@lru_cache
def _read_rows(path: Path) -> tuple[dict, ...]:
    """주별 전체 열. 빈 칸은 빈 값 그대로 둔다."""
    with path.open(encoding="cp949", newline="") as file:
        return tuple(csv.DictReader(file))


class CsvObservations:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, location: Location) -> Path:
        river = location.river
        return self.root / "DATA" / river / f"{location.code}_{river.lower()}_total.csv"

    def _series(self, location: Location) -> dict[date, float | None]:
        return _read_observed(self._path(location), f"Cyanocell_{location.code}")

    def weeks(
        self, location: Location, start: date | None, end: date | None, count: int
    ) -> list[date]:
        weeks = sorted(self._series(location))
        if end is not None:
            weeks = [week for week in weeks if week <= end]
        if start is not None:
            return [week for week in weeks if week >= start]
        return weeks[-count:]

    def cyanocell(self, location: Location, week: date) -> float | None:
        return self._series(location).get(week)

    def recent(self, location: Location, base: date, count: int) -> list[dict]:
        """기준일까지의 최근 주별 자료. 기준일 뒤의 행은 절대 돌려주지 않는다."""
        rows = _read_rows(self._path(location))
        past = [row for row in rows if date.fromisoformat(row["Date"]) <= base]
        return [dict(row) for row in past[-count:]]


class PrecomputedPredictor:
    name = "precomputed"

    def __init__(self, root: Path) -> None:
        self.root = root

    def predict(
        self, location: Location, target_date: date | None
    ) -> Prediction | None:
        directory = (
            self.root
            / "new_reproducibility_files"
            / location.river
            / f"{location.code}_final"
        )
        forecast = _read_forecast(directory)
        target = target_date or max(forecast)
        if target not in forecast:
            return None
        return Prediction(
            target_date=target,
            input_until=target - WEEK,
            q20=None,
            q50=forecast[target],
            q80=None,
        )
