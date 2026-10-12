import random
from datetime import date, timedelta

from app.ml.locations import Location
from app.ml.predictor import Prediction

LATEST_DATE = date(2023, 12, 25)  # 지점별 CSV의 마지막 주


class MockPredictor:
    name = "mock"

    def predict(self, location: Location, target_date: date | None) -> Prediction:
        target = target_date or LATEST_DATE
        # 같은 지점·같은 주에는 항상 같은 숫자가 나오게 한다
        rng = random.Random(f"{location.code}-{target.isoformat()}")

        is_summer = 6 <= target.month <= 10
        q50 = rng.uniform(3.0, 5.5) if is_summer else rng.uniform(0.5, 3.0)
        spread = rng.uniform(0.2, 0.6)

        return Prediction(
            target_date=target,
            input_until=target - timedelta(weeks=1),
            q20=max(q50 - spread, 0.0),
            q50=q50,
            q80=min(q50 + spread, 8.0),
        )
