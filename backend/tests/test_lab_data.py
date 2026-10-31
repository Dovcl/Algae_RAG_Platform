import math
from datetime import date

import pytest

from app.ml.lab_data import CsvObservations, PrecomputedPredictor
from app.ml.locations import LOCATIONS
from app.ml.observations import get_observations
from app.ml.predictor import get_predictor

CS = LOCATIONS["CS"]


@pytest.fixture
def lab_root(tmp_path):
    data = tmp_path / "DATA" / "NAK"
    data.mkdir(parents=True)
    (data / "CS_nak_total.csv").write_text(
        "Date,WT_CS,Cyanocell_CS\n"
        "2023-01-02,4.1,99\n"
        "2023-01-09,4.2,\n"
        "2023-01-16,4.3,999\n"
        "2023-01-23,4.4,9999\n",
        encoding="cp949",
    )
    saved = tmp_path / "new_reproducibility_files" / "NAK" / "CS_final"
    saved.mkdir(parents=True)
    (saved / "model_values.csv").write_text(
        "Date,Observed,Forecasted\n2023-01-16,3.0,3.5\n2023-01-23,4.0,4.5\n",
        encoding="utf-8",
    )
    return tmp_path


def test_observed_is_log_scale_and_keeps_missing(lab_root):
    observations = CsvObservations(lab_root)

    assert observations.cyanocell(CS, date(2023, 1, 2)) == pytest.approx(2.0)
    assert observations.cyanocell(CS, date(2023, 1, 9)) is None
    assert observations.cyanocell(CS, date(2022, 1, 3)) is None


def test_weeks_counts_back_from_end(lab_root):
    observations = CsvObservations(lab_root)

    weeks = observations.weeks(CS, None, date(2023, 1, 20), 2)

    assert weeks == [date(2023, 1, 9), date(2023, 1, 16)]


def test_weeks_start_ignores_count(lab_root):
    observations = CsvObservations(lab_root)

    weeks = observations.weeks(CS, date(2023, 1, 9), None, 1)

    assert weeks == [date(2023, 1, 9), date(2023, 1, 16), date(2023, 1, 23)]


def test_recent_stops_at_base_date(lab_root):
    rows = CsvObservations(lab_root).recent(CS, date(2023, 1, 16), 2)

    assert [row["Date"] for row in rows] == ["2023-01-09", "2023-01-16"]


def test_predict_reads_saved_median(lab_root):
    result = PrecomputedPredictor(lab_root).predict(CS, date(2023, 1, 23))

    assert result is not None
    assert result.q50 == 4.5
    assert result.input_until == date(2023, 1, 16)
    assert result.q20 is None
    assert result.q80 is None


def test_predict_default_is_last_saved_week(lab_root):
    result = PrecomputedPredictor(lab_root).predict(CS, None)

    assert result is not None
    assert result.target_date == date(2023, 1, 23)


def test_predict_without_saved_week(lab_root):
    assert PrecomputedPredictor(lab_root).predict(CS, date(2023, 1, 9)) is None


@pytest.fixture
def lab_client(client, lab_root):
    client.app.dependency_overrides[get_predictor] = lambda: PrecomputedPredictor(
        lab_root
    )
    client.app.dependency_overrides[get_observations] = lambda: CsvObservations(
        lab_root
    )
    yield client
    client.app.dependency_overrides.clear()


def test_predict_api_without_saved_week(lab_client):
    response = lab_client.post(
        "/api/predict", json={"location": "CS", "target_date": "2023-01-09"}
    )

    assert response.status_code == 404


def test_timeseries_skips_weeks_without_prediction(lab_client):
    response = lab_client.get("/api/timeseries", params={"location": "CS"})

    points = response.json()["points"]
    assert [point["date"] for point in points] == ["2023-01-16", "2023-01-23"]
    assert points[-1]["observed"] == 9999
    assert points[-1]["predicted"]["q50"] == round(10**4.5 - 1)
    assert points[-1]["predicted"]["q20"] is None


def test_chat_forecast_uses_next_week(lab_client):
    response = lab_client.post(
        "/api/chat",
        json={"question": "전망", "location": "CS", "base_date": "2023-01-16"},
    )

    body = response.json()
    assert body["current_status"]["cells_per_ml"] == 999
    assert body["forecast_status"]["target_date"] == "2023-01-23"
    assert body["predict_backend"] == "precomputed"
    assert math.isclose(body["forecast_status"]["cells_per_ml"]["q50"], 31622)


def test_chat_without_forecast(lab_client):
    response = lab_client.post(
        "/api/chat",
        json={"question": "전망", "location": "CS", "base_date": "2023-01-23"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["forecast_status"] is None
    assert body["risk_level"] is None
    assert body["management_actions"] == []
