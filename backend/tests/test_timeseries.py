from datetime import date, timedelta
from itertools import pairwise


def test_timeseries_default_is_latest_12_weeks(client):
    response = client.get("/api/timeseries", params={"location": "CGB"})

    assert response.status_code == 200
    body = response.json()
    assert body["location"] == "CGB"
    assert body["backend"] == "mock"
    dates = [date.fromisoformat(point["date"]) for point in body["points"]]
    assert len(dates) == 12
    assert dates[-1] == date(2023, 12, 25)
    assert all(b - a == timedelta(weeks=1) for a, b in pairwise(dates))


def test_timeseries_range(client):
    response = client.get(
        "/api/timeseries",
        params={"location": "CGB", "start": "2023-05-01", "end": "2023-07-10"},
    )

    points = response.json()["points"]
    assert len(points) == 11
    assert points[0]["date"] == "2023-05-01"
    assert points[-1]["date"] == "2023-07-10"


def test_timeseries_point_shape(client):
    response = client.get("/api/timeseries", params={"location": "GJB"})

    for point in response.json()["points"]:
        predicted = point["predicted"]
        assert predicted["q20"] <= predicted["q50"] <= predicted["q80"]
        assert 0 <= point["stage"] <= 4
        assert point["observed"] is None or point["observed"] >= 0


def test_timeseries_matches_predict(client):
    series = client.get(
        "/api/timeseries",
        params={"location": "CGB", "start": "2023-07-10", "end": "2023-07-10"},
    ).json()
    single = client.post(
        "/api/predict", json={"location": "CGB", "target_date": "2023-07-10"}
    ).json()

    assert series["points"][0]["predicted"] == single["cells_per_ml"]
    assert series["points"][0]["stage"] == single["stage"]


def test_timeseries_unknown_location(client):
    response = client.get("/api/timeseries", params={"location": "XXX"})

    assert response.status_code == 404


def test_timeseries_start_after_end(client):
    response = client.get(
        "/api/timeseries",
        params={"location": "CGB", "start": "2023-07-10", "end": "2023-05-01"},
    )

    assert response.status_code == 400


def test_timeseries_weeks(client):
    response = client.get("/api/timeseries", params={"location": "CGB", "weeks": 8})

    points = response.json()["points"]
    assert len(points) == 8
    assert points[-1]["date"] == "2023-12-25"


def test_timeseries_weeks_must_be_positive(client):
    response = client.get("/api/timeseries", params={"location": "CGB", "weeks": 0})

    assert response.status_code == 422
