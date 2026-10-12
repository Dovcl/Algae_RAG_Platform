import pytest

from app.ml.predictor import classify_stage, to_cells


def test_locations(client):
    response = client.get("/api/locations")

    assert response.status_code == 200
    locations = response.json()
    assert len(locations) == 21
    assert {"code": "CGB", "river": "NAK", "river_name": "낙동강"} in locations


def test_predict(client):
    response = client.post(
        "/api/predict", json={"location": "CGB", "target_date": "2023-07-10"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["location"] == "CGB"
    assert body["target_date"] == "2023-07-10"
    assert body["input_until"] == "2023-07-03"
    assert body["backend"] == "mock"
    assert 0 <= body["stage"] <= 4
    cells = body["cells_per_ml"]
    assert cells["q20"] <= cells["q50"] <= cells["q80"]


def test_predict_same_request_same_result(client):
    request = {"location": "GJB", "target_date": "2023-08-07"}

    first = client.post("/api/predict", json=request).json()
    second = client.post("/api/predict", json=request).json()

    assert first == second


def test_predict_default_date(client):
    response = client.post("/api/predict", json={"location": "CGB"})

    assert response.json()["target_date"] == "2023-12-25"


def test_predict_unknown_location(client):
    response = client.post("/api/predict", json={"location": "XXX"})

    assert response.status_code == 404


def test_predict_invalid_date(client):
    response = client.post(
        "/api/predict", json={"location": "CGB", "target_date": "어제"}
    )

    assert response.status_code == 422


# 단계 기준은 DEV_PLAN.md 5장. 이 테스트가 깨지면 기준을 건드린 것이다.
@pytest.mark.parametrize(
    ("log_value", "stage"),
    [(0, 0), (2.99, 0), (3, 1), (3.99, 1), (4, 2), (5, 3), (5.99, 3), (6, 4), (8, 4)],
)
def test_classify_stage(log_value, stage):
    assert classify_stage(log_value) == stage


def test_to_cells():
    assert to_cells(0) == 0
    assert to_cells(3) == 999
    assert to_cells(9) == to_cells(8)  # 상한 8로 자른다
