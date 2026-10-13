def ask(client, **body):
    return client.post("/api/chat", json=body)


def test_chat_response_shape(client):
    response = ask(client, question="다음 주 녹조 전망을 알려줘")

    assert response.status_code == 200
    body = response.json()
    assert body["llm"] == "mock"
    assert body["predict_backend"] == "mock"
    assert body["answer"]
    assert body["limitations"]
    assert body["management_actions"]
    assert body["risk_level"] == body["forecast_status"]["stage"]
    evidence = body["evidence"][0]
    assert {"id", "document", "publisher", "page", "section"} <= evidence.keys()


def test_chat_default_is_latest_week(client):
    body = ask(client, question="다음 주 녹조 전망을 알려줘").json()

    assert body["current_status"]["date"] == "2023-12-18"
    assert body["forecast_status"]["target_date"] == "2023-12-25"
    assert body["forecast_status"]["input_until"] == "2023-12-18"


def test_chat_finds_location_and_date_in_question(client):
    body = ask(client, question="HCB 지점의 2023-08-07 기준 전망과 대응은?").json()

    assert body["location"] == "HCB"
    assert body["current_status"]["date"] == "2023-08-07"
    assert body["forecast_status"]["target_date"] == "2023-08-14"


def test_chat_prefers_longer_location_code(client):
    body = ask(client, question="GJGRB 지점 전망").json()

    assert body["location"] == "GJGRB"


def test_chat_explicit_fields_win(client):
    body = ask(
        client,
        question="HCB 지점의 2023-08-07 기준 전망",
        location="CGB",
        base_date="2023-07-03",
    ).json()

    assert body["location"] == "CGB"
    assert body["current_status"]["date"] == "2023-07-03"


def test_chat_matches_predict(client):
    chat = ask(client, question="전망", location="CGB", base_date="2023-07-03").json()
    single = client.post(
        "/api/predict", json={"location": "CGB", "target_date": "2023-07-10"}
    ).json()

    assert chat["forecast_status"]["cells_per_ml"] == single["cells_per_ml"]
    assert chat["forecast_status"]["stage"] == single["stage"]


def test_chat_same_request_same_result(client):
    first = ask(client, question="HCB 지점 전망").json()
    second = ask(client, question="HCB 지점 전망").json()

    assert first == second


def test_chat_empty_question(client):
    assert ask(client, question="").status_code == 422


def test_chat_unknown_location(client):
    assert ask(client, question="전망", location="XXX").status_code == 404
