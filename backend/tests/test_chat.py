from app.routers import chat as chat_router


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


class FollowUpStub:
    """연속 질문 경로가 어떤 값으로 답변을 부르는지 기록한다."""

    def __init__(self, monkeypatch, follow_up=True, rewritten="대응은?"):
        self.called = None
        self.rewrites = 0
        stub = self

        def fake_rewrite(*_):
            stub.rewrites += 1
            return rewritten

        def fake_build(_session, question, location, base, *_):
            stub.called = (question, location, base)
            raise chat_router.LlmError("여기서 멈춘다")

        monkeypatch.setattr(chat_router.settings, "llm_provider", "ollama")
        monkeypatch.setattr(chat_router, "is_follow_up", lambda *_: follow_up)
        monkeypatch.setattr(chat_router, "rewrite_question", fake_rewrite)
        monkeypatch.setattr(chat_router, "build_rag_answer", fake_build)


PAST = {
    "question": "칠서 전망은?",
    "answer": "칠서 답변",
    "location": "CS",
    "base_date": "2023-06-12",
}


def test_follow_up_keeps_previous_site_and_date(client, monkeypatch):
    stub = FollowUpStub(monkeypatch)

    ask(client, question="그럼 대응은?", history=[PAST])

    question, location, base = stub.called
    assert question == "대응은?"
    assert location.code == "CS"
    assert base.isoformat() == "2023-06-12"


def test_rewritten_date_does_not_replace_previous_base_date(client, monkeypatch):
    stub = FollowUpStub(monkeypatch, rewritten="칠서의 2023-06-19 주 대응은?")

    ask(client, question="그럼 대응은?", history=[PAST])

    assert stub.called[2].isoformat() == "2023-06-12"


def test_follow_up_with_new_site_uses_that_site(client, monkeypatch):
    stub = FollowUpStub(monkeypatch)

    ask(client, question="그럼 해평은?", history=[PAST])

    assert stub.called[1].code == "HP"


def test_independent_question_ignores_history(client, monkeypatch):
    stub = FollowUpStub(monkeypatch, follow_up=False)

    ask(client, question="DO가 뭐야?", history=[PAST])

    assert stub.called == ("DO가 뭐야?", None, None)
    assert stub.rewrites == 0


def test_no_history_never_rewrites(client, monkeypatch):
    stub = FollowUpStub(monkeypatch)

    ask(client, question="그럼 대응은?")

    assert stub.called == ("그럼 대응은?", None, None)
    assert stub.rewrites == 0
