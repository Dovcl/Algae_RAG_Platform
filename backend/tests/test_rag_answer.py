import json
from datetime import date

import pytest

from app.ml.locations import LOCATIONS
from app.ml.mock import MockObservations, MockPredictor
from app.rag import answer as rag
from app.rag.retrieve import Hit


def make_hit(number: int) -> Hit:
    return Hit(
        chunk_id=f"c{number}",
        document_id="DOC02",
        document="매뉴얼",
        publisher="환경부",
        page=30,
        pdf_page=34,
        section="1.3",
        table_id="표 6",
        text="경계 10,000 이상",
        distance=0.3,
    )


@pytest.fixture
def stubbed(monkeypatch):
    calls = {}

    def fake_ollama(system, user):
        calls["user"] = user
        return json.dumps(calls["output"], ensure_ascii=False)

    monkeypatch.setattr(rag, "ask_ollama", fake_ollama)
    monkeypatch.setattr(rag, "search", lambda *_: [make_hit(1), make_hit(2)])
    return calls


def run(location=None, base=None):
    return rag.build_answer(
        None, "질문", location, base, MockPredictor(), MockObservations()
    )


def test_keeps_only_given_evidence(stubbed):
    stubbed["output"] = {
        "answer": "제공된 근거에 따라 정리한 충분히 긴 답변 문장입니다.",
        "management_actions": [],
        "evidence_ids": ["E2", "E9"],
        "limitations": "",
    }

    result = run()

    assert result.evidence_ids == ["E2"]
    assert "E9" in result.limitations


def test_renumbers_citations_for_the_screen(stubbed):
    stubbed["output"] = {
        "answer": "근거에 따라 이렇게 정리해 답을 드립니다 [E2] 추가 설명 [E1].",
        "management_actions": [],
        "evidence_ids": ["E2"],
        "limitations": "",
    }

    assert run().answer == "근거에 따라 이렇게 정리해 답을 드립니다 [1] 추가 설명."


def test_no_location_has_no_numbers(stubbed):
    stubbed["output"] = {
        "answer": "제공된 근거에 따라 정리한 충분히 긴 답변 문장입니다.",
        "management_actions": [],
        "evidence_ids": [],
        "limitations": "",
    }

    result = run()

    assert result.forecast is None
    assert result.stage is None
    assert "관측 자료가 없습니다" in stubbed["user"]


def test_forecast_section_has_stage_range(stubbed):
    stubbed["output"] = {
        "answer": "제공된 근거에 따라 정리한 충분히 긴 답변 문장입니다.",
        "management_actions": [],
        "evidence_ids": [],
        "limitations": "",
    }

    result = run(LOCATIONS["CS"], date(2023, 6, 12))

    assert result.forecast.target_date == date(2023, 6, 19)
    assert "이 구간명만 쓴다" in stubbed["user"]
    assert result.answer.endswith(stubbed["output"]["answer"])
    assert "예측은 공식 발령이 아닙니다." in result.answer


def test_bad_output_raises(stubbed, monkeypatch):
    monkeypatch.setattr(rag, "ask_ollama", lambda *_: "형식이 틀린 답")

    with pytest.raises(rag.LlmError):
        run()


def test_cut_off_answer_raises(stubbed):
    stubbed["output"] = {
        "answer": "현재 상태:",
        "management_actions": [],
        "evidence_ids": [],
        "limitations": "",
    }

    with pytest.raises(rag.LlmError):
        run()
