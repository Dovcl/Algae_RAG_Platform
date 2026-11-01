import json
from datetime import date

import pytest

from app.ml.locations import LOCATIONS
from app.ml.mock import MockObservations, MockPredictor
from app.rag import answer as rag
from app.rag import retrieve
from app.rag.mock import find_location
from app.rag.retrieve import Hit, is_criteria_question


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
    monkeypatch.setattr(rag, "search_evidence", lambda *_: [make_hit(1), make_hit(2)])
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


def test_splits_grouped_citations(stubbed):
    stubbed["output"] = {
        "answer": "근거에 따라 이렇게 정리해 답을 드립니다 [E1, E2].",
        "management_actions": [],
        "evidence_ids": ["E1", "E2"],
        "limitations": "",
    }

    assert run().answer.endswith("[1][2].")


def test_removes_html_line_breaks(stubbed):
    stubbed["output"] = {
        "answer": "상수원 구간 기준을 먼저 정리합니다.<br><br>친수활동 구간 기준은 다릅니다.",
        "management_actions": [],
        "evidence_ids": [],
        "limitations": "",
    }

    assert "<br>" not in run().answer


def test_status_and_llm_text_are_separate_paragraphs(stubbed):
    stubbed["output"] = {
        "answer": "수질 변화를 먼저 정리합니다.\n \n\n대응의 요지를 이어서 정리합니다.",
        "management_actions": [],
        "evidence_ids": [],
        "limitations": "",
    }

    paragraphs = run(LOCATIONS["CS"], date(2023, 6, 12)).answer.split("\n\n")

    assert len(paragraphs) == 3
    assert "cells/mL" in paragraphs[0]
    assert paragraphs[1:] == [
        "수질 변화를 먼저 정리합니다.",
        "대응의 요지를 이어서 정리합니다.",
    ]


def test_cleans_bold_and_inline_numbering(stubbed):
    stubbed["output"] = {
        "answer": "**상수원 구간** 대응입니다: 1) 차단막을 설치합니다. 2) 정수를 강화합니다.",
        "management_actions": [],
        "evidence_ids": [],
        "limitations": "",
    }

    assert run().answer == (
        "상수원 구간 대응입니다:\n1) 차단막을 설치합니다.\n2) 정수를 강화합니다."
    )


def test_tidy_text_splits_long_paragraph_every_two_sentences():
    text = "첫 문장입니다. [1] 둘째 문장입니다. 셋째 문장입니다."

    assert (
        rag.tidy_text(text) == "첫 문장입니다. [1] 둘째 문장입니다.\n\n셋째 문장입니다."
    )


def test_tidy_text_counts_sentence_with_citation_before_period():
    text = "첫 문장입니다 [1]. 둘째 문장입니다 [2]. 셋째 문장입니다."

    assert rag.tidy_text(text) == (
        "첫 문장입니다 [1]. 둘째 문장입니다 [2].\n\n셋째 문장입니다."
    )


def test_tidy_text_keeps_lists_and_joins_lone_citations():
    text = "기준입니다:\n\n- 상수원: 1,000 미만입니다.\n\n[1], [2]"

    assert rag.tidy_text(text) == "기준입니다:\n\n- 상수원: 1,000 미만입니다. [1][2]"


def test_tidy_text_removes_comma_left_by_dropped_citation():
    assert (
        rag.tidy_text("강화해야 합니다 (, [3] 참조).") == "강화해야 합니다 ([3] 참조)."
    )


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


@pytest.mark.parametrize(
    ("question", "code"),
    [
        ("칠서의 전망은?", "CS"),
        ("강정고령보 상태", "GJGRB"),
        ("강정고령 상태", "GJGR"),
        ("cgb 지점", "CGB"),
    ],
)
def test_find_location(question, code):
    assert find_location(question).code == code


def test_base_date_moves_to_monday(stubbed):
    stubbed["output"] = {
        "answer": "제공된 근거에 따라 정리한 충분히 긴 답변 문장입니다.",
        "management_actions": [],
        "evidence_ids": [],
        "limitations": "",
    }

    assert run(LOCATIONS["CS"], date(2023, 6, 15)).base_date == date(2023, 6, 12)


def test_unknown_zone_is_disclosed(stubbed):
    stubbed["output"] = {
        "answer": "제공된 근거에 따라 정리한 충분히 긴 답변 문장입니다.",
        "management_actions": [],
        "evidence_ids": [],
        "limitations": "",
    }

    result = run(LOCATIONS["CGB"], date(2023, 6, 12))

    assert "구간(상수원/친수활동)은 확인되지 않아" in result.limitations


def test_river_question_asks_for_site(monkeypatch):
    monkeypatch.setattr(
        rag, "ask_ollama", lambda *_: pytest.fail("LLM을 부르면 안 된다")
    )

    result = rag.build_answer(
        None,
        "낙동강 2023-06-12 녹조값 알려줘",
        None,
        date(2023, 6, 12),
        MockPredictor(),
        MockObservations(),
    )

    assert "칠서" in result.answer
    assert "세종보" not in result.answer
    assert result.evidence == []


@pytest.mark.parametrize(
    "question",
    ["낙동강에서 경계 단계 발령 기준은?", "조류예측 정보는 얼마나 자주 발표되나?"],
)
def test_guideline_question_needs_no_site(question):
    assert rag.site_request(question) is None


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("조류경보 해제 기준은?", True),
        ("경계 단계 발령 기준은?", True),
        ("경보는 언제 해제돼?", True),
        ("조류 예측 1단계 기준은?", False),
        ("경계 단계에서 취수장은 뭘 해야 해?", False),
    ],
)
def test_is_criteria_question(question, expected):
    assert is_criteria_question(question) is expected


def test_criteria_table_comes_first_without_duplicates(monkeypatch):
    table = [make_hit(1), make_hit(2)]
    monkeypatch.setattr(retrieve, "search", lambda *_: [make_hit(3), make_hit(1)])
    monkeypatch.setattr(retrieve, "table_chunks", lambda *_: table)

    hits = retrieve.search_evidence(None, "해제 기준은?", "해제 기준은?", 6)

    assert [hit.chunk_id for hit in hits] == ["c1", "c2", "c3"]
