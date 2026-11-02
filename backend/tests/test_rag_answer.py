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

    def fake_ollama(system, user, **options):
        calls["user"] = user
        calls.setdefault("options", []).append(options)
        if "retry_output" in calls and len(calls["options"]) == 1:
            return "잘린 출력"
        return json.dumps(calls["output"], ensure_ascii=False)

    monkeypatch.setattr(rag, "ask_ollama", fake_ollama)
    monkeypatch.setattr(rag, "search_evidence", lambda *_: [make_hit(1), make_hit(2)])
    monkeypatch.setattr(rag, "table_chunks", lambda *_: [])
    monkeypatch.setattr(rag, "classify_route", lambda *_: "document")
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

    assert run().answer == "근거에 따라 이렇게 정리해 답을 드립니다 [1] 추가 설명 [2]."


def test_splits_grouped_citations(stubbed):
    stubbed["output"] = {
        "answer": "근거에 따라 이렇게 정리해 답을 드립니다 [E1, E2].",
        "management_actions": [],
        "evidence_ids": ["E1", "E2"],
        "limitations": "",
    }

    assert run().answer.endswith("[1][2].")


def test_removes_leaked_field_name(stubbed):
    stubbed["output"] = {
        "answer": "근거에 따라 이렇게 정리해 답을 드립니다.\n\n[evidence_ids]: [E1], [E2]",
        "management_actions": [],
        "evidence_ids": ["E1", "E2"],
        "limitations": "",
    }

    answer = run().answer

    assert "evidence_ids" not in answer
    assert answer.endswith("[1][2]")


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


def test_removes_number_prefix_from_actions(stubbed):
    stubbed["output"] = {
        "answer": "제공된 근거에 따라 정리한 충분히 긴 답변 문장입니다.",
        "management_actions": ["1) 시료 채취", "2. 정수 처리 강화"],
        "evidence_ids": [],
        "limitations": "",
    }

    assert run().actions == ["시료 채취", "정수 처리 강화"]


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


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("경계 단계가 뭐야?", True),
        ("조류경보 해제 기준은 무엇인가?", True),
        ("취수장은 뭘 해야 해?", True),
        ("수면관리자는 어떻게 돼?", True),
        ("활성탄을 더 넣으면 어떻게 돼?", True),
        ("2024년 녹조가 많았던 원인은?", True),
        ("칠서는 왜 세포수가 늘었어?", True),
        ("낙동강에서 녹조가 많은 이유는?", True),
        ("산소가 부족하다는건 수중에 DO가 줄어든다는건가?", False),
        ("클로로필-a 농도가 증가하면 어떻게 돼?", False),
    ],
)
def test_needs_document(question, expected):
    assert rag.needs_document(question) is expected


@pytest.mark.parametrize("question", ["지금 위험해?", "세포수가 왜 줄었지?"])
def test_site_cues_skip_llm_when_site_is_selected(monkeypatch, question):
    monkeypatch.setattr(rag, "classify_route", pytest.fail)

    assert rag.is_general_question(question, has_site=True) is False


def test_general_route_comes_from_llm(monkeypatch):
    monkeypatch.setattr(rag, "classify_route", lambda *_: "general")
    assert rag.is_general_question("DO가 뭐야?", has_site=True) is True

    monkeypatch.setattr(rag, "classify_route", lambda *_: "site")
    assert rag.is_general_question("왜 늘었어?", has_site=True) is False


def test_classify_route_falls_back_to_document(monkeypatch):
    def broken(*_, **__):
        raise rag.LlmError("실패")

    monkeypatch.setattr(rag, "ask_ollama", broken)

    assert rag.classify_route("DO가 뭐야?", False) == "document"


def test_concept_question_skips_documents(stubbed, monkeypatch):
    monkeypatch.setattr(rag, "classify_route", lambda *_: "general")

    def fail(*_):
        raise AssertionError("개념 질문은 문서를 검색하지 않는다")

    monkeypatch.setattr(rag, "search_evidence", fail)
    stubbed["output"] = {
        "answer": "중앙값은 값을 크기순으로 놓았을 때 가운데 값입니다 [1].",
        "management_actions": ["지어낸 조치"],
        "evidence_ids": ["E1"],
        "limitations": "",
    }

    result = rag.build_answer(
        None, "중앙값이 뭐야?", None, None, MockPredictor(), MockObservations()
    )

    assert result.answer == "중앙값은 값을 크기순으로 놓았을 때 가운데 값입니다."
    assert result.actions == []
    assert result.evidence == []
    assert result.limitations == ""


def test_retries_with_repeat_penalty_when_output_is_cut(stubbed):
    stubbed["retry_output"] = True
    stubbed["output"] = {
        "answer": "제공된 근거에 따라 정리한 충분히 긴 답변 문장입니다.",
        "management_actions": [],
        "evidence_ids": [],
        "limitations": "",
    }

    run()

    assert stubbed["options"] == [{}, {"repeat_penalty": rag.RETRY_REPEAT_PENALTY}]


def test_uses_citations_written_only_in_text(stubbed):
    stubbed["output"] = {
        "answer": "근거에 따라 정리한 충분히 긴 답변 문장입니다 [E2].",
        "management_actions": [],
        "evidence_ids": [],
        "limitations": "",
    }

    result = run()

    assert result.evidence_ids == ["E2"]
    assert result.answer.endswith("문장입니다 [1].")


def test_adds_missing_citations_to_end(stubbed):
    stubbed["output"] = {
        "answer": "근거에 따라 정리한 충분히 긴 답변 문장입니다.",
        "management_actions": [],
        "evidence_ids": ["E1", "E2"],
        "limitations": "",
    }

    assert run().answer.endswith("문장입니다.[1][2]")


@pytest.mark.parametrize(
    "label",
    [
        "management_actions",
        "[management_actions]",
        "**management_actions**",
        "관리조치",
    ],
)
def test_removes_actions_leaked_into_answer(stubbed, label):
    stubbed["output"] = {
        "answer": f"근거에 따라 정리한 충분히 긴 답변 문장입니다.\n\n{label}: 1. 지어낸 조치",
        "management_actions": [],
        "evidence_ids": [],
        "limitations": "",
    }

    assert "지어낸" not in run().answer


def test_asks_again_when_criteria_answer_has_one_zone(stubbed):
    stubbed["output"] = {
        "answer": "상수원 구간은 10,000 이상일 때 경계 단계입니다.",
        "management_actions": [],
        "evidence_ids": [],
        "limitations": "",
    }

    rag.build_answer(
        None, "얼마면 경계 단계야?", None, None, MockPredictor(), MockObservations()
    )

    assert stubbed["user"].endswith(rag.ZONE_REMINDER)


def test_removes_evidence_marks_from_actions_and_limitations(stubbed):
    stubbed["output"] = {
        "answer": "근거에 따라 정리한 충분히 긴 답변 문장입니다.",
        "management_actions": ["시료 채취 [E1]"],
        "evidence_ids": [],
        "limitations": "자료가 부족합니다 [E2, E3 참고].",
    }

    result = run()

    assert result.actions == ["시료 채취"]
    assert result.limitations == "자료가 부족합니다."


def test_limits_actions_and_says_so(stubbed):
    stubbed["output"] = {
        "answer": "제공된 근거에 따라 정리한 충분히 긴 답변 문장입니다.",
        "management_actions": [f"조치 {n}" for n in range(1, 8)],
        "evidence_ids": [],
        "limitations": "",
    }

    result = run()

    assert len(result.actions) == rag.MAX_ACTIONS
    assert "5개까지만" in result.limitations


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("남조류가 얼마면 경계 단계야?", True),
        ("경계 단계가 뭐야?", True),
        ("경계 단계 조류 예측은 얼마야?", False),
        ("클로로필-a가 뭐야?", False),
    ],
)
def test_is_criteria_question_covers_value_questions(question, expected):
    assert is_criteria_question(question) is expected


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
    monkeypatch.setattr(rag, "ask_ollama", lambda *_, **__: "형식이 틀린 답")

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


def test_question_phrases_drop_particles():
    phrases = retrieve.question_phrases("댐과 보의 방류량 조정은 어느 기관이?")

    assert "방류량 조정" in phrases
    assert "보 방류량" not in phrases  # 한 글자 단어는 표현이 되지 못한다


def test_phrase_chunks_go_first_and_are_capped(monkeypatch):
    hits = [make_hit(n) for n in range(1, 6)]
    monkeypatch.setattr(retrieve, "search", lambda *_: hits[:2])
    monkeypatch.setattr(retrieve, "phrase_chunks", lambda *_: hits[1:5])

    found = retrieve.search_evidence(None, "방류량 조정은?", "방류량 조정은?", 6)

    assert [hit.chunk_id for hit in found] == [
        hit.chunk_id for hit in [hits[1], hits[2], hits[3], hits[0]]
    ]


CRITERIA_ROWS = {
    "상수원": [
        (
            "관심",
            "2회 연속 채취 시 남조류 세포수가 1,000 세포/mL 이상 10,000 세포/mL 미만인 경우",
        ),
        (
            "경계",
            "2회 연속 채취 시 남조류 세포수가 10,000 세포/mL 이상 1,000,000 세포/mL 미만인 경우",
        ),
        (
            "조류 대발생",
            "2회 연속 채취 시 남조류 세포수가 1,000,000 세포/mL 이상인 경우",
        ),
        ("해제", "2회 연속 채취 시 남조류 세포수가 1,000 세포/mL 미만인 경우"),
    ],
    "친수활동": [
        (
            "관심",
            "2회 연속 채취 시 남조류 세포수가 20,000 세포/mL 이상 100,000 세포/mL 미만인 경우",
        ),
        ("경계", "2회 연속 채취 시 남조류 세포수가 100,000 세포/mL 이상인 경우"),
        ("해제", "2회 연속 채취 시 남조류 세포수가 20,000 세포/mL 미만인 경우"),
    ],
}


def criteria_hits() -> list[Hit]:
    return [
        Hit(
            chunk_id=f"c{zone}",
            document_id="DOC02",
            document="문서",
            publisher="기관",
            page=30,
            pdf_page=34,
            section="1.3.2",
            table_id="표 6",
            text="\n".join(
                f"구간: {zone}구간 | 단계: {stage} | 발령·해제 기준: {value}"
                for stage, value in rows
            ),
            distance=0.0,
        )
        for zone, rows in CRITERIA_ROWS.items()
    ]


def test_release_criteria_come_from_the_table(monkeypatch):
    monkeypatch.setattr(rag, "table_chunks", lambda *_: criteria_hits())

    result = rag.criteria_answer(None, "조류경보 해제 기준은?")

    assert (
        "상수원 구간: 2회 연속 채취 시 남조류 세포수가 1,000 세포/mL 미만"
        in result.answer
    )
    assert (
        "친수활동 구간: 2회 연속 채취 시 남조류 세포수가 20,000 세포/mL 미만"
        in result.answer
    )
    assert "10,000" not in result.answer
    assert len(result.evidence) == 2


def test_stage_criteria_keep_the_upper_limit(monkeypatch):
    monkeypatch.setattr(rag, "table_chunks", lambda *_: criteria_hits())

    result = rag.criteria_answer(None, "남조류가 얼마면 경계 단계야?")

    assert "10,000 세포/mL 이상 1,000,000 세포/mL 미만" in result.answer
    assert "100,000 세포/mL 이상" in result.answer
    assert "관심" not in result.answer


def test_criteria_keep_the_asked_zone_only(monkeypatch):
    monkeypatch.setattr(rag, "table_chunks", lambda *_: criteria_hits())

    result = rag.criteria_answer(None, "친수활동 구간 해제 기준은?")

    assert "20,000" in result.answer
    assert "상수원" not in result.answer
    assert [hit.chunk_id for hit in result.evidence] == ["c친수활동"]


def test_criteria_say_when_a_zone_has_no_such_stage(monkeypatch):
    monkeypatch.setattr(rag, "table_chunks", lambda *_: criteria_hits())

    result = rag.criteria_answer(None, "조류대발생 발령 기준은?")

    assert "1,000,000 세포/mL 이상" in result.answer
    assert "친수활동 구간에는 이 단계의 기준이 없습니다." in result.answer


def test_criteria_fall_back_without_table(monkeypatch):
    monkeypatch.setattr(rag, "table_chunks", lambda *_: [])

    assert rag.criteria_answer(None, "조류경보 해제 기준은?") is None


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("조류경보 발령 기준은?", True),
        ("남조류가 얼마면 경계 단계야?", True),
        ("경보는 언제 해제돼?", True),
        ("남조류 세포수가 한 번만 기준을 넘으면 경보를 발령하는가?", False),
        ("경계 단계에서 취수장은 뭘 해야 해?", False),
        ("조류경보를 어기면 벌금이 얼마야?", False),
        ("조류경보제 연간 예산은 얼마야?", False),
        ("일본의 조류경보 기준은 어떻게 돼?", False),
        ("2030년에 조류경보 기준은 어떻게 바뀌어?", False),
    ],
)
def test_asks_criteria_value(question, expected):
    assert rag.asks_criteria_value(question) is expected


def action_hit() -> Hit:
    rows = [
        "단계: 경계 | 관계 기관: 4대강 물환경연구소장 (보건환경연구원장 또는 수면관리자) | 조치사항: 1) 주 2회 시료 채취",
        "단계: 경계 | 관계 기관: 수면관리자 (수면관리자) | 조치사항: 차단막 설치",
        "단계: 경계 | 관계 기관: 취수장·정수장 관리자 (취수장·정수장 관리자) | 조치사항: 1) 취수구 이동 2) 정수처리 강화 3) 정수의 독소분석 실시",
    ]
    return Hit(
        chunk_id="c8",
        document_id="DOC02",
        document="문서",
        publisher="기관",
        page=33,
        pdf_page=37,
        section="1.4.1.2",
        table_id="표 8",
        text="\n".join(["열: 단계 / 관계 기관 / 조치사항", *rows]),
        distance=0.0,
    )


def test_action_answer_uses_only_the_asked_role(monkeypatch):
    monkeypatch.setattr(rag, "table_chunks", lambda *_: [action_hit()])

    result = rag.action_answer(
        None, "경계 단계에서 취수장과 정수장 관리자가 해야 할 일은?"
    )

    assert result.actions == ["취수구 이동", "정수처리 강화", "정수의 독소분석 실시"]
    assert "상수원" in result.limitations


def test_action_answer_ignores_role_in_brackets(monkeypatch):
    monkeypatch.setattr(rag, "table_chunks", lambda *_: [action_hit()])

    result = rag.action_answer(None, "경계 단계에서 수면관리자는 뭘 해?")

    assert result.actions == ["차단막 설치"]


@pytest.mark.parametrize(
    "question",
    [
        "정수장에서 지오스민이 20 ng/L를 넘으면 어떻게 대응하나?",
        "관심과 경계 단계에서 수면관리자는 뭘 해?",
        "경계 단계 기준은 얼마야? 수면관리자 말고",
    ],
)
def test_action_target_needs_one_stage(question):
    assert rag.action_target(question) is None


def test_action_answer_falls_back_when_row_is_missing(monkeypatch):
    monkeypatch.setattr(rag, "table_chunks", lambda *_: [])

    assert rag.action_answer(None, "경계 단계에서 수면관리자는 뭘 해?") is None


def forecast_hit() -> Hit:
    rows = [
        "열: 구분 / 예측 값 / 예측 값",
        "구분: 구분 | 예측 값: 상수원 구간 | 예측 값: 친수활동 구간",
        "구분: 0 단 계 | 예측 값: 1,000세포/mL 미만 | 예측 값: 20,000세포/mL 미만",
        "구분: 1 단 계(관심*) | 예측 값: 1,000 ~ 10,000세포/mL 미만 | 예측 값: 20,000 ~ 100,000세포/mL 미만",
        "구분: 2 단 계(경계*) | 예측 값: 10,000 ~ 100,000세포/mL 미만 | 예측 값: 100,000세포/mL 이상",
        "구분: 3 단 계 | 예측 값: 100,000 ~ 1,000,000세포/mL 미만 | 예측 값: -",
    ]
    return Hit(
        chunk_id="c7",
        document_id="DOC01",
        document="문서",
        publisher="기관",
        page=154,
        pdf_page=174,
        section="예측",
        table_id="표 3-1-7",
        text="\n".join(rows),
        distance=0.0,
    )


def test_forecast_stage_uses_table_rows(monkeypatch):
    monkeypatch.setattr(rag, "table_chunks", lambda *_: [forecast_hit()])

    result = rag.forecast_answer(None, "조류 예측 1단계와 2단계의 세포수 구간은?")

    assert "1단계(관심): 상수원 구간 1,000 ~ 10,000세포/mL 미만" in result.answer
    assert "친수활동 구간 20,000 ~ 100,000세포/mL 미만" in result.answer
    assert "3단계" not in result.answer


def test_forecast_stage_marks_missing_zone(monkeypatch):
    monkeypatch.setattr(rag, "table_chunks", lambda *_: [forecast_hit()])

    result = rag.forecast_answer(None, "조류 예측 3단계 세포수 구간은?")

    assert "친수활동 구간 해당 없음" in result.answer


def test_forecast_frequency_is_left_to_the_llm():
    assert rag.forecast_answer(None, "조류예측 정보는 얼마나 자주 발표되나?") is None


def test_drop_restated_removes_only_status_sentences():
    text = "유해남조류 세포수는 22,114 cells/mL로 경계 단계에 근접합니다. 취수구 조치를 강화합니다. [1]"

    assert rag.drop_restated(text) == "취수구 조치를 강화합니다. [1]"


def test_drop_restated_keeps_quantile_notice():
    text = "20%와 80% 분위수 값은 제공된 자료로 알 수 없습니다."

    assert rag.drop_restated(text) == text


def test_clean_limitations_cuts_broken_markup():
    text = "**구간별 기준이 다릅니다.** 조정 필요 -></span></pre>`}**주의:**"

    assert rag.clean_limitations(text) == "구간별 기준이 다릅니다. 조정 필요"


def test_broken_evidence_id_is_not_shown(stubbed):
    stubbed["output"] = {
        "answer": "근거에 따라 정리한 충분히 긴 답변 문장입니다.",
        "management_actions": [],
        "evidence_ids": ["E1", "E2}] }  <!-- 문단 구분 -->"],
        "limitations": "",
    }

    assert "<!--" not in run().limitations


def test_bracket_actions_without_colon_are_removed(stubbed):
    stubbed["output"] = {
        "answer": "근거에 따라 정리한 충분히 긴 답변 문장입니다.\n\n[management_actions]\n1. 조치",
        "management_actions": ["조치"],
        "evidence_ids": [],
        "limitations": "",
    }

    assert "management_actions" not in run().answer


def test_leaked_action_examples_are_removed(stubbed):
    stubbed["output"] = {
        "answer": "근거에 따라 정리한 충분히 긴 답변 문장입니다.\n\n[관리조치 예시]:\n1. 정기 점검",
        "management_actions": [],
        "evidence_ids": [],
        "limitations": "",
    }

    assert "관리조치" not in run().answer


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("1정기적인 수질 모니터링", "정기적인 수질 모니터링"),
        ("1. 시료 채취", "시료 채취"),
        ("2회 이상 시료 채취", "2회 이상 시료 채취"),
        ("* 경계 경보 발령 검토", "경계 경보 발령 검토"),
        ("- 오염원 단속 강화", "오염원 단속 강화"),
        ("1 취수구 위치 조정", "취수구 위치 조정"),
    ],
)
def test_action_number_keeps_real_counts(action, expected):
    assert rag.ACTION_NUMBER.sub("", action) == expected


@pytest.mark.parametrize(
    "question", ["그건 왜 그래?", "그럼 대응은?", "더 자세히 설명해줘"]
)
def test_reference_words_skip_llm(monkeypatch, question):
    monkeypatch.setattr(rag, "ask_ollama", pytest.fail)

    assert rag.is_follow_up(question) is True


def test_follow_up_comes_from_llm(monkeypatch):
    monkeypatch.setattr(rag, "ask_ollama", lambda *_, **__: '{"kind": "independent"}')
    assert rag.is_follow_up("DO가 뭐야?") is False

    monkeypatch.setattr(rag, "ask_ollama", lambda *_, **__: '{"kind": "dependent"}')
    assert rag.is_follow_up("왜 줄어들어?") is True


def test_follow_up_failure_keeps_question_independent(monkeypatch):
    def broken(*_, **__):
        raise rag.LlmError("실패")

    monkeypatch.setattr(rag, "ask_ollama", broken)

    assert rag.is_follow_up("왜 줄어들어?") is False


def test_rewrite_uses_llm_question(monkeypatch):
    monkeypatch.setattr(
        rag, "ask_ollama", lambda *_, **__: '{"question": "남조류가 독소를 만들어?"}'
    )

    assert rag.rewrite_question("그게 독소를 만들어?", [("남조류?", "세균")]) == (
        "남조류가 독소를 만들어?"
    )


@pytest.mark.parametrize(
    "output",
    ['{"question": "산소 소비가 늘어 부족해집니다."}', "깨진 출력"],
)
def test_rewrite_keeps_original_when_output_is_not_a_question(monkeypatch, output):
    monkeypatch.setattr(rag, "ask_ollama", lambda *_, **__: output)

    assert rag.rewrite_question("왜 그래?", [("질문", "답")]) == "왜 그래?"


def test_stage_question_lists_every_role(monkeypatch):
    monkeypatch.setattr(rag, "table_chunks", lambda *_: [action_hit()])

    result = rag.action_answer(None, "경계 단계 조치는?")

    assert len(result.actions) == 3
    assert result.actions[2].startswith("취수장·정수장 관리자: 취수구 이동, ")


def test_monitoring_question_uses_monitoring_role(monkeypatch):
    monkeypatch.setattr(rag, "table_chunks", lambda *_: [action_hit()])

    result = rag.action_answer(None, "경계 단계의 모니터링 주기는?")

    assert all("주 2회 시료 채취" in action for action in result.actions)
    assert result.actions[0].startswith("상수원 구간: ")


def test_lookup_finds_action_word(monkeypatch):
    monkeypatch.setattr(rag, "table_chunks", lambda *_: [action_hit()])

    result = rag.lookup_answer(None, "차단막 설치는 누가 언제 하나?")

    assert "수면관리자" in result.answer


def test_lookup_needs_who_or_when():
    assert rag.lookup_answer(None, "차단막 설치 방법은?") is None


def test_site_actions_follow_observed_stage(monkeypatch):
    monkeypatch.setattr(rag, "table_chunks", lambda *_: [action_hit()])
    location = next(iter(LOCATIONS.values()))

    actions, hits = rag.site_actions(None, location, 4.5)

    assert len(actions) == 3
    assert len(hits) == 1


def test_drop_listed_actions():
    text = "전망은 관심입니다.\n다음 조치를 권장합니다:\n1) 시료 채취\n2) 정수 강화"

    assert rag.drop_listed_actions(text).strip() == "전망은 관심입니다."


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("예측 데이터가 제공되지 않아 판단이 어렵습니다.", ""),
        ("분위수 예측 범위가 없습니다.", "분위수 예측 범위가 없습니다."),
    ],
)
def test_drop_false_limits(text, expected):
    assert rag.drop_false_limits(text, 1.0, object()) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("26.42857143", 26.4), ("0.01234", 0.012), ("없음", "없음"), (None, None)],
)
def test_short_number(value, expected):
    assert rag.short_number(value) == expected


def test_no_data_answer_drops_invented_actions(stubbed):
    stubbed["output"] = {
        "answer": "제공된 자료로는 이 내용을 확인할 수 없어 판단이 어렵습니다.",
        "management_actions": ["지어낸 조치"],
        "evidence_ids": [],
        "limitations": "",
    }

    assert run().actions == []


@pytest.mark.parametrize(
    "question",
    [
        "일본의 조류경보 기준은 어떻게 돼?",
        "클로로필-a는 어떤 방법으로 측정해?",
        "측정 방법 알려줘",
    ],
)
def test_out_of_scope_questions_skip_llm(question):
    result = rag.out_of_scope_answer(question)

    assert "답할 수 없습니다" in result.answer
    assert result.evidence == []


def test_in_scope_question_is_not_refused():
    assert rag.out_of_scope_answer("조류경보 해제 기준은?") is None


def test_trend_sentence_only_states_values():
    rows = [
        {"WT_CS": "24.3", "Chla_CS": "34.7", "Prec_CS": "30.1"},
        {"WT_CS": "26.1", "Chla_CS": "30.0", "Prec_CS": "1.0"},
    ]

    text = rag.trend_sentence(rows)

    assert "수온은 24.3에서 26.1°C로 높아졌습니다." in text
    assert "클로로필a는 34.7에서 30.0 mg/m³로 낮아졌습니다." in text
    assert "이번 주 강수량은 1.0 mm입니다." in text


def test_trend_sentence_skips_missing_values():
    rows = [{"WT_CS": "", "Prec_CS": ""}, {"WT_CS": "26.1", "Prec_CS": ""}]

    assert rag.trend_sentence(rows) == ""


def test_site_overview_does_not_call_llm(monkeypatch):
    monkeypatch.setattr(rag, "ask_ollama", pytest.fail)
    monkeypatch.setattr(rag, "table_chunks", lambda *_: [action_hit()])
    monkeypatch.setattr(rag, "classify_route", lambda *_: "site")
    monkeypatch.setattr(rag, "search_evidence", pytest.fail)

    result = rag.build_answer(
        None,
        "칠서 전망과 대응을 알려줘",
        LOCATIONS["CS"],
        date(2023, 6, 12),
        MockPredictor(),
        MockObservations(),
    )

    assert result.observed is not None
    assert rag.OFFICIAL_UNKNOWN in result.answer
