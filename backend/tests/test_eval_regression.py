import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.eval_regression import (
    audit_input,
    check,
    compare,
    digest,
    load_run,
    number_pattern,
    recheck,
    report,
    run_case,
    validate_cases,
)


def result(**changes):
    return {
        "route": "document",
        "answer": "상수원 구간 기준은 10,000입니다 [1].",
        "actions": [],
        "evidence": [{"document_id": "DOC02", "pdf_page": 34}],
        "limitations": "",
        "error": "",
        "seconds": 20.0,
    } | changes


CASE = {
    "route": "document",
    "numbers": [10000],
    "must_not": ["일종"],
    "expected": [{"document_id": "DOC02", "pdf_page": 34}],
}


def test_passes_when_all_checks_hold():
    assert check(CASE, result()) == []


@pytest.mark.parametrize(
    "text", ["1,000,000", "21,000", "1,000.5", "10001", "0.1000", "-1000", "−1,000"]
)
def test_numeric_boundary_rejects_substring(text):
    import re

    assert not re.search(number_pattern(1000), text)


@pytest.mark.parametrize(
    "text",
    [
        "1,000 세포/mL",
        "1000미만",
        "(1,000)",
        "값은 1,000.",
        "1,000, 다음 값",
        "앞 문장.1000",
    ],
)
def test_numeric_boundary_accepts_exact_value(text):
    import re

    assert re.search(number_pattern(1000), text)


def test_rejects_swapped_zone_values_even_when_both_exist():
    case = {
        "route": "document",
        "scoped": [
            {"label": "상수원", "numbers": [1000], "forbid_numbers": [20000]},
            {"label": "친수활동", "numbers": [20000], "forbid_numbers": [1000]},
        ],
    }
    good = "상수원 구간은 1,000 미만입니다. 친수활동 구간은 20,000 미만입니다."
    assert check(case, result(answer=good)) == []
    bad = (
        good.replace("1,000", "SWAP")
        .replace("20,000", "1,000")
        .replace("SWAP", "20,000")
    )
    failures = check(case, result(answer=bad))
    assert len(failures) == 4
    assert all("구간수치" in failure for failure in failures)


def test_evidence_must_actually_be_cited():
    failed = check(CASE, result(answer="상수원 기준은 10,000입니다."))
    assert any("기대쪽미인용" in f for f in failed)


@pytest.mark.parametrize("citation", ["[0]", "[2]"])
def test_invalid_citations_include_zero(citation):
    failed = check({"route": "document"}, result(answer=f"본문 {citation}"))
    assert any("없는출처번호" in f for f in failed)


def test_checks_actions_and_limitations_too():
    failures = check(
        CASE, result(actions=["조치 [E1]", "**중복**"] * 3, limitations="<br>일종")
    )
    for text in (
        "금지형식:[E",
        "금지형식:**",
        "금지형식:<br",
        "금지어:일종",
        "중복조치",
        "조치과다",
    ):
        assert any(text in f for f in failures)


def test_error_includes_safe_reason():
    assert check(CASE, result(error="형식 오류", error_type="LlmError")) == [
        "오류:LlmError:형식 오류"
    ]


def test_missing_consecutive_condition_fails():
    case = {"route": "document", "required_patterns": [r"2\s*회\s*연속"]}
    assert check(case, result(answer="한 번 넘으면 됩니다."))


def test_concept_cannot_have_evidence_or_actions():
    failures = check({"route": "concept"}, result(route="concept", actions=["조치"]))
    assert "출처불필요경로에출처" in failures
    assert "출처불필요경로에조치" in failures


def test_river_does_not_mix_other_sites():
    from app.ml.locations import LOCATIONS

    text = ", ".join(loc.name for loc in LOCATIONS.values() if loc.river == "GUM")
    case = {"route": "request", "river": "GUM"}
    assert not check(case, result(route="request", answer=text, evidence=[]))
    assert "다른수계지점:칠서" in check(
        case, result(route="request", answer=text + " 칠서", evidence=[])
    )


def test_input_audit_distinguishes_future_observations_from_prediction():
    answer = SimpleNamespace(
        base_date=date(2023, 6, 12),
        prompt=json.dumps({"user": "[관측]\n2023-06-12\n[예측]\n2023-06-19"}),
        forecast=SimpleNamespace(
            input_until=date(2023, 6, 12), target_date=date(2023, 6, 19)
        ),
    )
    assert audit_input(answer, [{"Date": "2023-06-12"}])["failed"] == []
    assert "미래관측혼입" in audit_input(answer, [{"Date": "2023-06-19"}])["failed"][0]
    answer.prompt = json.dumps({"user": "[관측]\n2023-06-19\n[예측]\n2023-06-19"})
    assert any("프롬프트미래관측" in f for f in audit_input(answer, [])["failed"])
    answer.forecast.input_until = date(2023, 6, 19)
    assert any("예측입력미래누출" in f for f in audit_input(answer, [])["failed"])


def test_error_isolated_and_secrets_redacted(monkeypatch):
    from app.rag import answer

    session = Mock()
    calls = 0

    def fail(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise answer.LlmError("Ollama 호출 실패: https://secret:token@example.test")
        raise RuntimeError("postgresql://user:password@host")

    monkeypatch.setattr(answer, "build_answer", fail)
    for error_type in ("LlmError", "RuntimeError"):
        record = run_case(session, {"question": "질문"}, None, None)
        assert record["error_type"] == error_type
        assert "secret" not in json.dumps(record)
        assert "password" not in json.dumps(record)
    assert session.rollback.call_count == 2


def write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    )


def test_checkpoint_survives_interruption_and_truncated_last_line(tmp_path):
    path = tmp_path / "partial.jsonl"
    write_jsonl(
        path,
        [{"type": "run", "total": 2}, {"type": "result", "id": "D01", "failed": []}],
    )
    with path.open("a") as f:
        f.write('{"type": "res')
    run = load_run(str(path))
    assert not run["complete"]
    assert run["passed"] == 1
    assert run["results"][0]["id"] == "D01"


def test_header_only_checkpoint(tmp_path):
    path = tmp_path / "partial.jsonl"
    write_jsonl(path, [{"type": "run", "total": 2}])
    assert load_run(str(path))["results"] == []
    assert not load_run(str(path))["complete"]


def test_corrupt_middle_line_is_not_silently_ignored(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"type":"run"}\n{broken\n{"type":"summary"}\n')
    with pytest.raises(ValueError, match="중간 행"):
        load_run(str(path))


def test_comparison_detects_regression_improvement_changes_and_removed(
    tmp_path, capsys
):
    meta = {
        "llm_model": "m",
        "rag_collection": "c",
        "prompt_sha256": "a",
        "passed": 2,
        "total": 3,
    }
    base = {"question": "질문", "answer": "답", "seconds": 1.0, "actions": []}
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    old_items = [
        base | {"id": "D01", "failed": []},
        base | {"id": "D02", "failed": ["오류"]},
        base | {"id": "D03", "failed": []},
    ]
    new_items = [
        base | {"id": "D01", "failed": ["수치누락"], "actions": ["바뀐 조치"]},
        base | {"id": "D02", "failed": []},
    ]
    old.write_text(json.dumps(meta | {"results": old_items}))
    new.write_text(
        json.dumps(
            meta | {"passed": 1, "total": 2, "prompt_sha256": "b", "results": new_items}
        )
    )
    compare(str(old), str(new))
    output = capsys.readouterr().out
    for text in (
        "통과 2/3 -> 1/2",
        "prompt_sha256: a -> b",
        "나빠짐 D01",
        "좋아짐 D02",
        "삭제/미실행 D03",
        "답변이 바뀐 질문 1개",
    ):
        assert text in output


def test_changed_case_is_excluded_from_comparison(tmp_path, capsys):
    meta = {"passed": 1, "total": 1}
    item = {
        "id": "A",
        "question": "Q",
        "failed": [],
        "answer": "A",
        "seconds": 0,
        "case_sha256": "old",
    }
    old, new = tmp_path / "a.json", tmp_path / "b.json"
    old.write_text(json.dumps(meta | {"results": [item]}))
    new.write_text(
        json.dumps(
            meta | {"results": [item | {"failed": ["오류"], "case_sha256": "new"}]}
        )
    )
    compare(str(old), str(new))
    output = capsys.readouterr().out
    assert "기준 변경 A" in output
    assert "나빠짐" not in output


def test_report_contains_answer_and_failure(tmp_path, capsys):
    path = tmp_path / "r.json"
    path.write_text(
        json.dumps(
            {
                "passed": 0,
                "total": 1,
                "results": [
                    result() | {"id": "D01", "question": "Q", "failed": ["이유"]}
                ],
            }
        )
    )
    report(str(path))
    text = capsys.readouterr().out
    assert "실패 원인: 이유" in text
    assert "10,000" in text
    assert "논문 성능 수치가 아닙니다" in text


def test_fixed_suite_is_valid_and_keeps_all_search_development_questions():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    cases = json.loads((root / "corpus/regression_cases.json").read_text())
    validate_cases(cases)
    dev = json.loads((root / "corpus/dev_questions.json").read_text())
    assert {q["question"] for q in dev} <= {q["question"] for q in cases}
    assert len(cases) == 49
    assert digest(cases) == digest(json.loads(json.dumps(cases)))


def test_duplicate_ids_rejected():
    c = {"id": "A", "question": "Q", "route": "concept"}
    with pytest.raises(ValueError, match="중복"):
        validate_cases([c, c])


def test_recheck_preserves_generation_and_original_results(tmp_path, capsys):
    cases = [{"id": "A", "question": "Q", "route": "document", "numbers": [1000]}]
    cases_path, source = tmp_path / "cases.json", tmp_path / "run.json"
    cases_path.write_text(json.dumps(cases))
    item = result(answer="값은 1,000.") | {
        "id": "A",
        "question": "Q",
        "case": cases[0],
        "failed": ["수치누락"],
    }
    data = {
        "total": 1,
        "passed": 0,
        "suite_sha256": "suite-v1",
        "evaluator_sha256": "checker-v1",
        "results": [item],
    }
    source.write_text(json.dumps(data))
    recheck(str(cases_path), str(source))
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0]["generation_evaluator_sha256"] == "checker-v1"
    assert records[1]["previous_failed"] == ["수치누락"]
    assert records[1]["failed"] == []
    assert records[-1]["passed"] == 1
    assert json.loads(source.read_text()) == data


def test_recheck_refuses_different_generation_input(tmp_path):
    case = {"id": "A", "question": "Q", "route": "document"}
    cases_path, source = tmp_path / "cases.json", tmp_path / "run.json"
    cases_path.write_text(json.dumps([case | {"question": "다른 질문"}]))
    source.write_text(
        json.dumps(
            {
                "total": 1,
                "passed": 1,
                "results": [
                    result() | {"id": "A", "question": "Q", "case": case, "failed": []}
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="생성 입력이 달라"):
        recheck(str(cases_path), str(source))
