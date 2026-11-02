"""개발용 답변 회귀 검사. 사용법: corpus/REGRESSION_EVAL.md.

실행은 컨테이너에서 순차적으로 하고 JSONL을 stdout, 진행을 stderr로 낸다.
비교·보고서는 앱이나 DB 없이 호스트의 Python 표준 라이브러리로 실행한다.
"""

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

MAX_SECONDS = 120
MAX_ACTIONS = 6  # 조치 표의 기관 수
FORBIDDEN = ("**", "<br", "[E")


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def number_pattern(value: int) -> str:
    # 쉼표 없는 표기도 허용하지만 1,000이 21,000이나 1,000,000에 걸리면 안 된다.
    forms = {str(value), f"{value:,}"}
    return (
        r"(?<![-−\d])(?<!\d[,.])(?:"
        + "|".join(map(re.escape, sorted(forms)))
        + r")(?!\d|[,.]\d)"
    )


def scoped_text(text: str, label: str) -> str:
    """다음 구간명까지의 내용만 사용한다. 서로 바꿔 쓴 값이 전역 검사로 통과하지 않게 한다."""
    labels = list(re.finditer(r"상수원|친수활동", text))
    return "\n".join(
        text[m.end() : labels[i + 1].start() if i + 1 < len(labels) else len(text)]
        for i, m in enumerate(labels)
        if m.group() == label
    )


def check(case: dict, result: dict) -> list[str]:
    """자동 검사는 내용 전체의 정답 판정이 아니다. 실패에는 근거를 붙인다."""
    if result["error"]:
        return [f"오류:{result.get('error_type', 'Error')}:{result['error']}"]
    failed = []
    answer = result["answer"]
    visible = "\n".join([answer, *result["actions"], result["limitations"]])
    if result["route"] != case["route"]:
        failed.append(f"경로:기대={case['route']},실제={result['route']}")
    if result["seconds"] > case.get("max_seconds", MAX_SECONDS):
        failed.append(f"시간초과:{result['seconds']}초")
    failed += [f"금지형식:{s}" for s in FORBIDDEN if s.lower() in visible.lower()]
    failed += [f"누락:{s}" for s in case.get("must", []) if s not in answer]
    failed += [f"금지어:{s}" for s in case.get("must_not", []) if s in visible]
    for text in case.get("limitations_must", []):
        if text not in result["limitations"]:
            failed.append(f"한계안내누락:{text}")
    if "river" in case:
        from app.ml.locations import LOCATIONS

        for loc in LOCATIONS.values():
            # 강정고령/강정고령보처럼 겹치는 이름은 같은 수계이므로 문제없다.
            if loc.river == case["river"] and loc.name not in answer:
                failed.append(f"수계지점누락:{loc.name}")
            if loc.river != case["river"] and loc.name in answer:
                failed.append(f"다른수계지점:{loc.name}")
    for n in case.get("numbers", []):
        if not re.search(number_pattern(n), answer):
            failed.append(f"수치누락:{n:,}")
    for rule in case.get("scoped", []):
        text = scoped_text(answer, rule["label"])
        for n in rule.get("numbers", []):
            if not re.search(number_pattern(n), text):
                failed.append(f"구간수치:{rule['label']}:{n:,} 누락/오배치")
        for n in rule.get("forbid_numbers", []):
            if re.search(number_pattern(n), text):
                failed.append(f"구간수치:{rule['label']}:{n:,} 잘못 포함")
        for pattern in rule.get("patterns", []):
            if not re.search(pattern, text):
                failed.append(f"구간조건:{rule['label']}:{pattern}")
    for pattern in case.get("required_patterns", []):
        if not re.search(pattern, answer, re.IGNORECASE):
            failed.append(f"필수표현:{pattern}")
    for pattern in case.get("forbidden_patterns", []):
        if re.search(pattern, visible, re.IGNORECASE):
            failed.append(f"금지표현:{pattern}")
    if len(result["actions"]) > MAX_ACTIONS:
        failed.append(f"조치과다:{len(result['actions'])}개")
    if len(result["actions"]) != len(set(result["actions"])):
        failed.append("중복조치")
    if re.search(r"(?:^|\s)(?:[6-9]|\d{2,})[.)]\s", visible):
        failed.append("반복번호목록")
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", visible)}
    invalid = sorted(n for n in cited if not 1 <= n <= len(result["evidence"]))
    if invalid:
        failed.append(f"없는출처번호:{invalid}")
    # 기대 쪽이 evidence에 있어도 본문/조치에서 그 번호를 안 썼다면 인용이 아니다.
    expected = {(e["document_id"], e["pdf_page"]) for e in case.get("expected", [])}
    used = {
        (e["document_id"], e["pdf_page"])
        for n, e in enumerate(result["evidence"], 1)
        if n in cited
    }
    if expected and not expected & used:
        failed.append(f"기대쪽미인용:기대={sorted(expected)},인용={sorted(used)}")
    if case["route"] in ("request", "concept"):
        if result["evidence"] or cited:
            failed.append("출처불필요경로에출처")
        if result["actions"]:
            failed.append("출처불필요경로에조치")
    if result["evidence"] and re.search(r"문서\s*근거가\s*아닙니다", visible):
        failed.append("근거부인")
    for key, value in case.get("status", {}).items():
        actual = result.get("status", {}).get(key)
        if actual != value:
            failed.append(f"상태:{key}:기대={value},실제={actual}")
    audit = result.get("audit", {})
    failed += audit.get("failed", [])
    return failed


class AuditedObservations:
    """제품 코드 변경 없이 build_answer가 실제 요청한 최근 관측 행을 기록한다."""

    def __init__(self, source):
        self.source = source
        self.rows = []

    def weeks(self, *args):
        return self.source.weeks(*args)

    def cyanocell(self, *args):
        return self.source.cyanocell(*args)

    def recent(self, *args):
        rows = self.source.recent(*args)
        self.rows.extend(rows)
        return rows


def audit_input(answer, rows: list[dict]) -> dict:
    failed = []
    base = answer.base_date
    dates = [row.get("Date", "") for row in rows]
    if base is not None:
        for day in dates:
            if not day or date.fromisoformat(day) > base:
                failed.append(f"미래관측혼입:{day},기준={base}")
        # 실제 프롬프트 관측 구획도 검사. 예측 대상일과 문서 발행일은 누출로 보지 않는다.
        if answer.prompt:
            user = json.loads(answer.prompt)["user"]
            observed = user.split("[관측]\n", 1)[-1].split("[예측]\n", 1)[0]
            for day in re.findall(r"\d{4}-\d{2}-\d{2}", observed):
                if date.fromisoformat(day) > base:
                    failed.append(f"프롬프트미래관측:{day},기준={base}")
        forecast = answer.forecast
        if forecast is not None:
            if forecast.input_until > base:
                failed.append(f"예측입력미래누출:{forecast.input_until}")
            if forecast.target_date != base + timedelta(weeks=1):
                failed.append(f"예측대상일오류:{forecast.target_date}")
    return {"observation_dates": dates, "failed": sorted(set(failed))}


def safe_error(error: Exception) -> str:
    # HTTP/SQL 예외 원문에는 URL·접속 정보가 있을 수 있어 저장하지 않는다.
    from app.rag.answer import LlmError

    if isinstance(error, LlmError):
        message = str(error)
        if message.startswith("LLM 응답 형식 오류"):
            return "LLM 응답 형식 오류(JSON/필수 필드 확인 필요)"
        if message.startswith("LLM 답변이 비정상적으로 짧습니다"):
            return "LLM 답변이 비정상적으로 짧습니다"
        return "Ollama 호출 실패(연결/시간 제한/HTTP 상태 확인 필요)"
    return "문항 실행 실패(예외 종류 확인 필요; 접속 정보 보호를 위해 원문 생략)"


def run_case(session, case: dict, predictor, observations) -> dict:
    from app.ml.locations import LOCATIONS
    from app.ml.predictor import to_cells
    from app.rag.answer import CONCEPT_SYSTEM, build_answer

    start = time.monotonic()
    result = {
        "route": "",
        "answer": "",
        "actions": [],
        "evidence": [],
        "limitations": "",
        "error": "",
        "prompt": "",
        "raw_output": "",
        "status": {},
        "audit": {},
    }
    try:
        location = LOCATIONS[case["location"]] if "location" in case else None
        base = date.fromisoformat(case["base_date"]) if "base_date" in case else None
        observed = AuditedObservations(observations)
        answer = build_answer(
            session, case["question"], location, base, predictor, observed
        )
        route = "document"
        if not answer.raw_output and not answer.evidence:
            route = "request"
        elif json.loads(answer.prompt).get("system") == CONCEPT_SYSTEM:
            route = "concept"
        forecast = answer.forecast
        result.update(
            route=route,
            answer=answer.answer,
            actions=answer.actions,
            evidence=[asdict(hit) for hit in answer.evidence],
            limitations=answer.limitations,
            prompt=answer.prompt,
            raw_output=answer.raw_output,
            status={
                "location": answer.location.code if answer.location else None,
                "base_date": str(answer.base_date) if answer.base_date else None,
                "observed_cells": round(to_cells(answer.observed))
                if answer.observed is not None
                else None,
                "forecast_cells": round(to_cells(forecast.q50)) if forecast else None,
                "target_date": str(forecast.target_date) if forecast else None,
                "input_until": str(forecast.input_until) if forecast else None,
                "q20": forecast.q20 if forecast else None,
                "q80": forecast.q80 if forecast else None,
                "stage": answer.stage,
            },
            audit=audit_input(answer, observed.rows),
        )
    except Exception as error:  # noqa: BLE001 - 실패를 기록하고 다른 문항은 계속 평가
        # LlmError뿐 아니라 검색/DB 오류도 한 문항의 실패로 기록하고 다음 문항을 실행한다.
        session.rollback()
        result.update(error=safe_error(error), error_type=type(error).__name__)
    result["seconds"] = round(time.monotonic() - start, 2)
    return result


def validate_cases(cases: list[dict]) -> None:
    if not isinstance(cases, list) or not cases:
        raise ValueError("질문 파일은 비어 있지 않은 배열이어야 합니다")
    ids = []
    for case in cases:
        for key in ("id", "question", "route"):
            if not isinstance(case.get(key), str) or not case[key]:
                raise ValueError(f"질문 필드 누락: {key}")
        if case["route"] not in ("request", "concept", "document"):
            raise ValueError(f"알 수 없는 경로: {case['id']}")
        ids.append(case["id"])
        for pattern in case.get("required_patterns", []) + case.get(
            "forbidden_patterns", []
        ):
            re.compile(pattern)
        for rule in case.get("scoped", []):
            for pattern in rule.get("patterns", []):
                re.compile(pattern)
    if len(ids) != len(set(ids)):
        raise ValueError("중복 질문 ID")


def emit(record: dict) -> None:
    print(json.dumps(record, ensure_ascii=False), flush=True)


def metadata(cases: list[dict]) -> dict:
    import httpx2
    from sqlalchemy import select

    from app.core.config import settings
    from app.db.models import Chunk, ChunkEmbedding, Collection
    from app.db.session import SessionLocal
    from app.rag.answer import CONCEPT_SYSTEM, SYSTEM

    # 명시적인 허용 목록. settings 전체나 .env를 출력하지 않는다.
    keys = (
        "llm_provider",
        "llm_model",
        "embed_model",
        "predict_backend",
        "rag_collection",
        "rag_top_k",
        "llm_num_ctx",
        "llm_num_predict",
        "llm_timeout",
    )
    config = {key: getattr(settings, key) for key in keys}
    with SessionLocal() as session:
        collection = session.get(Collection, settings.rag_collection)
        if collection is None or collection.status != "ready":
            raise ValueError("평가할 ready 컬렉션이 없습니다")
        chunks = session.execute(
            select(Chunk.chunk_id, Chunk.text_sha256)
            .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.chunk_id)
            .where(ChunkEmbedding.collection_id == collection.collection_id)
            .order_by(Chunk.chunk_id)
        ).all()
        config["corpus_sha256"] = digest([list(row) for row in chunks])
        config["embed_model_digest"] = collection.embed_model_digest
    response = httpx2.get(f"{settings.ollama_base_url}/api/tags", timeout=10)
    response.raise_for_status()
    models = response.json()["models"]
    config["llm_model_digest"] = next(
        (m["digest"] for m in models if m["name"] == settings.llm_model), None
    )
    sources = ["app/rag/answer.py", "app/rag/retrieve.py", "app/ml/lab_data.py"]
    root = Path(__file__).resolve().parents[1]
    return config | {
        "schema_version": 2,
        "type": "run",
        "started_at": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "prompt_sha256": digest([SYSTEM, CONCEPT_SYSTEM]),
        "suite_sha256": digest(cases),
        "total": len(cases),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources
        },
    }


def run() -> None:
    from app.core.config import settings
    from app.db.session import SessionLocal
    from app.ml.observations import get_observations
    from app.ml.predictor import get_predictor

    cases = json.load(sys.stdin)
    validate_cases(cases)
    if settings.llm_provider != "ollama" or settings.predict_backend != "precomputed":
        raise ValueError(
            "실제 기준 평가는 ollama + precomputed 설정에서 실행해야 합니다"
        )
    emit(metadata(cases))
    predictor, observations = get_predictor(), get_observations()
    passed = 0
    for number, case in enumerate(cases, 1):
        with SessionLocal() as session:
            result = run_case(session, case, predictor, observations)
        result["failed"] = check(case, result)
        passed += not result["failed"]
        emit(
            {
                "type": "result",
                "id": case["id"],
                "question": case["question"],
                "case_sha256": digest(case),
                "case": case,
            }
            | result
        )
        mark = "통과" if not result["failed"] else f"실패 {result['failed']}"
        print(
            f"{number}/{len(cases)} {case['id']} {result['seconds']}초 {mark}",
            file=sys.stderr,
            flush=True,
        )
    emit(
        {
            "type": "summary",
            "passed": passed,
            "total": len(cases),
            "finished_at": datetime.now(UTC).isoformat(),
        }
    )


def load_run(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        lines = text.splitlines()
        records = []
        truncated = False
        for i, line in enumerate(lines):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if i != len(lines) - 1:
                    raise ValueError("JSONL의 중간 행이 손상되었습니다") from None
                truncated = True
        if not records or records[0].get("type") != "run":
            raise ValueError("실행 메타데이터가 없습니다")
        data = records[0] | {
            "results": [r for r in records if r.get("type") == "result"]
        }
        data["complete"] = not truncated and records[-1].get("type") == "summary"
        data["passed"] = sum(not r["failed"] for r in data["results"])
    else:
        # 한 행만 저장되고 중단된 JSONL도 읽는다.
        if data.get("type") == "run":
            data = data | {"results": [], "passed": 0, "complete": False}
        else:
            data["complete"] = True
    ids = [r["id"] for r in data["results"]]
    if len(ids) != len(set(ids)):
        raise ValueError("중복 결과 ID")
    return data


def compare(old_path: str, new_path: str) -> None:
    old, new = load_run(old_path), load_run(new_path)
    print(f"통과 {old['passed']}/{old['total']} -> {new['passed']}/{new['total']}")
    if not old["complete"] or not new["complete"]:
        print("주의: 중단된 실행이 포함되어 있습니다. 완료된 공통 문항만 비교합니다.")
    for key in (
        "llm_model",
        "rag_collection",
        "prompt_sha256",
        "suite_sha256",
        "evaluator_sha256",
        "corpus_sha256",
        "llm_model_digest",
        "host",
    ):
        if old.get(key) != new.get(key):
            print(f"{key}: {old.get(key)} -> {new.get(key)}")
    before = {r["id"]: r for r in old["results"]}
    after = {r["id"]: r for r in new["results"]}
    changed = 0
    for item in new["results"]:
        previous = before.get(item["id"])
        if previous is None:
            print(f"새 항목 {item['id']}: {item['failed'] or '통과'}")
            continue
        if previous.get("case_sha256") != item.get("case_sha256"):
            print(f"기준 변경 {item['id']}: 회귀 판정 제외")
            continue
        changed += any(
            previous.get(k) != item.get(k)
            for k in ("answer", "actions", "evidence", "limitations")
        )
        if item["failed"] and not previous["failed"]:
            print(f"나빠짐 {item['id']}: {item['failed']} | {item['question']}")
        elif previous["failed"] and not item["failed"]:
            print(f"좋아짐 {item['id']}: {previous['failed']} | {item['question']}")
        elif previous["failed"] != item["failed"]:
            print(
                f"실패 내용 변경 {item['id']}: {previous['failed']} -> {item['failed']}"
            )
    for key in sorted(before.keys() - after.keys()):
        print(f"삭제/미실행 {key}")
    print(f"답변이 바뀐 질문 {changed}개")
    print(f"전체 생성 시간 {sum(r['seconds'] for r in new['results']):.0f}초")


def recheck(cases_path: str, result_path: str) -> None:
    """응답은 다시 생성하지 않고 새 검사 기준으로 재판정한다. 원본은 별도 보존한다."""
    # 호스트 실행에서도 앱의 순수 지점 상수를 읽을 수 있도록 한다(DB/설정 로드 없음).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    validate_cases(cases)
    original = load_run(result_path)
    by_id = {case["id"]: case for case in cases}
    metadata = {
        k: v for k, v in original.items() if k not in ("results", "passed", "complete")
    }
    metadata.update(
        type="run",
        total=len(cases),
        suite_sha256=digest(cases),
        generation_suite_sha256=original.get("suite_sha256"),
        generation_evaluator_sha256=original.get("evaluator_sha256"),
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rechecked_from=Path(result_path).name,
        rechecked_at=datetime.now(UTC).isoformat(),
    )
    emit(metadata)
    passed = completed = 0
    for item in original["results"]:
        case = by_id.get(item["id"])
        if case is None:
            continue
        # 다른 질문문·지점·날짜에 과거 응답을 붙여 재판정하면 안 된다.
        previous = item.get("case", {})
        for key in ("question", "location", "base_date"):
            if case.get(key) != previous.get(key, item.get(key)):
                raise ValueError(
                    f"생성 입력이 달라 재판정할 수 없습니다: {item['id']}:{key}"
                )
        updated = item | {
            "case": case,
            "case_sha256": digest(case),
            "previous_failed": item["failed"],
        }
        updated["failed"] = check(case, updated)
        passed += not updated["failed"]
        completed += 1
        emit(updated)
    if original["complete"] and completed == len(cases):
        emit(
            {
                "type": "summary",
                "passed": passed,
                "total": completed,
                "finished_at": datetime.now(UTC).isoformat(),
            }
        )


def report(path: str) -> None:
    data = load_run(path)
    print("# 챗봇 회귀 평가 결과\n")
    print("이 결과는 개발용 자동 검사이며 논문 성능 수치가 아닙니다.\n")
    print(f"- 원본: `{Path(path).name}`")
    print(f"- 실행 완료: {data['complete']}")
    print(
        f"- 자동 검사 통과: {data['passed']}/{len(data['results'])} (예정 {data['total']}문항)"
    )
    print(f"- 순차 실행 누적 시간: {sum(r['seconds'] for r in data['results']):.1f}초")
    print("\n## 재현 설정\n")
    for key, value in data.items():
        if key not in ("results", "type", "passed", "total"):
            print(f"- {key}: `{value}`")
    print("\n## 질문별 답변과 판정\n")
    for item in data["results"]:
        print(f"### {item['id']} — {item['question']}\n")
        print(f"- 판정: {'실패' if item['failed'] else '자동 검사 통과'}")
        print(f"- 경로: {item['route']} / {item['seconds']}초")
        for failure in item["failed"]:
            print(f"- 실패 원인: {failure}")
        print("\n답변:\n\n" + (item["answer"] or "(생성 실패)"))
        if item["actions"]:
            print("\n조치:\n")
            for action in item["actions"]:
                print(f"- {action}")
        print("\n한계: " + (item["limitations"] or "없음"))
        if item["evidence"]:
            print("\n출처:\n")
            for n, hit in enumerate(item["evidence"], 1):
                print(
                    f"- [{n}] {hit['document_id']} PDF {hit['pdf_page']}쪽 / {hit.get('chunk_id', '')}"
                )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"))
    group.add_argument("--report", metavar="RESULT")
    group.add_argument("--recheck", nargs=2, metavar=("CASES", "RESULT"))
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
    elif args.report:
        report(args.report)
    elif args.recheck:
        recheck(*args.recheck)
    else:
        run()


if __name__ == "__main__":
    main()
