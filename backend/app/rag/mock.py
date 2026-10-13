"""LLM과 문서 검색 없이 정해진 틀로 답을 만든다.

숫자는 관측·예측 모듈에서 가져오고, 조치와 출처는 예시다.
화면과 응답 형태를 먼저 만들기 위한 것이다.
"""

import re
from dataclasses import dataclass
from datetime import date, timedelta

from app.ml.locations import LOCATIONS, Location
from app.ml.observations import Observations
from app.ml.predictor import Prediction, Predictor, classify_stage, to_cells

_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")

_ACTIONS_BY_STAGE = {
    0: ["정기 모니터링 유지"],
    1: ["주 1회 이상 조류 모니터링", "취수구 주변 육안 점검"],
    2: [
        "주 2회 이상 조류 모니터링",
        "취수구 주변 조류 차단막 점검",
        "정수처리 공정 점검",
    ],
    3: [
        "주 2회 이상 조류 모니터링",
        "취수구 주변 조류 차단막 설치",
        "정수처리 강화 (활성탄 투입 준비)",
        "친수 활동 자제 안내",
    ],
    4: [
        "매일 조류 모니터링",
        "취수구 이동 또는 심층 취수 검토",
        "정수처리 강화 (활성탄 투입)",
        "친수 활동 금지 안내",
    ],
}


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    document: str
    publisher: str
    page: int
    section: str


_EVIDENCE = [
    EvidenceItem(
        id="DOC01-p14",
        document="(예시) 녹조 대응 지침 A",
        publisher="(예시) 발행기관",
        page=14,
        section="단계별 조치",
    ),
    EvidenceItem(
        id="DOC02-p7",
        document="(예시) 취·정수장 운영 지침 B",
        publisher="(예시) 발행기관",
        page=7,
        section="취수구 관리",
    ),
]


@dataclass(frozen=True)
class MockAnswer:
    location: Location
    base_date: date
    observed: float | None  # log10(1 + cells/mL)
    forecast: Prediction
    stage: int
    actions: list[str]
    evidence: list[EvidenceItem]
    limitations: str
    answer: str


def find_location(question: str) -> Location | None:
    """질문에 지점 코드가 들어 있으면 그 지점. 긴 코드부터 본다 (GJGRB가 GJGR보다 먼저)."""
    upper = question.upper()
    for code in sorted(LOCATIONS, key=len, reverse=True):
        if code.upper() in upper:
            return LOCATIONS[code]
    return None


def find_date(question: str) -> date | None:
    match = _DATE_PATTERN.search(question)
    if match is None:
        return None
    try:
        return date.fromisoformat(match.group())
    except ValueError:
        return None


def build_answer(
    question: str,
    location: Location | None,
    base_date: date | None,
    predictor: Predictor,
    observations: Observations,
) -> MockAnswer:
    location = location or find_location(question) or next(iter(LOCATIONS.values()))
    base = base_date or find_date(question)
    if base is None:
        # 자료의 마지막 주를 예측 대상으로 삼고, 그 전 주를 기준일로 한다
        base = observations.weeks(location, None, None, 2)[0]

    observed = observations.cyanocell(location, base)
    forecast = predictor.predict(location, base + timedelta(weeks=1))
    stage = classify_stage(forecast.q50)

    current = (
        "관측값이 없습니다"
        if observed is None
        else f"{to_cells(observed):,.0f} cells/mL로 관측됐습니다"
    )
    answer = (
        f"{location.code} 지점의 유해남조류는 {base.isoformat()}에 {current}. "
        f"다음 주({forecast.target_date.isoformat()})에는 "
        f"{to_cells(forecast.q50):,.0f} cells/mL"
        f"(범위 {to_cells(forecast.q20):,.0f}~{to_cells(forecast.q80):,.0f})로 예측되며, "
        f"{stage}단계에 해당합니다. 이 단계에서는 아래 조치가 권고됩니다 [1][2]."
    )

    return MockAnswer(
        location=location,
        base_date=base,
        observed=observed,
        forecast=forecast,
        stage=stage,
        actions=_ACTIONS_BY_STAGE[stage],
        evidence=_EVIDENCE,
        limitations=(
            "모의 응답입니다. 언어 모델과 문서 검색이 아직 연결되지 않아 "
            "조치와 출처는 예시이고, 질문의 내용은 지점 코드와 날짜만 반영됩니다."
        ),
        answer=answer,
    )
