"""질문에 주어진 수치를 계산하고 검색된 기준표의 조건과 대조한다."""

import re
from decimal import Decimal

from app.rag.retrieve import Hit

NUMBER = r"\d+(?:,\d{3})*(?:\.\d+)?"
VALUE = re.compile(rf"(?<![\w.+-])({NUMBER})(?!\d|[,.]\d)")
BOUND = re.compile(rf"({NUMBER})\s*세포\s*/\s*mL\s*(이상|미만)", re.IGNORECASE)
CRITERIA = re.compile(r"구간: (.+?)구간 \| 단계: (.+?) \| 발령·해제 기준: (.+)")
THRESHOLD = re.compile(r"(?:세포\s*/\s*mL\s*)?(?:미만|이상|이하|초과)")
GENUS = r"마이크로시스티스|아나베나|아파니조메논|오실라토리아"
GENUS_GROUP = re.compile(
    rf"((?:{GENUS})(?:[·ㆍ,와과\s]*(?:{GENUS}))*)"
    rf"(?:가|이|는|은)?\s*(각각\s*)?({NUMBER})"
)


def decimal(text: str) -> Decimal:
    return Decimal(text.replace(",", ""))


def given_values(question: str) -> tuple[list[Decimal], Decimal | None]:
    """명시된 값만 읽는다. 4개 속 합계가 계산되면 합계만 기준에 비교한다."""
    total = Decimal(0)
    counted: list[str] = []
    for match in GENUS_GROUP.finditer(question):
        genera = re.findall(GENUS, match[1])
        # 여러 속에 '각각'이 없으면 합계인지 불명확하므로 계산하지 않는다.
        if len(genera) > 1 and not match[2]:
            continue
        total += decimal(match[3]) * len(genera)
        counted.extend(genera)
    values = []
    mentioned = []
    if len(counted) == 4 and len(set(counted)) == 4:
        return [total], total
    question = re.sub(r"\d{4}-\d{2}-\d{2}", "", question)
    for match in VALUE.finditer(question):
        # 측정 횟수·날짜·단계 번호를 세포수 값으로 해석하지 않는다.
        tail = question[match.end() :].lstrip()
        if re.match(
            r"(?:회|번|년|월|일|단계|%|copies|ng|ug|mg|분|시)", tail, re.IGNORECASE
        ):
            continue
        value = decimal(match[1])
        # "한 번은 1,000 미만이니"의 1,000은 측정값이 아니라 사용자가 말한 기준이다
        target = mentioned if THRESHOLD.match(tail) else values
        if value not in target:
            target.append(value)
    # 기준값만 말한 질문("1,000 미만이면 해제야?")은 그 값을 비교 대상으로 쓴다
    return values or mentioned, None


def criteria_calculations(question: str, hits: list[Hit]) -> str:
    """숫자는 계산·비교만 한다. 발령·전환의 행정적 결론은 생성하지 않는다."""
    values, total = given_values(question)
    lines = (
        []
        if total is None
        else [f"질문에 준 4개 속의 세포수 합계 = {total:,} 세포/mL."]
    )
    for hit in hits:
        for line in hit.text.splitlines():
            row = CRITERIA.fullmatch(line)
            if row is None:
                continue
            bounds = [(decimal(n), op) for n, op in BOUND.findall(row[3])]
            if not bounds:
                continue
            for value in values:
                if all(value >= n if op == "이상" else value < n for n, op in bounds):
                    lines.append(
                        f"{value:,} 세포/mL는 {row[1]} {row[2].replace(' ', '')}의 수치 범위에 해당."
                    )
    if lines:
        lines.append(
            "이는 개별 값의 범위 비교다. 2회 연속 채취는 해당 조건을 각 측정에서 "
            "충족해야 한다는 뜻이며 평균·측정 횟수 합산으로 대체할 수 없다. "
            "질문에 없는 측정값·추세·공식 발령 사실을 추가하지 않는다."
        )
    return "\n".join(dict.fromkeys(lines))
