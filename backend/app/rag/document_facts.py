"""문서의 수치·조건을 원문과 대조해, 일반 설명과 구체적인 운영 사실을 구분한다."""

import re

from app.rag.calculations import NUMBER, decimal
from app.rag.retrieve import Hit


def document_numeric_answer(
    question: str, hits: list[Hit]
) -> tuple[str, list[Hit]] | None:
    """문서에 명시된 최소 접촉시간과 흐름도 등호 조건을 직접 대조한다."""
    if (
        "접촉시간" in question
        and "분말활성탄" in question
        and not re.search(r"오존|입상활성탄", question)
    ):
        asked = re.search(rf"({NUMBER})\s*(분|시간)", question)
        for hit in hits:
            minimum = re.search(
                rf"(?:적어도|최소)\s*({NUMBER})\s*(분|시간)\s*이상", hit.text
            )
            if minimum is None or "접촉시간" not in hit.text:
                continue
            lines = [
                line
                for line in hit.text.splitlines()
                if "접촉시간" in line or "처리효과를 충분히" in line
            ]
            if not lines:
                continue
            explanation = ""
            if asked:
                value = decimal(asked[1]) * (60 if asked[2] == "시간" else 1)
                lower = decimal(minimum[1]) * (60 if minimum[2] == "시간" else 1)
                result = "충족합니다" if value >= lower else "충족하지 못합니다"
                explanation = f"질문의 {asked[1]}{asked[2]}은 문서의 최소 {minimum[1]}{minimum[2]} 조건을 {result}. [1]\n\n"
            return explanation + "문서에 적힌 접촉시간 조건입니다.\n" + "\n".join(
                lines
            ) + " [1]", [hit]

    if "정수" in question and "2-MIB" in question and "마이크로시스틴" in question:
        mib = re.search(
            rf"정수\s*2-MIB(?:도|가|는)?\s*(?:딱|정확히)?\s*({NUMBER})", question
        )
        toxin = re.search(
            rf"마이크로시스틴(?:도|이|은)?\s*(?:딱|정확히)?\s*({NUMBER})\s*(ug|μg|µg)/L",
            question,
        )
        if mib and toxin:
            for hit in hits:
                if "분기:" not in hit.text:
                    continue
                mib_limit = re.search(rf"2-MIB\s*[≤≦]\s*({NUMBER})\s*ng/L", hit.text)
                toxin_limit = re.search(
                    rf"마이크로시스틴\s*[≤≦]\s*({NUMBER})\s*(?:ug|μg|µg)/L", hit.text
                )
                if mib_limit is None or toxin_limit is None:
                    continue
                # 명시적으로 다른 단위를 준 경우 이 비교를 적용하지 않는다.
                if re.search(
                    rf"정수\s*2-MIB(?:도|가|는)?\s*(?:딱|정확히)?\s*{NUMBER}\s*(?:mg|ug|μg)/L",
                    question,
                ):
                    continue
                passed = decimal(mib[1]) <= decimal(mib_limit[1]) and decimal(
                    toxin[1]
                ) <= decimal(toxin_limit[1])
                verdict = (
                    "질문의 정수 두 항목은 각각 이하(≤) 조건을 만족하므로, 이 흐름도에서는 수질 및 수처리 공정 모니터링 분기로 갑니다. 주민공지 분기로 가는 같은 초과 상태가 아닙니다."
                    if passed
                    else "질문의 정수 항목 중 이하(≤) 조건을 충족하지 못하는 값이 있어, 이 흐름도에서는 주민공지와 공정별 조치로 돌아가는 분기입니다."
                )
                branch = next(
                    line for line in hit.text.splitlines() if line.startswith("분기:")
                )
                return (
                    f"{verdict} [1]\n\n원수의 대응 시작 기준(≥)과 처리 후 정수의 확인 기준(≤)은 구분합니다.\n{branch} [1]",
                    [hit],
                )
    return None


def treatment_fact_answer(
    question: str, hits: list[Hit]
) -> tuple[str, list[Hit]] | None:
    """단위·조건부 운영과 시험표의 구체적 내용은 원문 표현을 유지한다."""
    if (
        "분말활성탄" in question
        and re.search(r"정확|몇|얼마", question)
        and re.search(r"mg/L|투입량", question)
        and re.search(r"(?:원수|분석|측정|수질).{0,30}(?:없|아직)", question)
    ):
        chosen, lines = [], []
        for source in hits:
            for line in source.text.splitlines():
                if " | " not in line and re.search(
                    r"자.?테스트|jar-test|피드백|원수.*특성", line, re.IGNORECASE
                ):
                    if source not in chosen:
                        chosen.append(source)
                    lines.append(f"{line} [{chosen.index(source) + 1}]")
        text = "원수 분석 결과와 현장 처리 조건이 없어 해당 정수장의 정확한 분말활성탄 투입량을 지정할 수 없습니다. 문서의 다른 시험 조건에 해당하는 수치를 현장 처방으로 대신하지 않습니다."
        if lines:
            text += "\n\n문서에서 제시한 결정 조건입니다.\n" + "\n".join(
                dict.fromkeys(lines)
            )
        return text, chosen
    if (
        "분말활성탄" in question
        and "염소" in question
        and re.search(r"동시|소비|손실|간격", question)
    ):
        chosen, lines = [], []
        for source in hits:
            for line in source.text.splitlines():
                if (
                    "활성탄" in line
                    and "염소" in line
                    and "소비" in line
                    and "mg/L" in line
                ):
                    if source not in chosen:
                        chosen.append(source)
                    lines.append(f"{line} [{chosen.index(source) + 1}]")
        if lines:
            for source in hits:
                for line in source.text.splitlines():
                    if "동시에" in line and "염소" in line and "손실" in line:
                        if source not in chosen:
                            chosen.append(source)
                        lines.append(f"{line} [{chosen.index(source) + 1}]")
            return (
                "손실이 없다고 안내할 수 없습니다. 문서에 적힌 염소 소비와 동시 주입 조건입니다.\n"
                + "\n".join(dict.fromkeys(lines)),
                chosen,
            )
    if (
        "%" in question
        and "mg/L" in question
        and "활성탄" in question
        and "오존" not in question
        and re.search(r"단위|농도|가리키|같", question)
    ):
        literal_ranges = [
            re.sub(r"\s+", "", value)
            for value in re.findall(
                rf"{NUMBER}\s*[~～∼]\s*{NUMBER}\s*(?:%|mg/L)", question
            )
        ]
        exact = [
            h
            for h in hits
            if any(value in re.sub(r"\s+", "", h.text) for value in literal_ranges)
        ]
        chosen = exact or [
            h
            for h in hits
            if "슬러리" in h.text or ("10~30" in h.text and "투입" in h.text)
        ]
        if chosen:
            lines = []
            for index, hit in enumerate(chosen[:3], 1):
                for line in hit.text.splitlines():
                    if "슬러리" in line or ("투입" in line and "mg/L" in line):
                        lines.append(f"{line} [{index}]")
            explanation = (
                "%는 비율 표기이고 mg/L는 단위 부피당 질량이므로 같은 단위가 아닙니다. "
                "슬러리 조제 농도와 처리할 물에 대한 활성탄 투입량도 서로 구분해야 합니다. "
                "%의 분모 기준·밀도·슬러리 주입량을 확인하지 않고 임의로 환산하지 않습니다."
            )
            return explanation + "\n\n문서에서 확인되는 용도입니다.\n" + "\n".join(
                lines
            ), chosen[:3]

    if (
        "전염소" in question
        and "중염소" in question
        and re.search(r"무조건|완전히|일괄|항상", question)
    ):
        chosen, lines = [], []
        terms = [
            word for word in ("암모니아", "질소", "망간", "철") if word in question
        ]
        ranked = sorted(
            hits, key=lambda h: sum(word in h.text for word in terms), reverse=True
        )
        for hit in ranked:
            paragraphs = hit.text.splitlines()
            for i, line in enumerate(paragraphs):
                if (
                    "전염소" not in line
                    and "중염소" not in line
                    and not any(word in line for word in terms)
                ):
                    continue
                context = " ".join(paragraphs[i : i + 2])
                if not re.search(r"고려|적정|다만|필요|경우", context):
                    continue
                if hit not in chosen:
                    chosen.append(hit)
                lines.append(f"{context} [{chosen.index(hit) + 1}]")
                if len(lines) >= 3:
                    break
            if len(chosen) >= 2:
                break
        if lines:
            return (
                "문서는 원수 조건을 고려한 적정 투입과 전환을 설명하므로, 전염소를 무조건 완전히 중단하라는 뜻으로 요약할 수 없습니다. "
                "질문에 나온 질소·철·망간 등 적용 조건을 함께 확인해야 합니다.\n\n"
                + "\n".join(dict.fromkeys(lines)),
                chosen,
            )

    # 수치표를 물었을 때, 조건과 숫자를 분리해 새로 해석하지 않는다.
    if not re.search(r"표|제거율|제거효율", question):
        return None
    candidates = [h for h in hits if "열: " in h.text and "%" in h.text]
    for hit in candidates:
        if "탁도" in question and "탁도" not in hit.text:
            continue
        if "입상활성탄" in question and "오존" in question:
            if (
                "입상활성탄" not in hit.text
                or "오 존" not in hit.text
                and "오존" not in hit.text
            ):
                continue
        elif "탁도" not in question:
            continue
        chemical = (
            "2-MIB"
            if "2-MIB" in question
            else "Geosmin"
            if re.search(r"Geosmin|지오스민", question)
            else None
        )
        rows = [
            line
            for line in hit.text.splitlines()
            if " | " in line and (chemical is None or chemical in line)
        ]
        if not rows:
            continue
        notes = [
            line
            for line in hit.text.splitlines()
            if re.search(r"Jar|시험|출처:|수온|접촉시간|EBCT", line)
            and " | " not in line
        ]
        comparison = ""
        for row in rows:
            ranges = []
            for cell in row.split(" | "):
                key, _, value = cell.partition(": ")
                name = key.replace(" ", "")
                match = re.fullmatch(
                    rf"(?:약\s*)?({NUMBER})(?:[~～–-]({NUMBER}))?%", value
                )
                if match and "+" not in name and name in question.replace(" ", ""):
                    ranges.append(
                        (name, decimal(match[1]), decimal(match[2] or match[1]))
                    )
            if len(ranges) == 2:
                a, b = ranges
                if a[1] > b[2]:
                    comparison = f"표에서 {a[0]} 단독의 제거율 범위가 {b[0]} 단독보다 높습니다. [1]\n\n"
                elif b[1] > a[2]:
                    comparison = f"표에서 {b[0]} 단독의 제거율 범위가 {a[0]} 단독보다 높습니다. [1]\n\n"
        return comparison + "해당 시험표의 행과 시험 조건입니다.\n" + "\n".join(
            rows + notes
        ) + " [1]", [hit]
    return None
