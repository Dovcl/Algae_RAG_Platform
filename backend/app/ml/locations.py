from dataclasses import dataclass

RIVER_NAMES = {"NAK": "낙동강", "HAN": "한강", "YOUNG": "영산강", "GUM": "금강"}

# 최종 모델이 있는 21개 지점 (DEV_PLAN.md 5장)
_CODES_BY_RIVER = {
    "NAK": [
        "CGB",
        "CHB",
        "CS",
        "DSB",
        "GJGR",
        "GJGRB",
        "GMB",
        "HCB",
        "HP",
        "MGMR",
        "NDB",
        "SJB",
    ],
    "HAN": ["GDDG", "GJG", "JSCG", "MSDG"],
    "YOUNG": ["JSB", "SCB"],
    "GUM": ["BJB", "GJB", "SaJB"],
}


# 조류_모니터링_지점매칭 표. 코드는 CSV의 상·하류 수문 열과 맞춰 확인했다
_NAMES = {
    "CGB": "칠곡보",
    "CHB": "창녕함안보",
    "CS": "칠서",
    "DSB": "달성보",
    "GJGR": "강정고령",
    "GJGRB": "강정고령보",
    "GMB": "구미보",
    "HCB": "합천창녕보",
    "HP": "해평",
    "MGMR": "물금매리",
    "NDB": "낙단보",
    "SJB": "상주보",
    "GDDG": "강동대교",
    "GJG": "광진교",
    "JSCG": "잠실철교",
    "MSDG": "미사대교",
    "JSB": "죽산보",
    "SCB": "승촌보",
    "BJB": "백제보",
    "GJB": "공주보",
    "SaJB": "세종보",
}

# 연차보고서 PDF 174쪽 조류예측 지점 중 친수활동 표시(†)가 없는 곳. 나머지는 구간을 확인하지 못했다
_WATER_SUPPLY = {"CS", "GJG", "GJGR", "HP", "MGMR", "MSDG"}


@dataclass(frozen=True)
class Location:
    code: str
    river: str
    name: str
    zone: str | None


LOCATIONS = {
    code: Location(
        code=code,
        river=river,
        name=_NAMES[code],
        zone="상수원" if code in _WATER_SUPPLY else None,
    )
    for river, codes in _CODES_BY_RIVER.items()
    for code in codes
}
