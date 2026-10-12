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


@dataclass(frozen=True)
class Location:
    code: str
    river: str


LOCATIONS = {
    code: Location(code=code, river=river)
    for river, codes in _CODES_BY_RIVER.items()
    for code in codes
}
