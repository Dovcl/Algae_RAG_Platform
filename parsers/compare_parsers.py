"""Upstage와 Docling 결과에서 숫자·기관·단계·단위가 다른 곳을 쪽별로 뽑는다.

사람이 검수할 곳을 줄이려고 쓴다. 판정이 아니라 후보 목록이다.
쪽 안의 개수를 비교한다. 같은 쪽에 두 기관이 모두 있어도 한쪽을 잘못 읽으면 개수가 달라진다.
Docling은 목차와 절 번호를 흘릴 때가 많아, Upstage 개수가 PDF 글자 층과 같으면 후보에서 뺀다.
글자 층이 Upstage 글자의 절반도 안 되는 쪽은 이미지 표로 보고 따로 둔다. 자동 대조가 안 된다.
옆 칸 글자가 섞이면 개수가 같을 수 있어 표 모양이 같으면 칸끼리도 비교한다.
Upstage의 문단·칸·그림 글자가 글자 층에 그대로 있는지도 본다. 숫자가 아닌 글자 오인식은 여기서 나온다.
두 파서가 똑같이 틀린 곳, 글자 층 없는 그림과 수식은 잡지 못한다.

    python3 parsers/compare_parsers.py
"""

import html as html_lib
import json
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

from score import Units, grid, norm

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.json"
EVAL = ROOT / "data" / "parser_eval"
REPORT = EVAL / "compare.md"
LENGTH_GAP = 0.2
IMAGE_LAYER = 0.5
MIN_PIECE = 2
INVENTED = "first page of the document"
LAYER_MODES = ("", "-raw", "-layout")
SAME_SIGNS = {
    "∼": "~",
    "〜": "~",
    "˚": "°",
    "′": "'",
    "″": '"',
    "`": "'",
    "℃": "°c",
    "内": "內",
}
CONTEXT = 30

RIVERS = ("한강", "낙동강", "금강", "영산강")
ORGS = [
    "기후에너지환경부장관",
    "기후에너지환경부",
    "환경부장관",
    "환경부",
    "국립환경과학원",
    "한강통합물환경센터",
    "국가녹조대응센터",
    "대구지방환경청",
    "대구환경청",
    "원주지방환경청",
    "전북지방환경청",
    "유역(지방)환경청장",
    "유역(지방)환경청",
    "유역지방환경청장",
    "유역환경청장",
    "유역환경청",
    "지방환경청장",
    "환경청장",
    "시도지사",
    "시장군수구청장",
    "수면관리자",
    "취수장정수장관리자",
    "홍수통제소장",
    "홍수통제소",
    "한국수자원공사사장",
    "한국수자원공사",
    "수자원공사",
    "한국환경공단이사장",
    "한국환경공단",
    "한국농어촌공사",
    "농어촌공사",
    "보건환경연구원장",
    "보건환경연구원",
    "물환경연구소장",
    "물환경연구소",
    "상수도사업본부",
]
ORGS += [
    f"{river}{office}"
    for river in RIVERS
    for office in ("유역환경청장", "유역환경청", "물환경연구소", "물환경센터")
]
STAGES = ["조류대발생", "관심", "경계", "해제"]
UNITS = ["cells/ml", "mg/l", "ug/l", "ng/l"]
KINDS = {"기관": ORGS, "단계": STAGES, "단위": UNITS}
BULLET = re.compile(r"^[○●◦◆■□▪•※‣▶o-]+")
DAYS = re.compile(r"[<〈]\s*(\d+)\s*일\s*[>〉]")
NUMBER = re.compile(r"[<>≤≥]?\s*\d[\d,]*(?:\.\d+)?")


def term_pattern(terms: list[str]) -> re.Pattern:
    # 긴 이름부터 찾아야 환경부장관이 환경부로 두 번 세이지 않는다
    return re.compile("|".join(map(re.escape, sorted(terms, key=len, reverse=True))))


PATTERNS = {kind: term_pattern(terms) for kind, terms in KINDS.items()}


def text_layer(pdf: Path, mode: str = "") -> list[str]:
    """PDF 글자 층을 쪽 목록으로 돌려준다. pdftotext는 쪽 끝마다 \\f를 넣는다."""
    command = ["pdftotext", *([mode] if mode else []), str(pdf), "-"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.split("\f")


def loose(text: str) -> str:
    """같은 뜻의 기호와 수식 표기를 맞춰 글자 자체의 차이만 남긴다."""
    text = text.replace("$\\pm$", "±").replace("$\\mu", "u")
    text = re.sub(r"[$\\{}_^]", "", text)
    text = BULLET.sub("", norm(text).lower())
    for sign, same in SAME_SIGNS.items():
        text = text.replace(sign, same)
    return text


def furniture(html: str) -> list[str]:
    """Upstage가 머리말·꼬리말로 분류한 글자. Docling은 이를 본문에 섞기도 한다."""
    found = re.findall(r"<(header|footer)\b[^>]*>(.*?)</\1>", html, flags=re.DOTALL)
    return [re.sub(r"<[^>]+>", "", text) for _, text in found]


def parsed(html: str) -> tuple[Units, int]:
    """비교할 글자만 파싱하고, 대조에서 뺀 Upstage 그림 수를 돌려준다."""
    html = re.sub(r"<head>.*?</head>", "", html, flags=re.DOTALL)
    html = re.sub(r"<(header|footer)\b.*?</\1>", "", html, flags=re.DOTALL)
    html, figures = re.subn(r"<figure id=.*?</figure>", "", html, flags=re.DOTALL)
    # 표 안의 caption은 칸이 시작될 때 버려지므로 표 밖 문단으로 옮긴다
    captions = re.findall(r"<caption>(.*?)</caption>", html, flags=re.DOTALL)
    html = re.sub(r"<caption>.*?</caption>", "", html, flags=re.DOTALL)
    html += "".join(f"<p>{caption}</p>" for caption in captions)
    parser = Units()
    parser.feed(html)
    parser.flush()
    return parser, figures


def page_units(html: str) -> tuple[list[str], int]:
    """문단과 표의 행을 돌려준다."""
    parser, figures = parsed(html)
    rows = [
        " | ".join(cell[0] for cell in row) for table in parser.tables for row in table
    ]
    return parser.blocks + rows, figures


def page_pieces(html: str) -> list[str]:
    """문단과 칸을 돌려준다. 글자 층이 없는 $$ 수식은 뺀다."""
    parser, _ = parsed(re.sub(r"\$\$.*?\$\$", "", html, flags=re.DOTALL))
    cells = [cell[0] for table in parser.tables for row in table for cell in row]
    return parser.blocks + cells


def figure_lines(html: str) -> list[str]:
    """Upstage 그림의 OCR 글자를 줄 단위로 돌려준다. 이 글자는 청크로 들어간다."""
    found = re.findall(r'figure-ocr-text">(.*?)</p>', html, flags=re.DOTALL)
    return [line for text in found for line in html_lib.unescape(text).split("\n")]


def missing_from_layer(pieces: list[str], layers: list[str]) -> list[str]:
    """글자 층 어느 읽기 순서에도 그대로 없는 조각. 셋 다 봐야 표 읽기 순서 오탐이 준다."""
    return [
        " ".join(piece.split())
        for piece in pieces
        if len(loose(piece)) >= MIN_PIECE
        and not any(loose(piece) in layer for layer in layers)
    ]


def cell_diffs(upstage_html: str, docling_html: str) -> tuple[list[str], int]:
    """모양이 같은 표끼리 칸을 비교한다. 모양이 달라 비교하지 못한 표의 수도 돌려준다."""
    upstage = [grid(table) for table in parsed(upstage_html)[0].tables]
    docling = [grid(table) for table in parsed(docling_html)[0].tables]
    if len(upstage) != len(docling):
        return [], max(len(upstage), len(docling))
    found, skipped = [], 0
    for left, right in zip(upstage, docling, strict=True):
        if [len(row) for row in left] != [len(row) for row in right]:
            skipped += 1
            continue
        for left_row, right_row in zip(left, right, strict=True):
            for number, (first, second) in enumerate(
                zip(left_row, right_row, strict=True)
            ):
                # Docling은 병합된 앞쪽 머리 칸을 비워 둔다
                if number < 2 and not second.strip():
                    continue
                if loose(first) != loose(second):
                    found.append(
                        f"U `{' '.join(first.split())}` / D `{' '.join(second.split())}`"
                    )
    return found, skipped


def snippet(text: str, start: int, end: int) -> str:
    return " ".join(text[max(0, start - CONTEXT) : end + CONTEXT].split())


def tokens(units: list[str]) -> dict[str, dict[str, list[str]]]:
    """종류별로 {토큰: 나온 곳 목록}을 돌려준다."""
    found: dict[str, dict[str, list[str]]] = {kind: {} for kind in ("숫자", *KINDS)}
    for unit in units:
        plain = unicodedata.normalize("NFKC", unit)
        # <13일> 같은 일수 묶음의 꺾쇠를 부등호로 읽지 않게 지운다
        plain = DAYS.sub(r"\1일", plain)
        for match in NUMBER.finditer(plain):
            token = re.sub(r"[\s,]", "", match[0])
            found["숫자"].setdefault(token, []).append(
                snippet(plain, match.start(), match.end())
            )
        squeezed = norm(unit).lower()
        for kind, pattern in PATTERNS.items():
            for match in pattern.finditer(squeezed):
                found[kind].setdefault(match[0], []).append(
                    snippet(squeezed, match.start(), match.end())
                )
    return found


def differences(upstage: dict, docling: dict, checked: dict, layer: dict) -> list[dict]:
    """두 파서의 차이 중 checked(머리말을 포함한 Upstage)가 글자 층과 어긋나는 것만 남긴다."""
    rows = []
    for kind, upstage_tokens in upstage.items():
        left = Counter({key: len(value) for key, value in upstage_tokens.items()})
        right = Counter({key: len(value) for key, value in docling[kind].items()})
        for token, count in (left - right).items():
            rows.append(
                {
                    "kind": kind,
                    "side": "Upstage",
                    "token": token,
                    "count": count,
                    "where": upstage_tokens[token][:2],
                }
            )
        for token, count in (right - left).items():
            rows.append(
                {
                    "kind": kind,
                    "side": "Docling",
                    "token": token,
                    "count": count,
                    "where": docling[kind][token][:2],
                }
            )
    return [
        row
        for row in rows
        if len(checked[row["kind"]].get(row["token"], []))
        != len(layer[row["kind"]].get(row["token"], []))
    ]


def compare_page(doc_id: str, page: int, layer_texts: list[str]) -> dict:
    layer_text = layer_texts[0]
    name = f"{doc_id}_p{page:03d}.html"
    docling_path = EVAL / "docling" / name
    upstage_html = (EVAL / "upstage-enhanced" / name).read_text(encoding="utf-8")
    upstage_units, figures = page_units(upstage_html)
    upstage_chars = sum(len(norm(unit)) for unit in upstage_units)
    result = {
        "doc": doc_id,
        "page": page,
        "figures": figures,
        "diffs": [],
        "image": len(norm(layer_text)) < IMAGE_LAYER * upstage_chars,
        "layer_chars": (upstage_chars, len(norm(layer_text))),
        "cells": [],
        "skipped_tables": 0,
        "invented": INVENTED in upstage_html,
    }
    layers = [loose(text) for text in layer_texts]
    result["missing"] = missing_from_layer(page_pieces(upstage_html), layers)
    lines = figure_lines(upstage_html)
    result["figure_missing"] = missing_from_layer(lines, layers)
    result["figure_lines"] = sum(len(loose(line)) >= MIN_PIECE for line in lines)
    if not docling_path.exists():
        return {**result, "no_docling": True}

    skip = furniture(upstage_html)
    joined = "".join(norm(text) for text in skip)
    docling_html = docling_path.read_text(encoding="utf-8")
    result["cells"], result["skipped_tables"] = cell_diffs(upstage_html, docling_html)
    docling_units, _ = page_units(docling_html)
    docling_units = [unit for unit in docling_units if norm(unit) not in joined]
    result["chars"] = (upstage_chars, sum(len(norm(unit)) for unit in docling_units))
    result["diffs"] = differences(
        tokens(upstage_units),
        tokens(docling_units),
        tokens(upstage_units + skip),
        tokens([layer_text]),
    )
    return result


def length_gap(result: dict) -> bool:
    first, second = result.get("chars", (0, 0))
    return abs(first - second) > LENGTH_GAP * max(first, second, 1)


def needs_review(result: dict) -> bool:
    return bool(
        result["diffs"]
        or result["cells"]
        or result["missing"]
        or result["figure_missing"]
        or result["invented"]
        or result["skipped_tables"]
        or result.get("no_docling")
        or length_gap(result)
    )


def page_lines(result: dict) -> list[str]:
    lines = [f"### {result['doc']} p{result['page']:03d}"]
    if result["invented"]:
        lines.append("- Upstage가 지어낸 문장 있음: 적재 범위에서 뺀다")
    if result["skipped_tables"]:
        lines.append(f"- 표 {result['skipped_tables']}개는 모양이 달라 칸 대조 못 함")
    lines += [f"- 칸 차이: {cell}" for cell in result["cells"]]
    lines += [f"- 글자 층에 없음: `{piece[:120]}`" for piece in result["missing"]]
    if result["figure_missing"]:
        lines.append(
            f"- 그림 글자 {result['figure_lines']}줄 중 {len(result['figure_missing'])}줄이"
            f" 글자 층에 없음: {' / '.join(result['figure_missing'][:6])}"
        )
    if result.get("no_docling"):
        lines.append("- Docling 결과 없음: 원문과 직접 대조")
    if length_gap(result):
        lines.append(f"- 글자 수 차이 큼: Upstage/Docling {result['chars']}")
    for row in sorted(result["diffs"], key=lambda r: (r["kind"], r["side"])):
        where = " / ".join(row["where"])
        lines.append(
            f"- {row['kind']} `{row['token']}` {row['side']}에만 {row['count']}회: {where}"
        )
    return lines + [""]


def write_report(results: list[dict]) -> None:
    lines = [
        "# Upstage·Docling 대조",
        "",
        "후보 목록이다. 정답은 원문 PDF로 확인한다.",
        "",
        "| 문서 | 쪽 | 후보 없음 | 칸 차이 | 글자 층 불일치 | 그림 글자 불일치 | 개수 차이 | 이미지 표 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for doc_id in sorted({r["doc"] for r in results}):
        everything = [r for r in results if r["doc"] == doc_id]
        part = [r for r in everything if not r["image"]]
        counts = [
            sum(bool(r[key]) for r in part)
            for key in ("cells", "missing", "figure_missing", "diffs")
        ]
        lines.append(
            f"| {doc_id} | {len(everything)} "
            f"| {sum(not needs_review(r) for r in part)} "
            f"| {' | '.join(map(str, counts))} | {len(everything) - len(part)} |"
        )
    lines += ["", "## 검수 후보 쪽", ""]
    for result in results:
        if not result["image"] and needs_review(result):
            lines += page_lines(result)
    lines += ["## 이미지 표 쪽: 자동 대조 불가, 원문과 직접 대조", ""]
    for result in results:
        if result["image"]:
            upstage_chars, layer_chars = result["layer_chars"]
            lines.append(
                f"- {result['doc']} p{result['page']:03d}: "
                f"Upstage {upstage_chars}자, 글자 층 {layer_chars}자"
            )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    results = []
    for doc in manifest["documents"]:
        doc_id = doc["document_id"]
        pages = sorted(
            int(path.stem.rsplit("_p", 1)[1])
            for path in (EVAL / "upstage-enhanced").glob(f"{doc_id}_p*.html")
        )
        layers = [text_layer(ROOT / doc["file"], mode) for mode in LAYER_MODES]
        results += [
            compare_page(doc_id, page, [layer[page - 1] for layer in layers])
            for page in pages
        ]
    write_report(results)
    print(REPORT.relative_to(ROOT))
    review = sum(needs_review(r) or r["image"] for r in results)
    print(f"쪽 {len(results)}, 검수 후보 {review}")


if __name__ == "__main__":
    main()
