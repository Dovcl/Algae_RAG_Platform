"""사람이 보지 않은 쪽에서 의심스러운 블록을 찾아 적재 제외 목록을 만든다.

고치지 않고 뺀다. 틀린 내용이 근거로 들어가는 것보다 잠시 빠져 있는 편이 안전하다.
블록의 글자가 PDF 글자 층에 없거나, 표 칸의 숫자·기관이 Docling과 다르면 뺀다.
결과는 corpus/auto_review.json이고 convert_upstage.py가 읽는다.

    python3 parsers/auto_review.py DOC01 163 186 260 302
"""

import json
import sys

from compare_parsers import (
    EVAL,
    LAYER_MODES,
    MANIFEST,
    ROOT,
    figure_lines,
    grid,
    loose,
    missing_from_layer,
    page_pieces,
    parsed,
    text_layer,
    tokens,
)

OUT = ROOT / "corpus" / "auto_review.json"
RAW = EVAL / "upstage-enhanced"
SKIP = {"header", "footer", "chart", "caption"}
REVIEWED_RANGES = {"DOC01": [(164, 168), (174, 174), (181, 183)]}


def facts(cell: str) -> set[tuple[str, str]]:
    """칸의 숫자, 기관, 단계, 단위. 글자 층 대조로는 기관이 뒤바뀐 것을 못 잡는다."""
    return {(kind, token) for kind, found in tokens([cell]).items() for token in found}


def number_gaps(table: list[list[str]], other: list[list[str]]) -> list[str]:
    """모양이 같은 표에서 칸의 사실이 다른 곳. 표현 차이는 무시한다."""
    if [len(row) for row in table] != [len(row) for row in other]:
        return []
    return [
        " ".join(left.split())
        for row, other_row in zip(table, other, strict=True)
        for left, right in zip(row, other_row, strict=True)
        if right.strip() and facts(left) != facts(right)
    ]


def extra_orgs(elements: list[dict], layer: str) -> set[str]:
    """글자 층보다 많이 나온 기관. 표 모양이 달라 칸을 비교하지 못해도 잡는다."""
    pieces = [
        piece
        for element in elements
        if element["category"] not in SKIP
        for piece in page_pieces(element["content"]["html"])
    ]
    found = tokens(pieces)["기관"]
    known = tokens([layer])["기관"]
    return {
        org for org, places in found.items() if len(places) > len(known.get(org, []))
    }


def review_page(doc_id: str, page: int, layers: list[str]) -> list[dict]:
    body = json.loads((RAW / f"{doc_id}_p{page:03d}.json").read_text(encoding="utf-8"))
    docling_path = EVAL / "docling" / f"{doc_id}_p{page:03d}.html"
    docling = []
    if docling_path.exists():
        html = docling_path.read_text(encoding="utf-8")
        docling = [grid(table) for table in parsed(html)[0].tables]
    tables = [e for e in body["elements"] if e["category"] == "table"]
    extra = extra_orgs(body["elements"], layers[0])
    found = []
    for element in body["elements"]:
        category, html = element["category"], element["content"]["html"]
        if category in SKIP or (
            category.startswith("heading") and len(loose(html)) <= 80
        ):
            continue
        block_id = f"{doc_id}-p{page:03d}-b{int(element['id']) + 1:02d}"
        pieces = figure_lines(html) if category == "figure" else page_pieces(html)
        reasons = [
            f"글자 층에 없음: {p[:60]}" for p in missing_from_layer(pieces, layers)
        ]
        if category == "table" and len(tables) == len(docling):
            own = [grid(t) for t in parsed(html)[0].tables]
            other = docling[tables.index(element)]
            if own:
                reasons += [f"칸 다름: {c[:60]}" for c in number_gaps(own[0], other)]
        if category == "table":
            orgs = extra & set(tokens(page_pieces(html))["기관"])
            reasons += [f"기관 수가 글자 층보다 많음: {org}" for org in orgs]
        if reasons:
            found.append(
                {
                    "block_id": block_id,
                    "category": category,
                    "reasons": reasons[:3],
                    "count": len(reasons),
                }
            )
    return found


def main() -> None:
    doc_id, *edges = sys.argv[1:]
    bounds = list(zip(map(int, edges[::2]), map(int, edges[1::2]), strict=True))
    doc = next(
        d
        for d in json.loads(MANIFEST.read_text(encoding="utf-8"))["documents"]
        if d["document_id"] == doc_id
    )
    reviewed = REVIEWED_RANGES.get(doc_id, [])
    pages = [
        page
        for first, last in bounds
        for page in range(first, last + 1)
        if not any(low <= page <= high for low, high in reviewed)
    ]
    layers = [text_layer(ROOT / doc["file"], mode) for mode in LAYER_MODES]
    dropped = [
        item
        for page in pages
        for item in review_page(
            doc_id, page, [loose(layer[page - 1]) for layer in layers]
        )
    ]
    report = {"document_id": doc_id, "pages": pages, "dropped": dropped}
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"{doc_id}: {len(pages)}쪽, 제외 {len(dropped)}블록")


if __name__ == "__main__":
    main()
