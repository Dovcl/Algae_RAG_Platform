"""파서가 낸 HTML을 원문 기준(corpus/parser_gold.json)으로 채점한다. 보정 없이 한 번만 본다.

    python parsers/score.py upstage-enhanced docling

표는 병합 셀을 펼쳐 한 행을 한 단위로 본다. 사실의 표현들이 같은 단위에 있어야 맞다.
그래서 숫자만 있고 기관·단계와의 연결이 끊긴 경우는 틀린 것으로 나온다.
scope=manual인 흐름도 분기는 자동 판정하지 않고 사람이 본다.
"""

import argparse
import json
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "corpus" / "parser_gold.json"
OUT = ROOT / "data" / "parser_eval"
BLOCKS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "div", "figcaption", "header", "footer"}


def norm(text: str) -> str:
    # 가운뎃점은 NFKC가 다른 글자로 바꾸기 전에 지운다
    text = re.sub(r"[\s·・･ㆍ․‧∙•]", "", text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[‘’“”]", "'", text)
    text = text.replace("$\\mu$", "u").replace("〉", ">").replace("〈", "<").replace("µ", "u").replace("μ", "u")
    return text


class Units(HTMLParser):
    """블록 글자와, 병합을 펼친 표 격자를 모은다."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self.tables: list[list[list[tuple[str, int, int]]]] = []
        self.text: list[str] = []
        self.rows: list[list[tuple[str, int, int]]] | None = None
        self.row: list[tuple[str, int, int]] = []
        self.span = (1, 1)

    def flush(self) -> None:
        text = "".join(self.text).strip()
        self.text = []
        if text:
            self.blocks.append(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            self.flush()
            self.rows = []
        elif self.rows is not None and tag == "tr":
            self.row = []
        elif self.rows is not None and tag in {"td", "th"}:
            self.text = []
            self.span = (int(attrs.get("rowspan") or 1), int(attrs.get("colspan") or 1))
        elif tag == "img" and attrs.get("alt"):
            self.blocks.append(attrs["alt"])
        elif tag == "br":
            self.text.append(" ")
        elif tag in BLOCKS and self.rows is None:
            self.flush()

    def handle_endtag(self, tag):
        if tag == "table" and self.rows is not None:
            self.tables.append(self.rows)
            self.rows = None
        elif self.rows is not None and tag == "tr":
            self.rows.append(self.row)
        elif self.rows is not None and tag in {"td", "th"}:
            self.row.append(("".join(self.text).strip(), *self.span))
            self.text = []
        elif tag in BLOCKS and self.rows is None:
            self.flush()

    def handle_data(self, data):
        self.text.append(data)


def grid(rows: list[list[tuple[str, int, int]]]) -> list[list[str]]:
    filled: dict[tuple[int, int], str] = {}
    for r, row in enumerate(rows):
        c = 0
        for text, rowspan, colspan in row:
            while (r, c) in filled:
                c += 1
            for i in range(rowspan):
                for j in range(colspan):
                    filled[(r + i, c + j)] = text
            c += colspan
    width = 1 + max((c for _, c in filled), default=0)
    return [[filled.get((r, c), "") for c in range(width)] for r in range(len(rows))]


def units(html: str) -> list[str]:
    parser = Units()
    parser.feed(html)
    parser.flush()
    found = list(parser.blocks)
    for rows in parser.tables:
        table = grid(rows)
        for row in table:
            found.append(" ".join(row))
        # 값의 뜻이 열 머리글에 있는 표는 머리글 + 행 머리 + 칸을 한 단위로 본다
        for row in table[1:]:
            for c, cell in enumerate(row):
                found.append(f"{table[0][c]} {row[0]} {cell}")
    return [norm(unit) for unit in found]


def hit(fact: dict, found: list[str]) -> bool:
    terms = [norm(term) for term in fact["terms"]]
    if fact["scope"] == "unit":
        return any(all(term in unit for term in terms) for unit in found)
    joined = "".join(found)
    return all(term in joined for term in terms)


def bundle_units(parser: str, bundle: dict) -> list[str]:
    found: list[str] = []
    for page in bundle["pdf_pages"]:
        path = OUT / parser / f"{bundle['document_id']}_p{page:03d}.html"
        found += units(path.read_text(encoding="utf-8"))
    return found


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("parsers", nargs="+")
    names = args.parse_args().parsers
    gold = json.loads(GOLD.read_text(encoding="utf-8"))

    for name in names:
        rows = []
        for key, bundle in gold["bundles"].items():
            found = bundle_units(name, bundle)
            facts = [fact for fact in gold["facts"] if fact["bundle"] == key]
            rows += [{**fact, "hit": hit(fact, found)} for fact in facts]
        report = OUT / name / "score.json"
        report.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n")

        auto = [row for row in rows if row["scope"] != "manual"]
        print(f"[{name}] {sum(r['hit'] for r in auto)}/{len(auto)}")
        for key in gold["bundles"]:
            part = [row for row in auto if row["bundle"] == key]
            print(f"  {key}: {sum(r['hit'] for r in part)}/{len(part)}")
        for row in auto:
            if not row["hit"]:
                print(f"  실패 {row['id']} {row['kind']}: {row['fact']}")


if __name__ == "__main__":
    main()
