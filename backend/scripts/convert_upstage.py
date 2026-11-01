"""Upstage 원본 응답을 공통 형식 parsed-v1로 바꾼다.

맥 호스트에서 실행한다. 표준 라이브러리만 쓴다. 표는 병합 셀을 펼쳐 columns, rows로 둔다.
머리말·꼬리말과 모델이 쓴 그림 설명(figure-description)은 넣지 않는다.
보정은 corpus/parsed_corrections.json에 적어 두고 여기서 적용한다. 원본 응답은 고치지 않는다.

    cd backend && source .venv/bin/activate
    python scripts/convert_upstage.py
"""

import hashlib
import html as html_lib
import json
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "corpus" / "manifest.json"
CORRECTIONS = ROOT / "corpus" / "parsed_corrections.json"
RAW = ROOT / "data" / "parser_eval" / "upstage-enhanced"
PARSE_RUN_ID = "upstage-enhanced-v1"
SKIP = {"header", "footer"}
PRIVATE_USE = re.compile(r"[\ue000-\uf8ff]")
TABLE_ID = re.compile(r"(표\s*[\d-]+|그림\s*[\d-]+|<표\s*[\d-]+>|<그림\s*[\d-]+>)")


class TableCells(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int, int]]] = []
        self.text: list[str] = []
        self.span = (1, 1)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "tr":
            self.rows.append([])
        elif tag in {"td", "th"}:
            self.text = []
            self.span = (int(attrs.get("rowspan") or 1), int(attrs.get("colspan") or 1))
        elif tag == "br":
            self.text.append("\n")

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.rows:
            self.rows[-1].append(("".join(self.text).strip(), *self.span))
            self.text = []

    def handle_data(self, data):
        self.text.append(data)


def table_grid(html: str) -> list[list[str]]:
    """병합 셀을 펼친 격자를 돌려준다."""
    parser = TableCells()
    parser.feed(html)
    filled: dict[tuple[int, int], str] = {}
    for r, row in enumerate(parser.rows):
        c = 0
        for text, rowspan, colspan in row:
            while (r, c) in filled:
                c += 1
            for i in range(rowspan):
                for j in range(colspan):
                    filled[(r + i, c + j)] = text
            c += colspan
    width = 1 + max((c for _, c in filled), default=0)
    return [
        [filled.get((r, c), "") for c in range(width)] for r in range(len(parser.rows))
    ]


def plain(html: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", html)
    text = re.sub(r"</(p|li|h\d)>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html_lib.unescape(text)
    text = PRIVATE_USE.sub("", text)  # PDF의 글머리 기호가 깨진 글자로 들어온다
    return text.replace("$\\mu$", "μ").replace("$CO_{2}$", "CO2").strip()


def figure_text(html: str) -> str:
    match = re.search(r'figure-ocr-text">(.*?)</p>', html, re.DOTALL)
    return plain(match.group(1)) if match else ""


def page_blocks(
    doc_id: str, page: int, printed: int, section: str | None
) -> tuple[list[dict], str | None]:
    body = json.loads((RAW / f"{doc_id}_p{page:03d}.json").read_text(encoding="utf-8"))
    blocks: list[dict] = []
    caption = None
    for number, element in enumerate(body["elements"], start=1):
        category = element["category"]
        html = element["content"]["html"]
        if category in SKIP:
            continue
        text = plain(html)
        if category.startswith("heading"):
            section = " ".join(text.split())
            continue
        if category == "caption":
            caption = " ".join(text.split())
            continue
        block = {
            "block_id": f"{doc_id}-p{page:03d}-b{number:02d}",
            "type": "paragraph",
            "pdf_page_start": page,
            "pdf_page_end": page,
            "printed_page_start": printed,
            "printed_page_end": printed,
            "section": section,
        }
        if category == "table":
            grid = table_grid(html)
            if len(grid) > 1:
                block["type"] = "table"
                block["columns"] = grid[0]
                block["rows"] = grid[1:]
            else:  # 왼쪽 칸이 제목이고 오른쪽 칸이 본문인 상자
                block["text"] = "\n".join(plain(cell) for cell in grid[0])
        elif category == "figure":
            block["type"] = "figure"
            block["text"] = figure_text(html)
        elif category == "footnote":
            block["type"] = "footnote"
            block["text"] = text
        else:
            block["text"] = text
        if category in {"table", "figure"} and caption:
            found = TABLE_ID.search(caption)
            block["table_id"] = found.group(1) if found else None
            block["caption"] = caption
            caption = None
        block["review"] = {"status": "raw"}
        blocks.append(block)
    return blocks, section


def replace_in_rows(block: dict, fix: dict) -> bool:
    changed = False
    for row in block["rows"]:
        if fix.get("row_contains", "") not in " ".join(row):
            continue
        for index, cell in enumerate(row):
            if fix["old"] in cell:
                row[index] = cell.replace(fix["old"], fix["new"])
                changed = True
    return changed


def apply_corrections(blocks: list[dict], corrections: list[dict]) -> None:
    by_id = {block["block_id"]: block for block in blocks}
    for fix in corrections:
        block = by_id.get(fix["block_id"])
        if block is None:
            continue
        block.update(fix.get("set", {}))
        if "add_text" in fix:
            block["text"] = f"{block['text']}\n{fix['add_text']}"
        if "fill_empty_first" in fix:
            for row in block["rows"]:
                row[0] = row[0] or fix["fill_empty_first"]
        if "old" in fix and not replace_in_rows(block, fix):
            raise SystemExit(f"보정 대상을 찾지 못했습니다: {fix}")
        block["review"] = {"status": "corrected", "reason": fix["reason"]}


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    review = json.loads(CORRECTIONS.read_text(encoding="utf-8"))
    out_dir = ROOT / "data" / "parsed" / PARSE_RUN_ID / review["status"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for doc in manifest["documents"]:
        blocks: list[dict] = []
        section = None
        for pages in doc["page_ranges"]:
            for page in range(pages["pdf_start"], pages["pdf_end"] + 1):
                printed = page - pages["printed_offset"]
                new, section = page_blocks(doc["document_id"], page, printed, section)
                blocks.extend(new)
        apply_corrections(blocks, review["corrections"])
        for block in blocks:
            if review["status"] == "reviewed" and block["review"]["status"] == "raw":
                block["review"] = {"status": "reviewed"}
        result = {
            "format_version": "parsed-v1",
            "parse_run_id": PARSE_RUN_ID,
            "document_id": doc["document_id"],
            "source_sha256": doc["sha256"],
            "parser": {"name": "upstage-document-parse", "version": "260930"},
            "normalization": [
                "머리말·꼬리말 제거",
                "표 병합 셀 펼침",
                "그림은 OCR 글자만 사용",
            ],
            "page_ranges": doc["page_ranges"],
            "blocks": blocks,
        }
        out = out_dir / f"{doc['document_id']}.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        print(f"{doc['document_id']}: 블록 {len(blocks)}개 → sha256 {digest[:12]}")


if __name__ == "__main__":
    main()
