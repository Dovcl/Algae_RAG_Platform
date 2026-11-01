"""검수를 마친 parsed-v1을 검색용 청크로 자른다. 맥 호스트에서 실행한다.

기준은 RAG.md 6-4절이다. 표는 구간별, 단계별로 나누고 제목과 열 이름을 반복한다.
결과는 data/chunks/에 저장하고, 적재는 컨테이너의 scripts/load_chunks.py가 한다.

    cd backend && source .venv/bin/activate
    python scripts/build_chunks.py
"""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "corpus" / "manifest.json"
PARSE_RUN_ID = "upstage-enhanced-v5"
CHUNKING_VERSION = "chunk-v4"
MAX_CHARS = 900
MAX_TABLE_CHARS = 1500
MIN_FIGURE_CHARS = 40
PARTS = ("상수원", "친수활동")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def part_of(text: str) -> str | None:
    found = [part for part in PARTS if part in text]
    return found[0] if len(found) == 1 else None


def header(doc: dict, block: dict) -> str:
    lines = [f"문서: {doc['title']}"]
    if block.get("section"):
        lines.append(f"절: {block['section']}")
    if block.get("caption"):
        lines.append(f"표제: {block['caption']}")
    return "\n".join(lines)


def chunk(doc: dict, blocks: list[dict], body: str, conditions: dict) -> dict:
    first, last = blocks[0], blocks[-1]
    return {
        "document_id": doc["document_id"],
        "text": f"{header(doc, first)}\n{body}",
        "pdf_page_start": first["pdf_page_start"],
        "pdf_page_end": last["pdf_page_end"],
        "printed_page_start": first["printed_page_start"],
        "printed_page_end": last["printed_page_end"],
        "section": first.get("section"),
        "table_id": first.get("table_id"),
        "conditions": conditions,
        "block_ids": [block["block_id"] for block in blocks],
    }


def row_line(columns: list[str], row: list[str]) -> str:
    cells = [
        f"{' '.join(name.split())}: {' '.join(cell.split())}"
        for name, cell in zip(columns, row, strict=False)
    ]
    return " | ".join(cells)


def split_rows(columns: list[str], rows: list[list[str]]) -> list[list[list[str]]]:
    """행을 자르지 않고 MAX_TABLE_CHARS 안에서 묶는다. 넘으면 임베딩이 거부한다."""
    parts: list[list[list[str]]] = []
    current: list[list[str]] = []
    size = 0
    for row in rows:
        length = len(row_line(columns, row))
        if current and size + length > MAX_TABLE_CHARS:
            parts.append(current)
            current, size = [], 0
        current.append(row)
        size += length
    return [*parts, current]


def table_chunks(doc: dict, block: dict, notes: list[str]) -> list[dict]:
    """첫 열 값(구간이나 단계)이 같은 행끼리 묶는다. 묶음마다 비고를 반복한다."""
    groups: dict[str, list[list[str]]] = {}
    for row in block["rows"]:
        groups.setdefault(row[0], []).append(row)

    # 구간이 두 칸에 걸친 표 6은 열 이름이 겹쳐 있어 뒤 열이 단계다
    columns = block["columns"]
    if len(set(columns[:2])) == 1:
        columns = ["구간", "단계", *columns[2:]]

    # 묶음이 너무 잘게 나뉘면(행마다 첫 칸이 다른 표) 표 하나로 둔다
    if len(groups) == len(block["rows"]) and len(block["rows"]) > 1:
        groups = {"": block["rows"]}

    chunks = []
    for label, rows in groups.items():
        part = part_of(f"{label} {block.get('caption') or ''}")
        conditions = {"구간": part} if part else {}
        if label:
            conditions["구분"] = " ".join(label.split())
        for piece in split_rows(columns, rows):
            lines = [f"열: {' / '.join(' '.join(c.split()) for c in columns)}"]
            lines += [row_line(columns, row) for row in piece]
            lines += notes
            chunks.append(chunk(doc, [block], "\n".join(lines), conditions))
    return chunks


def text_chunks(doc: dict, blocks: list[dict]) -> list[dict]:
    """같은 절의 이어진 문단을 MAX_CHARS 안에서 합친다."""
    chunks = []
    current: list[dict] = []
    size = 0

    def flush() -> None:
        nonlocal current, size
        if current:
            body = "\n".join(block["text"] for block in current)
            chunks.append(chunk(doc, current, body, {}))
        current, size = [], 0

    for block in blocks:
        length = len(block["text"])
        if current and (
            block.get("section") != current[-1].get("section")
            or size + length > MAX_CHARS
        ):
            flush()
        current.append(block)
        size += length
    flush()
    return chunks


def is_junk(block: dict) -> bool:
    """그림의 번호만 남은 블록처럼 검색에 쓸 수 없는 조각."""
    return block["type"] == "figure" and len(block["text"]) < MIN_FIGURE_CHARS


def document_chunks(doc: dict, blocks: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    pending: list[dict] = []
    # 지점 목록처럼 질문과 상관없는 단어로 조치 근거를 밀어내는 표는 검색에 넣지 않는다
    excluded = doc.get("exclude_tables", [])
    for index, block in enumerate(blocks):
        if is_junk(block) or block.get("table_id") in excluded:
            continue
        if block["type"] == "table":
            chunks += text_chunks(doc, pending)
            pending = []
            notes = []
            # 표 바로 뒤의 비고·각주는 표의 조건이므로 함께 둔다
            for after in blocks[index + 1 : index + 3]:
                if after["type"] == "footnote" or after.get("text", "").startswith(
                    ("비고", "*")
                ):
                    notes.append(" ".join(after["text"].split()))
            chunks += table_chunks(doc, block, notes)
        elif block["type"] == "figure":
            chunks += text_chunks(doc, pending)
            pending = []
            chunks.append(chunk(doc, [block], block["text"], {}))
        else:
            pending.append(block)
    return chunks + text_chunks(doc, pending)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    parsed_files = {}
    chunks: list[dict] = []
    for doc in manifest["documents"]:
        path = (
            ROOT
            / "data/parsed"
            / PARSE_RUN_ID
            / "reviewed"
            / f"{doc['document_id']}.json"
        )
        parsed_files[doc["document_id"]] = hashlib.sha256(path.read_bytes()).hexdigest()
        blocks = json.loads(path.read_text(encoding="utf-8"))["blocks"]
        chunks += document_chunks(doc, blocks)

    set_id = f"{PARSE_RUN_ID}-{CHUNKING_VERSION}"
    for number, item in enumerate(chunks, start=1):
        item["chunk_id"] = f"{set_id}-{item['document_id']}-{number:03d}"
        item["text_sha256"] = sha256(item["text"])

    out = ROOT / "data" / "chunks" / f"{set_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "chunk_set_id": set_id,
                "parse_run_id": PARSE_RUN_ID,
                "chunking_version": CHUNKING_VERSION,
                "parsed_files": parsed_files,
                "documents": manifest["documents"],
                "chunks": chunks,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n"
    )
    sizes = sorted(len(item["text"]) for item in chunks)
    print(
        f"청크 {len(chunks)}개, 글자 수 중앙값 {sizes[len(sizes) // 2]}, 최대 {sizes[-1]}"
    )
    print(re.sub(str(ROOT), ".", str(out)))


if __name__ == "__main__":
    main()
