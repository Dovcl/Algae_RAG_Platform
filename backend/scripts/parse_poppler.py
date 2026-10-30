"""Poppler(pdftotext -layout) 출력을 공통 형식 parsed-v1로 바꾼다.

맥 호스트에서 실행한다 (Poppler는 컨테이너에 없다). 표준 라이브러리만 쓴다.
모든 블록은 review.status="raw"다. 표 구조화와 페이지를 넘는 문단 결합은 검수 단계에서 한다.

    cd backend && source .venv/bin/activate
    python scripts/parse_poppler.py DOC02
"""

import argparse
import hashlib
import json
import re
import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "corpus" / "manifest.json"
PARSE_RUN_ID = "poppler-layout-v1"
OPTIONS = ["-layout"]

# "1.3.2 단계별 발령기준", "2. 조류예측과 ...", "제1절 ..." 같은 절 제목.
# 블록의 첫 줄이면서 짧을 때만 제목으로 본다. "2020년 ..."처럼 숫자로 시작하는 본문을 잡지 않게
SECTION = re.compile(r"^(\d+(\.\d+)*\.?|제\s*\d+\s*[부장절])\s+\S")
SECTION_MAX_LEN = 40
CAPTION = re.compile(r"^(표|그림)\s*([\d-]+)\s*(.*)$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pdftotext_version() -> str:
    # 판에 따라 -v의 종료 코드가 0이 아닐 수 있어 check=False로 두고 출력만 본다
    result = subprocess.run(
        ["pdftotext", "-v"], check=False, capture_output=True, text=True
    )
    match = re.search(r"pdftotext version (\S+)", result.stderr + result.stdout)
    if match is None:
        raise SystemExit("pdftotext 버전을 읽지 못했습니다. Poppler 설치를 확인하세요.")
    return match.group(1)


def extract_page(pdf: Path, page: int) -> str:
    args = ["pdftotext", *OPTIONS, "-enc", "UTF-8", "-f", str(page), "-l", str(page)]
    result = subprocess.run(
        [*args, str(pdf), "-"], capture_output=True, text=True, check=True
    )
    return result.stdout


def split_blocks(text: str, strip_lines: list[re.Pattern[str]]) -> list[list[str]]:
    """빈 줄을 경계로 블록을 나눈다. 머리말·쪽번호 줄은 빈 줄로 취급한다."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.replace("\f", "").splitlines():
        line = line.rstrip()
        if not line.strip() or any(p.match(line.strip()) for p in strip_lines):
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)
    return blocks


def page_blocks(
    text: str,
    doc_id: str,
    pdf_page: int,
    printed_page: int,
    section: str | None,
    strip_lines: list[re.Pattern[str]],
) -> tuple[list[dict], str | None]:
    """한 페이지의 블록 목록과, 다음 페이지로 이어질 절 제목을 돌려준다."""
    raw = split_blocks(text, strip_lines)
    out = []
    for n, lines in enumerate(raw, start=1):
        if not lines:
            continue  # 앞 표 제목에 합쳐진 블록
        first = lines[0].strip()
        if SECTION.match(first) and len(first) <= SECTION_MAX_LEN:
            section = " ".join(first.split())
            lines = lines[1:]
            if not lines:
                continue
            first = lines[0].strip()
        block = {
            "block_id": f"{doc_id}-p{pdf_page:03d}-b{n:02d}",
            "type": "paragraph",
            "pdf_page_start": pdf_page,
            "pdf_page_end": pdf_page,
            "printed_page_start": printed_page,
            "printed_page_end": printed_page,
            "section": section,
        }
        caption = CAPTION.match(first)
        if caption:
            # 제목만 있는 블록이면 바로 다음 블록(표 본문)을 붙인다
            if len(lines) == 1 and n < len(raw):
                lines = lines + raw[n]
                raw[n] = []
            # 표·그림은 열 배치를 남겨 검수할 때 원본과 대조한다
            block["type"] = "table" if caption.group(1) == "표" else "figure"
            block["table_id"] = f"{caption.group(1)} {caption.group(2)}"
            block["caption"] = " ".join(caption.group(3).split())
            block["text"] = textwrap.dedent("\n".join(lines))
        else:
            block["text"] = "\n".join(line.strip() for line in lines)
        block["review"] = {"status": "raw"}
        out.append(block)
    return out, section


def parse_document(doc: dict) -> dict:
    pdf = ROOT / doc["file"]
    if sha256(pdf) != doc["sha256"]:
        raise SystemExit(
            f"{doc['document_id']}: 원본 해시가 manifest와 다릅니다 ({pdf})"
        )
    strip_lines = [re.compile(p) for p in doc["strip_lines"]]

    blocks: list[dict] = []
    section = None
    for pages in doc["page_ranges"]:
        for page in range(pages["pdf_start"], pages["pdf_end"] + 1):
            text = extract_page(pdf, page)
            new, section = page_blocks(
                text,
                doc["document_id"],
                page,
                page - pages["printed_offset"],
                section,
                strip_lines,
            )
            blocks.extend(new)

    # 실행 시각은 넣지 않는다. 같은 입력이면 같은 파일(같은 해시)이 나와야 재적재를 건너뛸 수 있다
    return {
        "format_version": "parsed-v1",
        "parse_run_id": PARSE_RUN_ID,
        "document_id": doc["document_id"],
        "source_sha256": doc["sha256"],
        "parser": {
            "name": "pdftotext",
            "version": pdftotext_version(),
            "options": OPTIONS,
        },
        "normalization": [
            "manifest의 strip_lines에 맞는 머리말·쪽번호 줄 제거",
            "문단은 줄 앞뒤 공백만 제거하고 줄바꿈 유지",
            "표·그림은 공통 들여쓰기만 제거하고 배치 유지",
        ],
        "page_ranges": doc["page_ranges"],
        "blocks": blocks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("document_ids", nargs="+", help="예: DOC01 DOC02")
    args = parser.parse_args()

    docs = {d["document_id"]: d for d in json.loads(MANIFEST.read_text())["documents"]}
    out_dir = ROOT / "data" / "parsed" / PARSE_RUN_ID / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    for doc_id in args.document_ids:
        if doc_id not in docs:
            raise SystemExit(f"manifest에 없는 문서입니다: {doc_id}")
        if not docs[doc_id]["page_ranges"]:
            raise SystemExit(f"{doc_id}: manifest에 page_ranges가 비어 있습니다")
        result = parse_document(docs[doc_id])
        out = out_dir / f"{doc_id}.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        types = [b["type"] for b in result["blocks"]]
        counts = {t: types.count(t) for t in sorted(set(types))}
        print(f"{doc_id}: 블록 {len(types)}개 {counts} → {out.relative_to(ROOT)}")
        print(f"  sha256 {sha256(out)}")


if __name__ == "__main__":
    main()
