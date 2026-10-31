"""Upstage Document Parse Enhanced로 PDF를 쪽 단위로 파싱해 원본 응답을 저장한다. parsers/.venv에서 실행한다.

쪽마다 따로 보내고 끝난 쪽은 건너뛰므로, 중간에 멈춰도 같은 쪽에 비용이 두 번 나가지 않는다.
보낸 쪽 수를 usage.json에 기록하고 상한을 넘으면 멈춘다. 모델은 별칭 대신 버전을 고정한다.

    parsers/.venv/bin/python parsers/run_upstage.py --gold
    parsers/.venv/bin/python parsers/run_upstage.py DOC02 1 116
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "parser_eval" / "upstage-enhanced"
MANIFEST = ROOT / "corpus" / "manifest.json"
URL = "https://api.upstage.ai/v1/document-digitization"
MODEL = "document-parse-260930"
PRICE_PER_PAGE = 0.03
BUDGET = 20.0


def api_key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("UPSTAGE_API_KEY="):
            return line.split("=", 1)[1].strip()
    sys.exit(".env에 UPSTAGE_API_KEY가 없습니다.")


def spent_pages() -> int:
    usage = OUT / "usage.json"
    return json.loads(usage.read_text())["pages"] if usage.exists() else 0


def parse_page(key: str, pdf: Path, doc_id: str, page: int) -> None:
    stem = OUT / f"{doc_id}_p{page:03d}"
    if stem.with_suffix(".html").exists():
        return
    pages = spent_pages()
    if (pages + 1) * PRICE_PER_PAGE > BUDGET:
        sys.exit(f"비용 상한 ${BUDGET}에 도달했습니다.")

    with tempfile.TemporaryDirectory() as folder:
        single = Path(folder) / "page.pdf"
        subprocess.run(
            ["pdfseparate", "-f", str(page), "-l", str(page), str(pdf), str(single)],
            check=True,
        )
        started = time.perf_counter()
        with single.open("rb") as file:
            response = requests.post(
                URL,
                headers={"Authorization": f"Bearer {key}"},
                files={"document": file},
                data={
                    "model": MODEL,
                    "mode": "enhanced",
                    "ocr": "auto",
                    "coordinates": "true",
                    "output_formats": '["html", "markdown"]',
                },
                timeout=180,
            )
    if not response.ok:
        sys.exit(f"{doc_id} p{page}: HTTP {response.status_code} {response.text[:300]}")

    body = response.json()
    (OUT / "usage.json").write_text(json.dumps({"pages": pages + 1, "model": MODEL}))
    stem.with_suffix(".json").write_text(
        json.dumps(body, ensure_ascii=False), encoding="utf-8"
    )
    stem.with_suffix(".html").write_text(body["content"]["html"], encoding="utf-8")
    print(f"{doc_id} p{page}: {time.perf_counter() - started:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", nargs="*", help="문서 ID, 시작 쪽, 끝 쪽")
    parser.add_argument("--gold", action="store_true", help="평가 묶음 쪽만 실행")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    key = api_key()
    files = {
        doc["document_id"]: ROOT / doc["file"]
        for doc in json.loads(MANIFEST.read_text(encoding="utf-8"))["documents"]
    }

    if args.gold:
        gold = json.loads((ROOT / "corpus" / "parser_gold.json").read_text())
        for bundle in gold["bundles"].values():
            for page in bundle["pdf_pages"]:
                doc_id = bundle["document_id"]
                parse_page(key, files[doc_id], doc_id, page)
        return
    if len(args.target) != 3:
        sys.exit("문서 ID, 시작 쪽, 끝 쪽을 주세요.")
    doc_id, first, last = args.target
    for page in range(int(first), int(last) + 1):
        parse_page(key, files[doc_id], doc_id, page)


if __name__ == "__main__":
    main()
