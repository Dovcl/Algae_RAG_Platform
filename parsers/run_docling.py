"""Docling으로 PDF를 쪽 단위로 파싱해 원본 결과를 저장한다. parsers/.venv에서 실행한다.

OCR은 끈다. 세 PDF 모두 글자 층이 있고, 숫자가 OCR을 거치면 오류가 생길 수 있다.
이미 저장한 쪽은 건너뛰므로 중간에 멈춰도 이어서 돌릴 수 있다.

    parsers/.venv/bin/python parsers/run_docling.py --gold
    parsers/.venv/bin/python parsers/run_docling.py DOC02 1 116
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.exceptions import ConversionError

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "parser_eval" / "docling"
MANIFEST = ROOT / "corpus" / "manifest.json"


def converter() -> DocumentConverter:
    options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
    options.table_structure_options.mode = TableFormerMode.ACCURATE
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def parse_page(engine: DocumentConverter, pdf: Path, doc_id: str, page: int) -> None:
    stem = OUT / f"{doc_id}_p{page:03d}"
    if stem.with_suffix(".html").exists():
        return
    with tempfile.TemporaryDirectory() as folder:
        single = Path(folder) / "page.pdf"
        subprocess.run(
            ["pdfseparate", "-f", str(page), "-l", str(page), str(pdf), str(single)],
            check=True,
        )
        started = time.perf_counter()
        try:
            document = engine.convert(single).document
        except ConversionError as error:
            print(f"{doc_id} p{page}: 실패 {str(error)[:80]}", flush=True)
            return
        seconds = round(time.perf_counter() - started, 1)
    stem.with_suffix(".json").write_text(
        json.dumps(document.export_to_dict(), ensure_ascii=False), encoding="utf-8"
    )
    stem.with_suffix(".html").write_text(document.export_to_html(), encoding="utf-8")
    print(f"{doc_id} p{page}: {seconds}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", nargs="*", help="문서 ID, 시작 쪽, 끝 쪽")
    parser.add_argument("--gold", action="store_true", help="평가 묶음 쪽만 실행")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "meta.json").write_text(
        json.dumps({"docling": version("docling"), "ocr": False, "table": "accurate"})
    )
    files = {
        doc["document_id"]: ROOT / doc["file"]
        for doc in json.loads(MANIFEST.read_text(encoding="utf-8"))["documents"]
    }
    engine = converter()

    if args.gold:
        gold = json.loads((ROOT / "corpus" / "parser_gold.json").read_text())
        for bundle in gold["bundles"].values():
            for page in bundle["pdf_pages"]:
                doc_id = bundle["document_id"]
                parse_page(engine, files[doc_id], doc_id, page)
        return
    if len(args.target) != 3:
        sys.exit("문서 ID, 시작 쪽, 끝 쪽을 주세요.")
    doc_id, first, last = args.target
    for page in range(int(first), int(last) + 1):
        parse_page(engine, files[doc_id], doc_id, page)


if __name__ == "__main__":
    main()
