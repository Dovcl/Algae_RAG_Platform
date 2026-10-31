"""개발용 질문으로 검색만 평가한다. 백엔드 컨테이너에서 실행한다. 질문은 data/chunks 옆이 아니라 표준 입력으로 받는다.

docker compose exec -T -e PYTHONPATH=/code backend python scripts/eval_search.py < corpus/dev_questions.json
"""

import json
import sys

from app.core.config import settings
from app.db.session import SessionLocal
from app.rag.retrieve import search


def main() -> None:
    questions = json.load(sys.stdin)
    top = settings.rag_top_k
    found = 0
    with SessionLocal() as session:
        for item in questions:
            hits = search(session, item["question"], top)
            pages = [(hit.document_id, hit.pdf_page) for hit in hits]
            expected = {(e["document_id"], e["pdf_page"]) for e in item["expected"]}
            ranks = [n for n, page in enumerate(pages, start=1) if page in expected]
            found += bool(ranks)
            rank = ranks[0] if ranks else "없음"
            print(f"{rank} | {item['question']} | {pages[:3]} | {hits[0].distance:.3f}")
    print(f"top-{top} 적중 {found}/{len(questions)}")


if __name__ == "__main__":
    main()
