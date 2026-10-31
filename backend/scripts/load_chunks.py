"""data/chunks의 청크를 임베딩해 DB에 넣는다. 백엔드 컨테이너에서 실행한다.

청크 묶음과 컬렉션은 새 ID로 만들고, 임베딩이 끝난 뒤에만 ready로 바꾼다.
그래서 실패해도 사용 중인 컬렉션이 반쯤 바뀌지 않는다. 같은 입력을 다시 넣으면 건너뛴다.

    docker compose exec backend python scripts/load_chunks.py upstage-enhanced-v1-chunk-v1
"""

import argparse
import json
from pathlib import Path

import httpx2
from sqlalchemy import select

from app.core.config import settings
from app.db.models import (
    EMBEDDING_DIM,
    Chunk,
    ChunkEmbedding,
    ChunkSet,
    Collection,
    Document,
)
from app.db.session import SessionLocal

CHUNKS = Path("/data/chunks")
BATCH = 16


def model_digest() -> str:
    tags = httpx2.get(f"{settings.ollama_base_url}/api/tags", timeout=30).json()
    for model in tags["models"]:
        if model["name"].split(":")[0] == settings.embed_model:
            return model["digest"]
    raise SystemExit(f"Ollama에 {settings.embed_model}이 없습니다.")


def embed(texts: list[str]) -> list[list[float]]:
    response = httpx2.post(
        f"{settings.ollama_base_url}/api/embed",
        json={"model": settings.embed_model, "input": texts, "truncate": False},
        timeout=300,
    )
    response.raise_for_status()
    vectors = response.json()["embeddings"]
    if any(len(vector) != EMBEDDING_DIM for vector in vectors):
        raise SystemExit("임베딩 차원이 DB와 다릅니다.")
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("chunk_set_id")
    args = parser.parse_args()

    data = json.loads((CHUNKS / f"{args.chunk_set_id}.json").read_text("utf-8"))
    digest = model_digest()
    collection_id = f"{args.chunk_set_id}-{settings.embed_model}"[:64]

    with SessionLocal() as session:
        if session.get(Collection, collection_id) is not None:
            print(f"이미 있습니다: {collection_id}")
            return

        for doc in data["documents"]:
            session.merge(
                Document(
                    document_id=doc["document_id"],
                    title=doc["title"],
                    publisher=doc["publisher"],
                    edition=doc["edition"],
                    published_on=doc["published_on"],
                    file_name=Path(doc["file"]).name,
                    sha256=doc["sha256"],
                    source_url=doc["source_url"],
                )
            )
        session.add(
            ChunkSet(
                chunk_set_id=data["chunk_set_id"],
                parse_run_id=data["parse_run_id"],
                parsed_files=data["parsed_files"],
                chunking_version=data["chunking_version"],
            )
        )
        session.flush()
        for item in data["chunks"]:
            session.add(
                Chunk(
                    chunk_id=item["chunk_id"],
                    chunk_set_id=data["chunk_set_id"],
                    document_id=item["document_id"],
                    text=item["text"],
                    text_sha256=item["text_sha256"],
                    pdf_page_start=item["pdf_page_start"],
                    pdf_page_end=item["pdf_page_end"],
                    printed_page_start=item["printed_page_start"],
                    printed_page_end=item["printed_page_end"],
                    section=item["section"],
                    table_id=item["table_id"],
                    conditions=item["conditions"],
                )
            )
        session.add(
            Collection(
                collection_id=collection_id,
                chunk_set_id=data["chunk_set_id"],
                embed_model=settings.embed_model,
                embed_model_digest=digest,
                dimensions=EMBEDDING_DIM,
                status="building",
            )
        )
        session.flush()

        chunks = data["chunks"]
        for start in range(0, len(chunks), BATCH):
            part = chunks[start : start + BATCH]
            vectors = embed([item["text"] for item in part])
            for item, vector in zip(part, vectors, strict=True):
                session.add(
                    ChunkEmbedding(
                        collection_id=collection_id,
                        chunk_id=item["chunk_id"],
                        embedding=vector,
                    )
                )
            session.flush()

        collection = session.scalar(
            select(Collection).where(Collection.collection_id == collection_id)
        )
        collection.status = "ready"
        session.commit()
    print(f"{collection_id}: 청크 {len(chunks)}개 적재")


if __name__ == "__main__":
    main()
