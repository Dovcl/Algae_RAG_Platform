"""질문을 bge-m3로 임베딩해 컬렉션에서 가까운 청크를 찾는다. 코사인 거리 기준 정확 검색이다."""

from dataclasses import dataclass

import httpx2
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Chunk, ChunkEmbedding, Collection, Document


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    document_id: str
    document: str
    publisher: str
    page: int | None
    pdf_page: int
    section: str | None
    table_id: str | None
    text: str
    distance: float  # 코사인 거리. 작을수록 가깝고 정답 확률은 아니다


def embed_query(question: str) -> list[float]:
    response = httpx2.post(
        f"{settings.ollama_base_url}/api/embed",
        json={"model": settings.embed_model, "input": question, "truncate": False},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def search(session: Session, question: str, top_k: int) -> list[Hit]:
    if settings.rag_collection is None:
        return []
    collection = session.get(Collection, settings.rag_collection)
    if collection is None or collection.status != "ready":
        raise RuntimeError(f"사용할 수 없는 컬렉션입니다: {settings.rag_collection}")

    distance = ChunkEmbedding.embedding.cosine_distance(embed_query(question))
    rows = session.execute(
        select(Chunk, Document, distance.label("distance"))
        .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.chunk_id)
        .join(Document, Document.document_id == Chunk.document_id)
        .where(ChunkEmbedding.collection_id == collection.collection_id)
        .order_by(distance)
        .limit(top_k)
    )
    return [
        Hit(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document=document.title,
            publisher=document.publisher,
            page=chunk.printed_page_start,
            pdf_page=chunk.pdf_page_start,
            section=chunk.section,
            table_id=chunk.table_id,
            text=chunk.text,
            distance=float(gap),
        )
        for chunk, document, gap in rows
    ]
