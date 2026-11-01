"""질문을 bge-m3로 임베딩해 컬렉션에서 가까운 청크를 찾는다. 코사인 거리 기준 정확 검색이다."""

from dataclasses import dataclass

import httpx2
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Chunk, ChunkEmbedding, Collection, Document

# 조류경보제 운영 매뉴얼의 구간별 발령·해제 기준표
CRITERIA_TABLE = "표 6"


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


def to_hit(chunk: Chunk, document: Document, distance: float) -> Hit:
    return Hit(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        document=document.title,
        publisher=document.publisher,
        page=chunk.printed_page_start,
        pdf_page=chunk.pdf_page_start,
        section=chunk.section,
        table_id=chunk.table_id,
        text=chunk.text,
        distance=distance,
    )


def ready_collection(session: Session) -> Collection:
    collection = session.get(Collection, settings.rag_collection)
    if collection is None or collection.status != "ready":
        raise RuntimeError(f"사용할 수 없는 컬렉션입니다: {settings.rag_collection}")
    return collection


def search(session: Session, question: str, top_k: int) -> list[Hit]:
    if settings.rag_collection is None:
        return []
    collection = ready_collection(session)

    distance = ChunkEmbedding.embedding.cosine_distance(embed_query(question))
    rows = session.execute(
        select(Chunk, Document, distance.label("distance"))
        .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.chunk_id)
        .join(Document, Document.document_id == Chunk.document_id)
        .where(ChunkEmbedding.collection_id == collection.collection_id)
        .order_by(distance)
        .limit(top_k)
    )
    return [to_hit(chunk, document, float(gap)) for chunk, document, gap in rows]


def table_chunks(session: Session, table_id: str) -> list[Hit]:
    """표의 청크 전부. 구간별로 나뉜 표는 한쪽만 검색되면 기준을 잘못 말하게 된다."""
    if settings.rag_collection is None:
        return []
    collection = ready_collection(session)
    rows = session.execute(
        select(Chunk, Document)
        .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.chunk_id)
        .join(Document, Document.document_id == Chunk.document_id)
        .where(
            ChunkEmbedding.collection_id == collection.collection_id,
            Chunk.table_id == table_id,
        )
        .order_by(Chunk.chunk_id)
    )
    return [to_hit(chunk, document, 0.0) for chunk, document in rows]


def is_criteria_question(question: str) -> bool:
    """조류경보 발령·해제 기준을 묻는가. 조류예측 단계 질문은 다른 표를 쓰므로 뺀다."""
    if "예측" in question:
        return False
    if "기준" in question:
        return any(word in question for word in ("발령", "해제", "경보", "단계"))
    return "언제" in question and any(word in question for word in ("발령", "해제"))


def search_evidence(
    session: Session, question: str, query: str, top_k: int
) -> list[Hit]:
    """벡터 검색 결과에, 기준 질문이면 공식 기준표(표 6)를 앞에 넣는다."""
    hits = search(session, query, top_k)
    if not is_criteria_question(question):
        return hits
    tables = table_chunks(session, CRITERIA_TABLE)
    pinned = {hit.chunk_id for hit in tables}
    return tables + [hit for hit in hits if hit.chunk_id not in pinned]
