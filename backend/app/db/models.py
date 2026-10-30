from datetime import date, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

# bge-m3의 실제 응답 차원. 다른 차원의 모델을 쓰려면 새 마이그레이션이 필요하다.
EMBEDDING_DIM = 1024


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Document(Base):
    """원본 PDF. 파서가 바뀌어도 한 줄이다."""

    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    publisher: Mapped[str] = mapped_column(Text)
    edition: Mapped[str] = mapped_column(Text)
    published_on: Mapped[date | None]
    file_name: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64))
    source_url: Mapped[str | None] = mapped_column(Text)


class ChunkSet(Base):
    """어떤 파싱 결과를 어떤 청킹 규칙으로 잘랐는가."""

    __tablename__ = "chunk_sets"

    chunk_set_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    parse_run_id: Mapped[str] = mapped_column(String(64))
    # {"DOC02": "<파싱 결과 파일의 sha256>", ...}
    parsed_files: Mapped[dict[str, Any]] = mapped_column(JSONB)
    chunking_version: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Chunk(Base):
    """검색 단위. 벡터는 따로 둔다 (임베딩만 바꿀 때 재사용)."""

    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    chunk_set_id: Mapped[str] = mapped_column(ForeignKey("chunk_sets.chunk_set_id"))
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.document_id"))
    text: Mapped[str] = mapped_column(Text)
    text_sha256: Mapped[str] = mapped_column(String(64))
    pdf_page_start: Mapped[int]
    pdf_page_end: Mapped[int]
    printed_page_start: Mapped[int | None]
    printed_page_end: Mapped[int | None]
    section: Mapped[str | None] = mapped_column(Text)
    table_id: Mapped[str | None] = mapped_column(String(32))
    # 적용 구간·단계 등. 예: {"구간": "친수활동"}
    conditions: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


class Collection(Base):
    """검색 대상 = 청크 묶음 × 임베딩 설정."""

    __tablename__ = "collections"

    collection_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    chunk_set_id: Mapped[str] = mapped_column(ForeignKey("chunk_sets.chunk_set_id"))
    embed_model: Mapped[str] = mapped_column(String(64))
    embed_model_digest: Mapped[str] = mapped_column(String(64))
    dimensions: Mapped[int]
    # building → ready / failed. 검색은 ready만 허용한다.
    status: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"

    collection_id: Mapped[str] = mapped_column(
        ForeignKey("collections.collection_id"), primary_key=True
    )
    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("chunks.chunk_id"), primary_key=True
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
