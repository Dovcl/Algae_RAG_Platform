"""질문을 bge-m3로 임베딩해 컬렉션에서 가까운 청크를 찾는다. 코사인 거리 기준 정확 검색이다."""

import re
from collections import Counter
from dataclasses import dataclass, replace
from itertools import pairwise

import httpx2
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Chunk, ChunkEmbedding, Collection, Document

# 조류경보제 운영 매뉴얼의 구간별 발령·해제 기준표
CRITERIA_TABLE = "표 6"
STAGE_WORDS = ("관심 단계", "경계 단계", "조류대발생")
PARTICLE = re.compile(r"(에서|으로|은|는|이|가|을|를|의|에|로|과|와|도|만)$")
PHRASE_LIMIT = 4  # 이보다 많은 청크에 나오는 표현은 질문을 가려내지 못한다
MAX_PHRASE_HITS = 3


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
    related_chunk_ids: tuple[str, ...] = ()


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


def page_chunks(session: Session, document_id: str, page: int) -> list[Hit]:
    """표와 같은 쪽의 설명·조건을 함께 읽는다."""
    if not settings.rag_collection:
        return []
    collection = ready_collection(session)
    rows = session.execute(
        select(Chunk, Document)
        .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.chunk_id)
        .join(Document, Document.document_id == Chunk.document_id)
        .where(
            ChunkEmbedding.collection_id == collection.collection_id,
            Chunk.document_id == document_id,
            Chunk.pdf_page_start <= page,
            Chunk.pdf_page_end >= page,
        )
        .order_by(Chunk.chunk_id)
    )
    return [to_hit(chunk, document, 0.0) for chunk, document in rows]


def merge_table_context(hit: Hit, neighbors: list[Hit]) -> Hit:
    """같은 표의 머리글·행을 연결하고 중복 열 이름에 하위 머리글을 붙인다."""
    header = next(
        (line for line in hit.text.splitlines() if line.startswith("열: ")), None
    )
    if header is None:
        return hit
    peers = [
        h
        for h in neighbors
        if h.document_id == hit.document_id
        and h.pdf_page == hit.pdf_page
        and h.table_id == hit.table_id
        and (hit.table_id is not None or h.section == hit.section)
        and header in h.text.splitlines()
    ]
    if not peers:
        return hit
    columns = header.removeprefix("열: ").split(" / ")
    counts = Counter(columns)
    lines = list(dict.fromkeys(line for h in peers for line in h.text.splitlines()))
    labels = columns
    for line in lines:
        cells = [part.split(": ", 1) for part in line.split(" | ")]
        if len(cells) != len(columns) or any(len(cell) != 2 for cell in cells):
            continue
        keys, values = zip(*cells, strict=True)
        if list(keys) == columns and any(
            v == c for v, c in zip(values, columns, strict=True)
        ):
            labels = [
                f"{c}({v})" if counts[c] > 1 and c != v else c
                for c, v in zip(columns, values, strict=True)
            ]
            break
    rewritten = []
    for line in lines:
        cells = [part.split(": ", 1) for part in line.split(" | ")]
        if (
            len(cells) == len(columns)
            and all(len(cell) == 2 for cell in cells)
            and [cell[0] for cell in cells] == columns
        ):
            line = " | ".join(
                f"{label}: {cell[1]}" for label, cell in zip(labels, cells, strict=True)
            )
        rewritten.append(line)
    return replace(
        hit,
        text="\n".join(rewritten),
        related_chunk_ids=tuple(
            h.chunk_id for h in peers if h.chunk_id != hit.chunk_id
        ),
    )


def complete_table_context(session: Session, hits: list[Hit]) -> list[Hit]:
    cache: dict[tuple[str, int], list[Hit]] = {}
    completed = []
    seen = set()
    for hit in hits:
        if hit.chunk_id in seen:
            continue
        if "\n열: " in hit.text:
            key = (hit.document_id, hit.pdf_page)
            if key not in cache:
                cache[key] = page_chunks(session, *key)
            hit = merge_table_context(hit, cache[key])
        completed.append(hit)
        seen.update((hit.chunk_id, *hit.related_chunk_ids))
    return completed


def question_phrases(question: str) -> list[str]:
    """질문의 이어진 두 단어. 조사를 떼어 문서의 표현과 맞춘다."""
    words = [PARTICLE.sub("", word.strip("?!.,")) for word in question.split()]
    return [
        f"{first} {second}"
        for first, second in pairwise(words)
        if len(first) >= 2 and len(second) >= 2
    ]


def phrase_chunks(session: Session, question: str) -> list[Hit]:
    """질문의 표현이 그대로 있는 청크. 긴 표의 한 행은 벡터 검색에서 밀리기 쉽다."""
    if settings.rag_collection is None:
        return []
    collection = ready_collection(session)
    found: dict[str, Hit] = {}
    # 소수 수치처럼 드문 표현은 표의 행·조건을 찾는 단서로 함께 쓴다.
    literals = list(dict.fromkeys(re.findall(r"(?<![\d.])\d+\.\d+(?![\d.])", question)))
    ranges = list(
        dict.fromkeys(
            re.sub(r"\s+", "", value).replace("～", "~").replace("∼", "~")
            for value in re.findall(
                r"\d+(?:\.\d+)?\s*[~～∼]\s*\d+(?:\.\d+)?\s*(?:%|mg/L|ng/L)", question
            )
        )
    )
    expressions = [(value, True) for value in ranges] + [
        (value, False) for value in [*literals, *question_phrases(question)]
    ]
    # 질문의 물질·투입량을 함께 찾아, 같은 mg/L라도 다른 공정의 값이 앞서지 않게 한다.
    anchors = [
        re.sub(r"\s+", "", value).removeprefix("분말")
        for value in re.findall(
            r"(?:분말활성탄|활성탄|오존|염소)\s*\d+(?:\.\d+)?\s*(?:mg/L|ng/L)", question
        )
    ]
    expressions = [(value, True) for value in anchors] + expressions
    for phrase, compact in expressions:
        column = (
            func.regexp_replace(Chunk.text, r"\s+", "", "g") if compact else Chunk.text
        )
        rows = session.execute(
            select(Chunk, Document)
            .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.chunk_id)
            .join(Document, Document.document_id == Chunk.document_id)
            .where(
                ChunkEmbedding.collection_id == collection.collection_id,
                column.contains(phrase, autoescape=True),
            )
            .order_by(Chunk.chunk_id)
            .limit(PHRASE_LIMIT + 1)
        ).all()
        if len(rows) <= PHRASE_LIMIT:
            for chunk, document in rows:
                found.setdefault(chunk.chunk_id, to_hit(chunk, document, 0.0))
    return list(found.values())


def is_criteria_question(question: str) -> bool:
    """조류경보 발령·해제 기준을 묻는가. 조류예측 단계 질문은 다른 표를 쓰므로 뺀다."""
    if "예측" in question:
        return False
    if any(word in question for word in ("기준", "얼마", "몇")):
        return any(word in question for word in ("발령", "해제", "경보", "단계"))
    if any(word in question for word in ("뭐야", "뜻", "의미")):
        return any(word in question for word in STAGE_WORDS)
    return "언제" in question and any(word in question for word in ("발령", "해제"))


def search_evidence(
    session: Session, question: str, query: str, top_k: int
) -> list[Hit]:
    """벡터 검색 결과에, 공식 기준표(표 6)와 질문의 표현이 그대로 있는 청크를 앞에 넣는다."""
    hits = search(session, query, top_k)
    if is_criteria_question(question):
        pinned = table_chunks(session, CRITERIA_TABLE)
    else:
        pinned = phrase_chunks(session, question)[:MAX_PHRASE_HITS]
    front = {hit.chunk_id: hit for hit in pinned}
    return [*front.values(), *(hit for hit in hits if hit.chunk_id not in front)]
