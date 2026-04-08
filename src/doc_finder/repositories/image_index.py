from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import sqrt
from typing import Protocol


@dataclass(slots=True)
class ImageDocument:
    unit_id: int
    data_id: int
    image_id: int
    asset_path: str
    sha256: str
    file_size: int
    width: int
    height: int
    keyword_tags: list[str]
    normalized_tags: list[str]
    tag_text_projection: str
    confidence: float
    review_status: str
    embedding: list[float]
    tagged_at: datetime

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.unit_id, self.data_id, self.image_id)


@dataclass(slots=True)
class RejectedAsset:
    asset_path: str
    reason: str
    detail: str | None = None
    recorded_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class SearchCandidate:
    document: ImageDocument
    matched_tags: list[str]
    score: float
    source: str


class ImageIndexRepository(Protocol):
    def ensure_schema(self) -> None: ...

    def has_sha256(self, sha256: str) -> bool: ...

    def upsert_image(self, document: ImageDocument) -> bool: ...

    def record_reject(self, rejected: RejectedAsset) -> None: ...

    def search_by_normalized_tags(
        self,
        normalized_tags: list[str],
        review_status: str | None,
        limit: int,
    ) -> list[SearchCandidate]: ...

    def search_by_embedding(
        self,
        embedding: list[float],
        review_status: str | None,
        limit: int,
    ) -> list[SearchCandidate]: ...


class InMemoryImageIndexRepository:
    def __init__(self) -> None:
        self._documents_by_key: dict[tuple[int, int, int], ImageDocument] = {}
        self._key_by_sha256: dict[str, tuple[int, int, int]] = {}
        self._rejects: list[RejectedAsset] = []

    def ensure_schema(self) -> None:
        return None

    def has_sha256(self, sha256: str) -> bool:
        return sha256 in self._key_by_sha256

    def upsert_image(self, document: ImageDocument) -> bool:
        existing_key = self._key_by_sha256.get(document.sha256)
        if existing_key is not None:
            return False

        self._documents_by_key[document.key] = document
        self._key_by_sha256[document.sha256] = document.key
        return True

    def record_reject(self, rejected: RejectedAsset) -> None:
        self._rejects.append(rejected)

    def search_by_normalized_tags(
        self,
        normalized_tags: list[str],
        review_status: str | None,
        limit: int,
    ) -> list[SearchCandidate]:
        normalized_query_tags = set(normalized_tags)
        candidates: list[SearchCandidate] = []
        for document in self._documents_by_key.values():
            if review_status and document.review_status != review_status:
                continue

            matched_tags = sorted(normalized_query_tags.intersection(document.normalized_tags))
            if not matched_tags:
                continue

            candidates.append(
                SearchCandidate(
                    document=document,
                    matched_tags=matched_tags,
                    score=1.0,
                    source="exact",
                )
            )

        candidates.sort(
            key=lambda candidate: (
                -candidate.document.confidence,
                -candidate.document.tagged_at.timestamp(),
            )
        )
        return candidates[:limit]

    def search_by_embedding(
        self,
        embedding: list[float],
        review_status: str | None,
        limit: int,
    ) -> list[SearchCandidate]:
        candidates: list[SearchCandidate] = []
        for document in self._documents_by_key.values():
            if review_status and document.review_status != review_status:
                continue

            score = _cosine_similarity(embedding, document.embedding)
            if score <= 0.0:
                continue

            candidates.append(
                SearchCandidate(
                    document=document,
                    matched_tags=[],
                    score=score,
                    source="semantic",
                )
            )

        candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                -candidate.document.confidence,
            )
        )
        return candidates[:limit]

    def all_documents(self) -> list[ImageDocument]:
        return list(self._documents_by_key.values())

    def all_rejects(self) -> list[RejectedAsset]:
        return list(self._rejects)


class PostgresImageIndexRepository:
    def __init__(self, database_url: str, vector_dimensions: int = 16) -> None:
        self._database_url = database_url
        self._vector_dimensions = vector_dimensions

    def ensure_schema(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS image_index (
                        unit_id BIGINT NOT NULL,
                        data_id BIGINT NOT NULL,
                        image_id INTEGER NOT NULL,
                        asset_path TEXT NOT NULL,
                        sha256 TEXT NOT NULL UNIQUE,
                        file_size BIGINT NOT NULL,
                        width INTEGER NOT NULL,
                        height INTEGER NOT NULL,
                        keyword_tags TEXT[] NOT NULL,
                        normalized_tags TEXT[] NOT NULL,
                        tag_text_projection TEXT NOT NULL,
                        confidence DOUBLE PRECISION NOT NULL,
                        review_status TEXT NOT NULL,
                        embedding VECTOR({self._vector_dimensions}) NOT NULL,
                        tagged_at TIMESTAMPTZ NOT NULL,
                        PRIMARY KEY (unit_id, data_id, image_id)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS image_rejects (
                        id BIGSERIAL PRIMARY KEY,
                        asset_path TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        detail TEXT,
                        recorded_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS image_index_review_status_idx
                    ON image_index (review_status)
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS image_index_normalized_tags_idx
                    ON image_index
                    USING GIN (normalized_tags)
                    """
                )

    def has_sha256(self, sha256: str) -> bool:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM image_index WHERE sha256 = %s LIMIT 1",
                    (sha256,),
                )
                return cursor.fetchone() is not None

    def upsert_image(self, document: ImageDocument) -> bool:
        if self.has_sha256(document.sha256):
            return False

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO image_index (
                        unit_id,
                        data_id,
                        image_id,
                        asset_path,
                        sha256,
                        file_size,
                        width,
                        height,
                        keyword_tags,
                        normalized_tags,
                        tag_text_projection,
                        confidence,
                        review_status,
                        embedding,
                        tagged_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s
                    )
                    ON CONFLICT (unit_id, data_id, image_id)
                    DO UPDATE SET
                        asset_path = EXCLUDED.asset_path,
                        sha256 = EXCLUDED.sha256,
                        file_size = EXCLUDED.file_size,
                        width = EXCLUDED.width,
                        height = EXCLUDED.height,
                        keyword_tags = EXCLUDED.keyword_tags,
                        normalized_tags = EXCLUDED.normalized_tags,
                        tag_text_projection = EXCLUDED.tag_text_projection,
                        confidence = EXCLUDED.confidence,
                        review_status = EXCLUDED.review_status,
                        embedding = EXCLUDED.embedding,
                        tagged_at = EXCLUDED.tagged_at
                    """,
                    (
                        document.unit_id,
                        document.data_id,
                        document.image_id,
                        document.asset_path,
                        document.sha256,
                        document.file_size,
                        document.width,
                        document.height,
                        document.keyword_tags,
                        document.normalized_tags,
                        document.tag_text_projection,
                        document.confidence,
                        document.review_status,
                        _vector_literal(document.embedding),
                        document.tagged_at,
                    ),
                )
                return True

    def record_reject(self, rejected: RejectedAsset) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO image_rejects (asset_path, reason, detail, recorded_at)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        rejected.asset_path,
                        rejected.reason,
                        rejected.detail,
                        rejected.recorded_at,
                    ),
                )

    def search_by_normalized_tags(
        self,
        normalized_tags: list[str],
        review_status: str | None,
        limit: int,
    ) -> list[SearchCandidate]:
        review_status_filter, review_status_params = _build_review_status_filter(
            review_status
        )
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        unit_id,
                        data_id,
                        image_id,
                        asset_path,
                        sha256,
                        file_size,
                        width,
                        height,
                        keyword_tags,
                        normalized_tags,
                        tag_text_projection,
                        confidence,
                        review_status,
                        embedding::text,
                        tagged_at,
                        ARRAY(
                            SELECT tag
                            FROM unnest(normalized_tags) AS tag
                            WHERE tag = ANY(%s::text[])
                        ) AS matched_tags
                    FROM image_index
                    WHERE normalized_tags && %s::text[]
                    {review_status_filter}
                    ORDER BY confidence DESC, tagged_at DESC
                    LIMIT %s
                    """,
                    (
                        normalized_tags,
                        normalized_tags,
                        *review_status_params,
                        limit,
                    ),
                )
                return [
                    SearchCandidate(
                        document=_row_to_document(row),
                        matched_tags=row["matched_tags"],
                        score=1.0,
                        source="exact",
                    )
                    for row in cursor.fetchall()
                ]

    def search_by_embedding(
        self,
        embedding: list[float],
        review_status: str | None,
        limit: int,
    ) -> list[SearchCandidate]:
        literal = _vector_literal(embedding)
        review_status_filter, review_status_params = _build_review_status_filter(
            review_status
        )
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        unit_id,
                        data_id,
                        image_id,
                        asset_path,
                        sha256,
                        file_size,
                        width,
                        height,
                        keyword_tags,
                        normalized_tags,
                        tag_text_projection,
                        confidence,
                        review_status,
                        embedding::text,
                        tagged_at,
                        GREATEST(0.0, 1 - (embedding <=> %s::vector)) AS score
                    FROM image_index
                    WHERE 1 = 1
                    {review_status_filter}
                    ORDER BY embedding <=> %s::vector, confidence DESC
                    LIMIT %s
                    """,
                    (
                        literal,
                        *review_status_params,
                        literal,
                        limit,
                    ),
                )
                return [
                    SearchCandidate(
                        document=_row_to_document(row),
                        matched_tags=[],
                        score=float(row["score"]),
                        source="semantic",
                    )
                    for row in cursor.fetchall()
                    if float(row["score"]) > 0.0
                ]

    def _connect(self):
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(
            self._database_url,
            autocommit=True,
            row_factory=dict_row,
        )


def _row_to_document(row: dict[str, object]) -> ImageDocument:
    return ImageDocument(
        unit_id=int(row["unit_id"]),
        data_id=int(row["data_id"]),
        image_id=int(row["image_id"]),
        asset_path=str(row["asset_path"]),
        sha256=str(row["sha256"]),
        file_size=int(row["file_size"]),
        width=int(row["width"]),
        height=int(row["height"]),
        keyword_tags=list(row["keyword_tags"]),
        normalized_tags=list(row["normalized_tags"]),
        tag_text_projection=str(row["tag_text_projection"]),
        confidence=float(row["confidence"]),
        review_status=str(row["review_status"]),
        embedding=_parse_vector_literal(str(row["embedding"])),
        tagged_at=row["tagged_at"],
    )


def _parse_vector_literal(vector_literal: str) -> list[float]:
    stripped = vector_literal.strip().strip("[]")
    if not stripped:
        return []
    return [float(value.strip()) for value in stripped.split(",")]


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def _build_review_status_filter(review_status: str | None) -> tuple[str, tuple[object, ...]]:
    if review_status is None:
        return "", ()
    return " AND review_status = %s", (review_status,)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0

    dot_product = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)
