from pathlib import Path

from doc_finder.repositories.image_index import InMemoryImageIndexRepository
from doc_finder.services.embedding_service import HashingEmbeddingService
from doc_finder.services.ingestion_service import IngestionService
from doc_finder.services.tagging_service import StaticVisionTagger, TaggingResult


def _write_svg(path: Path, label: str) -> None:
    path.write_text(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="80">'
            f'<text>{label}</text>'
            "</svg>"
        ),
        encoding="utf-8",
    )


def test_ingestion_service_indexes_multiple_images_for_same_unit_and_data(
    tmp_path: Path,
) -> None:
    first = tmp_path / "10565_20077_1.svg"
    second = tmp_path / "10565_20077_2.svg"
    _write_svg(first, "apple")
    _write_svg(second, "bus")

    repository = InMemoryImageIndexRepository()
    tagger = StaticVisionTagger(
        {
            first.name: TaggingResult(
                keyword_tags=["사과", "apple"],
                normalized_tags=["사과"],
                confidence=0.96,
            ),
            second.name: TaggingResult(
                keyword_tags=["버스", "bus"],
                normalized_tags=["버스"],
                confidence=0.91,
            ),
        }
    )
    service = IngestionService(
        repository=repository,
        tagger=tagger,
        embedding_service=HashingEmbeddingService(),
    )

    summary = service.ingest_directory(tmp_path)

    assert summary.indexed_count == 2
    assert summary.reject_count == 0
    assert sorted(
        (document.unit_id, document.data_id, document.image_id)
        for document in repository.all_documents()
    ) == [
        (10565, 20077, 1),
        (10565, 20077, 2),
    ]


def test_ingestion_service_records_reject_for_invalid_filename(tmp_path: Path) -> None:
    invalid = tmp_path / "wrong.svg"
    _write_svg(invalid, "ignored")

    repository = InMemoryImageIndexRepository()
    service = IngestionService(
        repository=repository,
        tagger=StaticVisionTagger({}),
        embedding_service=HashingEmbeddingService(),
    )

    summary = service.ingest_directory(tmp_path)

    assert summary.indexed_count == 0
    assert summary.reject_count == 1
    assert repository.all_rejects()[0].reason == "invalid_filename"


def test_ingestion_service_deduplicates_reingested_file_by_sha256(
    tmp_path: Path,
) -> None:
    image = tmp_path / "10565_20077_1.svg"
    _write_svg(image, "apple")

    repository = InMemoryImageIndexRepository()
    tagger = StaticVisionTagger(
        {
            image.name: TaggingResult(
                keyword_tags=["사과", "apple"],
                normalized_tags=["사과"],
                confidence=0.93,
            )
        }
    )
    service = IngestionService(
        repository=repository,
        tagger=tagger,
        embedding_service=HashingEmbeddingService(),
    )

    first = service.ingest_directory(tmp_path)
    second = service.ingest_directory(tmp_path)

    assert first.indexed_count == 1
    assert second.duplicate_count == 1
    assert len(repository.all_documents()) == 1
