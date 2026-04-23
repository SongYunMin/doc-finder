from doc_finder.services.query_normalizer import QueryNormalizer


def test_query_normalizer_maps_korean_aliases_to_english_canonical_tags() -> None:
    normalizer = QueryNormalizer()

    assert normalizer.normalize("사과").normalized_query == "apple"
    assert normalizer.normalize("apple").normalized_query == "apple"
    assert normalizer.normalize("사각형").normalized_query == "rectangle"
    assert normalizer.normalize("네모").normalized_query == "rectangle"
    assert normalizer.normalize("직각").normalized_query == "right_angle"
    assert normalizer.normalize("90도").normalized_query == "right_angle"


def test_query_normalizer_formats_display_tags_with_korean_aliases() -> None:
    normalizer = QueryNormalizer()

    assert normalizer.display_tags(["apple", "right_angle", "bus"]) == [
        "apple (사과)",
        "right_angle (직각, 90도)",
        "bus (버스)",
    ]


def test_query_normalizer_preserves_text_namespace_tags_and_query_variants() -> None:
    normalizer = QueryNormalizer()

    normalized = normalizer.normalize("서윤")

    assert "text:서윤" in normalized.variants
    assert normalizer.normalize_tag_candidates("text:서윤") == ["text:서윤"]
    assert normalizer.display_tag("text:서윤") == "text:서윤"
