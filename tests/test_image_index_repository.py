from doc_finder.repositories.image_index import _build_review_status_filter


def test_build_review_status_filter_omits_predicate_when_status_is_none() -> None:
    clause, params = _build_review_status_filter(None)

    assert clause == ""
    assert params == ()


def test_build_review_status_filter_adds_typed_predicate_when_status_is_set() -> None:
    clause, params = _build_review_status_filter("approved")

    assert clause == " AND review_status = %s"
    assert params == ("approved",)
