import pytest

from doc_finder.services.filename_parser import FilenameParseError, parse_asset_filename


def test_parse_asset_filename_extracts_unit_data_and_image_id() -> None:
    parsed = parse_asset_filename("10565_20077_1.svg")

    assert parsed.unit_id == 10565
    assert parsed.data_id == 20077
    assert parsed.image_id == 1
    assert parsed.extension == "svg"


def test_parse_asset_filename_rejects_invalid_pattern() -> None:
    with pytest.raises(FilenameParseError):
        parse_asset_filename("bad_name.svg")
