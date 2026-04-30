from pathlib import Path

from PIL import Image

from doc_finder.services.svg_asset_converter import convert_svg_assets


def _write_sample_svg(path: Path) -> None:
    path.write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg" width="2" height="1">
          <rect x="1" y="0" width="1" height="1" fill="red"/>
        </svg>
        """,
        encoding="utf-8",
    )


def test_convert_svg_assets_dry_run_does_not_write_files(tmp_path: Path) -> None:
    svg_path = tmp_path / "10565_20077_1.svg"
    png_path = tmp_path / "10565_20077_1.png"
    _write_sample_svg(svg_path)

    summary = convert_svg_assets(tmp_path)

    assert summary.scanned_count == 1
    assert summary.planned_count == 1
    assert summary.converted_count == 0
    assert summary.deleted_count == 0
    assert svg_path.exists()
    assert not png_path.exists()


def test_convert_svg_assets_converts_to_white_background_png_and_deletes_source(
    tmp_path: Path,
) -> None:
    svg_path = tmp_path / "10565_20077_1.svg"
    png_path = tmp_path / "10565_20077_1.png"
    _write_sample_svg(svg_path)

    summary = convert_svg_assets(tmp_path, apply=True, delete_source=True)

    assert summary.scanned_count == 1
    assert summary.planned_count == 1
    assert summary.converted_count == 1
    assert summary.deleted_count == 1
    assert not svg_path.exists()
    assert png_path.exists()

    image = Image.open(png_path)
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (255, 255, 255)
    assert image.getpixel((1, 0)) == (255, 0, 0)


def test_convert_svg_assets_skips_existing_png_without_overwrite(
    tmp_path: Path,
) -> None:
    svg_path = tmp_path / "10565_20077_1.svg"
    png_path = tmp_path / "10565_20077_1.png"
    _write_sample_svg(svg_path)
    png_path.write_bytes(b"existing")

    summary = convert_svg_assets(tmp_path, apply=True)

    assert summary.scanned_count == 1
    assert summary.converted_count == 0
    assert summary.skipped_count == 1
    assert png_path.read_bytes() == b"existing"
    assert svg_path.exists()
