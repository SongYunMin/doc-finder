from pathlib import Path

from PIL import Image

from doc_finder.services.white_background_asset_converter import (
    apply_white_background_to_png_assets,
)


def _write_transparent_png(path: Path) -> None:
    image = Image.new("RGBA", (2, 1), (0, 0, 0, 0))
    image.putpixel((1, 0), (255, 0, 0, 255))
    image.save(path)


def test_apply_white_background_dry_run_does_not_write_files(tmp_path: Path) -> None:
    png_path = tmp_path / "10565_20077_1.png"
    _write_transparent_png(png_path)
    original_bytes = png_path.read_bytes()

    summary = apply_white_background_to_png_assets(tmp_path)

    assert summary.scanned_count == 1
    assert summary.planned_count == 1
    assert summary.converted_count == 0
    assert summary.skipped_count == 0
    assert png_path.read_bytes() == original_bytes


def test_apply_white_background_flattens_transparent_png_in_place(
    tmp_path: Path,
) -> None:
    png_path = tmp_path / "10565_20077_1.png"
    _write_transparent_png(png_path)

    summary = apply_white_background_to_png_assets(tmp_path, apply=True)

    assert summary.scanned_count == 1
    assert summary.planned_count == 1
    assert summary.converted_count == 1

    image = Image.open(png_path)
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (255, 255, 255)
    assert image.getpixel((1, 0)) == (255, 0, 0)


def test_apply_white_background_skips_opaque_png(tmp_path: Path) -> None:
    png_path = tmp_path / "10565_20077_1.png"
    Image.new("RGB", (1, 1), (0, 0, 0)).save(png_path)
    original_bytes = png_path.read_bytes()

    summary = apply_white_background_to_png_assets(tmp_path, apply=True)

    assert summary.scanned_count == 1
    assert summary.planned_count == 0
    assert summary.converted_count == 0
    assert summary.skipped_count == 1
    assert png_path.read_bytes() == original_bytes
