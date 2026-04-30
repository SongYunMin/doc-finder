from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image


@dataclass(slots=True)
class WhiteBackgroundEvent:
    asset_path: str
    status: str
    detail: str | None = None


@dataclass(slots=True)
class WhiteBackgroundSummary:
    scanned_count: int = 0
    planned_count: int = 0
    converted_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    events: list[WhiteBackgroundEvent] = field(default_factory=list)


def apply_white_background_to_png_assets(
    directory: Path | str,
    *,
    apply: bool = False,
) -> WhiteBackgroundSummary:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    summary = WhiteBackgroundSummary()
    for asset_path in _iter_png_files(root):
        summary.scanned_count += 1
        try:
            if not _png_has_alpha(asset_path):
                summary.skipped_count += 1
                _record_event(summary, asset_path, "skipped", "already_opaque")
                continue

            summary.planned_count += 1
            if not apply:
                _record_event(summary, asset_path, "planned")
                continue

            _flatten_png_to_white_background(asset_path)
            summary.converted_count += 1
            _record_event(summary, asset_path, "converted")
        except Exception as exc:  # noqa: BLE001
            summary.failed_count += 1
            _record_event(summary, asset_path, "failed", str(exc))

    return summary


def _iter_png_files(root: Path):
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".png"
    )


def _png_has_alpha(asset_path: Path) -> bool:
    with Image.open(asset_path) as image:
        return "A" in image.getbands() or "transparency" in image.info


def _flatten_png_to_white_background(asset_path: Path) -> None:
    with Image.open(asset_path) as image:
        # 투명 영역은 모델이 검정 배경으로 오판하기 쉬우므로 파일 자체를 흰 배경 RGB PNG로 고정한다.
        source = image.convert("RGBA")
        background = Image.new("RGBA", source.size, (255, 255, 255, 255))
        background.alpha_composite(source)
        background.convert("RGB").save(asset_path, format="PNG")


def _record_event(
    summary: WhiteBackgroundSummary,
    asset_path: Path,
    status: str,
    detail: str | None = None,
) -> None:
    summary.events.append(
        WhiteBackgroundEvent(
            asset_path=str(asset_path),
            status=status,
            detail=detail,
        )
    )
