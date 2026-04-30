from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from doc_finder.services.image_payload import PNG_SIGNATURE, read_white_background_png_bytes


@dataclass(slots=True)
class SvgConversionEvent:
    source_path: str
    target_path: str
    status: str
    detail: str | None = None


@dataclass(slots=True)
class SvgConversionSummary:
    scanned_count: int = 0
    planned_count: int = 0
    converted_count: int = 0
    skipped_count: int = 0
    deleted_count: int = 0
    failed_count: int = 0
    events: list[SvgConversionEvent] = field(default_factory=list)


def convert_svg_assets(
    directory: Path | str,
    *,
    apply: bool = False,
    delete_source: bool = False,
    overwrite: bool = False,
) -> SvgConversionSummary:
    root = Path(directory)
    if delete_source and not apply:
        raise ValueError("delete_source requires apply=True.")
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    summary = SvgConversionSummary()
    for source_path in _iter_svg_files(root):
        target_path = source_path.with_suffix(".png")
        summary.scanned_count += 1

        if target_path.exists() and not overwrite:
            summary.skipped_count += 1
            _record_event(
                summary,
                source_path,
                target_path,
                "skipped",
                "target_png_exists",
            )
            continue

        summary.planned_count += 1
        if not apply:
            _record_event(summary, source_path, target_path, "planned")
            continue

        try:
            png_bytes = _convert_svg_to_png_bytes(source_path)
            target_path.write_bytes(png_bytes)
            summary.converted_count += 1
            _record_event(summary, source_path, target_path, "converted")

            if delete_source:
                # 실제 교체 모드에서만 원본 SVG를 지운다. 기본 동작은 원본 보존이다.
                source_path.unlink()
                summary.deleted_count += 1
                _record_event(summary, source_path, target_path, "deleted_source")
        except Exception as exc:  # noqa: BLE001
            summary.failed_count += 1
            _record_event(summary, source_path, target_path, "failed", str(exc))

    return summary


def _iter_svg_files(root: Path):
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".svg"
    )


def _convert_svg_to_png_bytes(source_path: Path) -> bytes:
    # Gemma 입력과 동일하게 SVG를 흰 배경 PNG로 렌더링한다.
    png_bytes = read_white_background_png_bytes(source_path)
    if not png_bytes.startswith(PNG_SIGNATURE):
        raise ValueError("SVG could not be rendered as PNG.")
    return png_bytes


def _record_event(
    summary: SvgConversionSummary,
    source_path: Path,
    target_path: Path,
    status: str,
    detail: str | None = None,
) -> None:
    summary.events.append(
        SvgConversionEvent(
            source_path=str(source_path),
            target_path=str(target_path),
            status=status,
            detail=detail,
        )
    )
