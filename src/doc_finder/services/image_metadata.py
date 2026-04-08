from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import struct
import xml.etree.ElementTree as ET


class ImageValidationError(ValueError):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


@dataclass(slots=True)
class ImageFileMetadata:
    asset_path: str
    sha256: str
    file_size: int
    width: int
    height: int


def read_image_metadata(path: Path, max_file_size_bytes: int = 500_000) -> ImageFileMetadata:
    file_size = path.stat().st_size
    if file_size > max_file_size_bytes:
        raise ImageValidationError("file_too_large", "Image exceeds maximum file size.")

    extension = path.suffix.lower()
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()

    if extension == ".png":
        width, height = _read_png_dimensions(raw)
    elif extension == ".svg":
        width, height = _read_svg_dimensions(raw)
    else:
        raise ImageValidationError(
            "unsupported_extension",
            f"Unsupported extension: {extension}",
        )

    return ImageFileMetadata(
        asset_path=str(path),
        sha256=digest,
        file_size=file_size,
        width=width,
        height=height,
    )


def _read_png_dimensions(raw: bytes) -> tuple[int, int]:
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ImageValidationError("invalid_image", "PNG header is invalid.")

    return struct.unpack(">II", raw[16:24])


def _read_svg_dimensions(raw: bytes) -> tuple[int, int]:
    try:
        root = ET.fromstring(raw.decode("utf-8"))
    except (UnicodeDecodeError, ET.ParseError) as exc:
        raise ImageValidationError("invalid_image", "SVG cannot be parsed.") from exc

    width = _parse_dimension(root.attrib.get("width"))
    height = _parse_dimension(root.attrib.get("height"))

    if width and height:
        return width, height

    view_box = root.attrib.get("viewBox")
    if view_box:
        parts = view_box.replace(",", " ").split()
        if len(parts) == 4:
            return int(float(parts[2])), int(float(parts[3]))

    raise ImageValidationError(
        "invalid_image",
        "SVG must include width/height or a valid viewBox.",
    )


def _parse_dimension(raw_value: str | None) -> int | None:
    if raw_value is None:
        return None

    cleaned = raw_value.strip().removesuffix("px")
    try:
        return int(float(cleaned))
    except ValueError:
        return None
