from __future__ import annotations

from io import BytesIO
from pathlib import Path


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def read_white_background_png_bytes(asset_path: Path) -> bytes:
    raw = asset_path.read_bytes()
    if asset_path.suffix.lower() == ".svg":
        raw = _render_svg_to_png_bytes(raw)
    return flatten_transparent_image_bytes(raw)


def _render_svg_to_png_bytes(raw: bytes) -> bytes:
    import cairosvg

    rendered = cairosvg.svg2png(bytestring=raw)
    if not isinstance(rendered, bytes):
        raise ValueError("SVG could not be rendered as PNG.")
    return rendered


def flatten_transparent_image_bytes(raw: bytes) -> bytes:
    from PIL import Image

    with Image.open(BytesIO(raw)) as image:
        if not _image_has_alpha(image):
            return raw

        # 투명 배경은 vision 모델에서 검정 실루엣으로 오판되기 쉬워 흰 배경으로 고정한다.
        source = image.convert("RGBA")
        background = Image.new("RGBA", source.size, (255, 255, 255, 255))
        background.alpha_composite(source)

        output = BytesIO()
        background.convert("RGB").save(output, format="PNG")
        return output.getvalue()


def _image_has_alpha(image) -> bool:
    return "A" in image.getbands() or "transparency" in image.info
