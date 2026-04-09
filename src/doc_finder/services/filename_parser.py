from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


_FILENAME_PATTERN = re.compile(
    r"^(?P<unit_id>\d+)_(?P<data_id>\d+)(?:_(?P<image_id>\d+))?\.(?P<extension>png|svg)$",
    re.IGNORECASE,
)


class FilenameParseError(ValueError):
    pass


@dataclass(slots=True)
class ParsedAssetFilename:
    unit_id: int
    data_id: int
    image_id: int
    extension: str
    original_name: str


def parse_asset_filename(path_or_name: str | Path) -> ParsedAssetFilename:
    original_name = Path(path_or_name).name
    match = _FILENAME_PATTERN.match(original_name)
    if match is None:
        raise FilenameParseError(
            "Asset filename must match <unitId>_<dataId>[ _<imageOrder>].(png|svg)."
        )

    return ParsedAssetFilename(
        unit_id=int(match.group("unit_id")),
        data_id=int(match.group("data_id")),
        image_id=int(match.group("image_id") or 1),
        extension=match.group("extension").lower(),
        original_name=original_name,
    )
