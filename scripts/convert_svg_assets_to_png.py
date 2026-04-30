#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from doc_finder.services.svg_asset_converter import convert_svg_assets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert SVG assets to white-background PNG files."
    )
    parser.add_argument(
        "image_dir",
        nargs="?",
        default="images",
        help="Directory containing SVG assets. Defaults to ./images.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write PNG files. Without this flag the script only reports planned work.",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete each SVG after its PNG is written successfully. Requires --apply.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files with the same basename.",
    )
    args = parser.parse_args(argv)

    summary = convert_svg_assets(
        Path(args.image_dir),
        apply=args.apply,
        delete_source=args.delete_source,
        overwrite=args.overwrite,
    )
    # 배치 작업 결과는 사람이 저장/비교하기 쉽도록 JSON으로 출력한다.
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    if not args.apply:
        print("dry-run: pass --apply to write PNG files.", file=sys.stderr)
    return 1 if summary.failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
