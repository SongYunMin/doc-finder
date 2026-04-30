#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from doc_finder.services.white_background_asset_converter import (
    apply_white_background_to_png_assets,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply a white background to transparent PNG assets."
    )
    parser.add_argument(
        "image_dir",
        nargs="?",
        default="images",
        help="Directory containing PNG assets. Defaults to ./images.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Overwrite transparent PNG files. Without this flag the script is dry-run only.",
    )
    args = parser.parse_args(argv)

    summary = apply_white_background_to_png_assets(
        Path(args.image_dir),
        apply=args.apply,
    )
    # 배치 결과를 로그로 남겨 변환/스킵/실패 파일을 추적할 수 있게 한다.
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    if not args.apply:
        print("dry-run: pass --apply to overwrite transparent PNG files.", file=sys.stderr)
    return 1 if summary.failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
