from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class GeoTagRecord:
    image_path: Path
    prompt: str
    target_text: str
    split: str
    notes: str | None = None


@dataclass(frozen=True)
class GeoTagExample:
    image: Image.Image
    image_path: Path
    prompt: str
    target_text: str
    split: str
    notes: str | None = None


def load_geotag_records(
    *,
    dataset_path: Path,
    image_root: Path | None = None,
    split: str | None = None,
) -> list[GeoTagRecord]:
    records: list[GeoTagRecord] = []

    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        for line_number, raw_line in enumerate(dataset_file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{dataset_path} {line_number}번째 줄 JSON 파싱에 실패했습니다."
                ) from exc

            record = _build_record(
                payload=payload,
                dataset_path=dataset_path,
                image_root=image_root,
            )
            if split is not None and record.split != split:
                continue
            records.append(record)

    return records


class GeoTagJsonlDataset(Dataset[GeoTagExample]):
    def __init__(
        self,
        *,
        dataset_path: Path,
        image_root: Path | None = None,
        split: str | None = None,
        limit: int | None = None,
    ) -> None:
        records = load_geotag_records(
            dataset_path=dataset_path,
            image_root=image_root,
            split=split,
        )
        if limit is not None:
            records = records[:limit]
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> GeoTagExample:
        record = self._records[index]

        # 학습 직전 항상 RGB로 맞춰서 processor 입력 규격을 고정한다.
        with Image.open(record.image_path) as source_image:
            image = source_image.convert("RGB")

        return GeoTagExample(
            image=image,
            image_path=record.image_path,
            prompt=record.prompt,
            target_text=record.target_text,
            split=record.split,
            notes=record.notes,
        )


def _build_record(
    *,
    payload: object,
    dataset_path: Path,
    image_root: Path | None,
) -> GeoTagRecord:
    if not isinstance(payload, dict):
        raise ValueError(f"{dataset_path} JSONL 각 줄은 객체여야 합니다.")

    image_value = _require_text(payload, "image", dataset_path)
    prompt_value = _require_text(payload, "prompt", dataset_path)
    target_value = _require_text(payload, "target_text", dataset_path)
    split_value = _require_text(payload, "split", dataset_path)
    notes_value = payload.get("notes")

    resolved_image_path = Path(image_value)
    if not resolved_image_path.is_absolute():
        base_path = image_root if image_root is not None else dataset_path.parent
        resolved_image_path = (base_path / resolved_image_path).resolve()

    if not resolved_image_path.exists():
        raise FileNotFoundError(
            f"학습 이미지가 존재하지 않습니다: {resolved_image_path}"
        )

    return GeoTagRecord(
        image_path=resolved_image_path,
        prompt=prompt_value,
        target_text=target_value,
        split=split_value,
        notes=str(notes_value).strip() if notes_value is not None else None,
    )


def _require_text(payload: dict[str, object], key: str, dataset_path: Path) -> str:
    raw_value = payload.get(key)
    if raw_value is None:
        raise ValueError(f"{dataset_path} JSONL에 `{key}` 필드가 없습니다.")

    value = str(raw_value).strip()
    if not value:
        raise ValueError(f"{dataset_path} JSONL의 `{key}` 값이 비어 있습니다.")
    return value
