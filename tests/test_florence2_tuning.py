from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from models.florence_2_tuning.dataset import GeoTagExample, load_geotag_records
from models.florence_2_tuning.metrics import compute_tag_metrics
from models.florence_2_tuning.training import FlorenceBatchCollator


class _FakeTokenBatch:
    def __init__(self, input_ids) -> None:
        self.input_ids = input_ids


class _FakeTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        *,
        text: list[str],
        return_tensors: str,
        padding: bool,
        return_token_type_ids: bool = False,
    ) -> _FakeTokenBatch:
        assert return_tensors == "pt"
        assert padding is True
        assert return_token_type_ids is False

        import torch

        tokens = {
            "triangle; point_a": [11, 12, 13],
            "circle": [21],
        }
        max_length = max(len(tokens[item]) for item in text)
        rows = []
        for item in text:
            row = tokens[item][:]
            row.extend([self.pad_token_id] * (max_length - len(row)))
            rows.append(row)
        return _FakeTokenBatch(torch.tensor(rows))


class _FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

    def __call__(
        self,
        *,
        text: list[str],
        images: list[Image.Image],
        return_tensors: str,
        padding: bool,
    ) -> dict[str, object]:
        assert text == ["<GeoTag>", "<GeoTag> point 중심"]
        assert [image.size for image in images] == [(12, 8), (10, 10)]
        assert return_tensors == "pt"
        assert padding is True

        import torch

        return {
            "input_ids": torch.tensor([[101, 102, 0], [103, 104, 105]]),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
            "pixel_values": torch.ones((2, 3, 4, 4)),
        }


def test_load_geotag_records_filters_split_and_resolves_image_paths(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (12, 8), color="white").save(images_dir / "sample-a.png")
    Image.new("RGB", (10, 10), color="white").save(images_dir / "sample-b.png")

    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                (
                    '{"image":"images/sample-a.png","prompt":"<GeoTag>",'
                    '"target_text":"triangle; point_a","split":"train"}'
                ),
                (
                    '{"image":"images/sample-b.png","prompt":"<GeoTag> point 중심",'
                    '"target_text":"circle","split":"validation"}'
                ),
            ]
        ),
        encoding="utf-8",
    )

    records = load_geotag_records(
        dataset_path=dataset_path,
        image_root=tmp_path,
        split="train",
    )

    assert len(records) == 1
    assert records[0].image_path == images_dir / "sample-a.png"
    assert records[0].prompt == "<GeoTag>"
    assert records[0].target_text == "triangle; point_a"


def test_florence_batch_collator_masks_padding_tokens_in_labels() -> None:
    collator = FlorenceBatchCollator(processor=_FakeProcessor(), ignore_index=-100)
    batch = collator(
        [
            GeoTagExample(
                image=Image.new("RGB", (12, 8), color="white"),
                image_path=Path("images/sample-a.png"),
                prompt="<GeoTag>",
                target_text="triangle; point_a",
                split="train",
            ),
            GeoTagExample(
                image=Image.new("RGB", (10, 10), color="white"),
                image_path=Path("images/sample-b.png"),
                prompt="<GeoTag> point 중심",
                target_text="circle",
                split="train",
            ),
        ]
    )

    assert batch["input_ids"].tolist() == [[101, 102, 0], [103, 104, 105]]
    assert batch["labels"].tolist() == [[11, 12, 13], [21, -100, -100]]


def test_compute_tag_metrics_scores_exact_match_and_macro_counts() -> None:
    metrics = compute_tag_metrics(
        predictions=["triangle; point_a", "circle"],
        references=["triangle; point_a", "circle; radius"],
    )

    assert metrics["exact_match"] == pytest.approx(0.5)
    assert metrics["tag_precision"] == pytest.approx(1.0)
    assert metrics["tag_recall"] == pytest.approx(0.75)
    assert metrics["tag_f1"] == pytest.approx(0.8571428571)
