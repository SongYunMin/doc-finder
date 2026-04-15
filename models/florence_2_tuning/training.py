from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from models.florence_2_tuning.dataset import GeoTagExample, GeoTagJsonlDataset
from models.florence_2_tuning.metrics import compute_tag_metrics


@dataclass(frozen=True)
class TrainingConfig:
    dataset_path: Path
    output_dir: Path
    image_root: Path | None
    model_id: str
    train_split: str
    validation_split: str | None
    batch_size: int
    validation_batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    gradient_accumulation_steps: int
    max_grad_norm: float
    num_workers: int
    max_new_tokens: int
    num_beams: int
    device: str
    torch_dtype: str
    freeze_vision_encoder: bool
    train_limit: int | None = None
    validation_limit: int | None = None
    log_every_steps: int = 10


class FlorenceBatchCollator:
    def __init__(self, *, processor: object, ignore_index: int = -100) -> None:
        self._processor = processor
        self.ignore_index = ignore_index

    def __call__(self, batch: list[GeoTagExample]) -> dict[str, object]:
        prompts = [example.prompt for example in batch]
        targets = [example.target_text for example in batch]
        images = [example.image for example in batch]

        inputs = self._processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        labels = self._processor.tokenizer(
            text=targets,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).input_ids

        pad_token_id = getattr(self._processor.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError("processor.tokenizer.pad_token_id가 필요합니다.")

        # loss 계산에서는 padding 위치를 완전히 무시해야 과대 패널티를 막을 수 있다.
        labels = labels.masked_fill(labels == pad_token_id, self.ignore_index)
        inputs["labels"] = labels
        return inputs


def train(config: TrainingConfig) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoProcessor, get_scheduler

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        config.output_dir / "training_config.json",
        _serialize_config(config),
    )

    train_dataset = GeoTagJsonlDataset(
        dataset_path=config.dataset_path,
        image_root=config.image_root,
        split=config.train_split,
        limit=config.train_limit,
    )
    if len(train_dataset) == 0:
        raise ValueError("학습 split에 샘플이 없습니다. dataset과 split 값을 확인하세요.")

    validation_dataset = None
    if config.validation_split is not None:
        validation_dataset = GeoTagJsonlDataset(
            dataset_path=config.dataset_path,
            image_root=config.image_root,
            split=config.validation_split,
            limit=config.validation_limit,
        )
        if len(validation_dataset) == 0:
            validation_dataset = None

    dtype = getattr(torch, config.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(config.device)
    processor = AutoProcessor.from_pretrained(
        config.model_id,
        trust_remote_code=True,
    )

    if config.freeze_vision_encoder:
        _freeze_vision_encoder(model)

    collator = FlorenceBatchCollator(processor=processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
    )
    validation_loader = None
    if validation_dataset is not None:
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config.validation_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collator,
        )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    update_steps_per_epoch = max(
        1,
        len(train_loader) // config.gradient_accumulation_steps,
    )
    if len(train_loader) % config.gradient_accumulation_steps:
        update_steps_per_epoch += 1
    total_training_steps = config.epochs * update_steps_per_epoch
    warmup_steps = int(total_training_steps * config.warmup_ratio)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, config.epochs + 1):
        train_loss = _run_training_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.device,
            torch_dtype=config.torch_dtype,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_grad_norm=config.max_grad_norm,
            log_every_steps=config.log_every_steps,
        )

        epoch_summary: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
        }

        if validation_loader is not None:
            validation_metrics = _evaluate_model(
                model=model,
                validation_loader=validation_loader,
                processor=processor,
                device=config.device,
                torch_dtype=config.torch_dtype,
                max_new_tokens=config.max_new_tokens,
                num_beams=config.num_beams,
                ignore_index=collator.ignore_index,
            )
            epoch_summary.update(validation_metrics)

        history.append(epoch_summary)
        print(json.dumps(epoch_summary, ensure_ascii=False))

        checkpoint_dir = config.output_dir / f"checkpoint-epoch-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)

    final_dir = config.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    result = {
        "config": _serialize_config(config),
        "history": history,
    }
    _write_json(config.output_dir / "metrics_history.json", result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Florence-2 GeoTag 튜닝 스크립트",
    )
    parser.add_argument("--dataset", required=True, type=Path, dest="dataset_path")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument(
        "--model-id",
        default="microsoft/Florence-2-base-ft",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--validation-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument(
        "--freeze-vision-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def parse_args(argv: list[str] | None = None) -> TrainingConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    default_device = _default_device()
    device = args.device or default_device
    torch_dtype = args.torch_dtype or _default_torch_dtype(device)
    validation_split = args.validation_split or None

    return TrainingConfig(
        dataset_path=args.dataset_path.resolve(),
        output_dir=args.output_dir.resolve(),
        image_root=args.image_root.resolve() if args.image_root is not None else None,
        model_id=args.model_id,
        train_split=args.train_split,
        validation_split=validation_split,
        batch_size=args.batch_size,
        validation_batch_size=args.validation_batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        num_workers=args.num_workers,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        device=device,
        torch_dtype=torch_dtype,
        freeze_vision_encoder=args.freeze_vision_encoder,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        log_every_steps=args.log_every_steps,
    )


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    train(config)
    return 0


def _run_training_epoch(
    *,
    model: object,
    train_loader: object,
    optimizer: object,
    scheduler: object,
    device: str,
    torch_dtype: str,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    log_every_steps: int,
) -> float:
    import torch

    model.train()
    total_loss = 0.0

    for step, batch in enumerate(train_loader, start=1):
        moved_batch = _move_batch_to_device(
            batch=batch,
            device=device,
            torch_dtype=torch_dtype,
        )
        outputs = model(**moved_batch)
        loss = outputs.loss
        total_loss += float(loss.item())

        (loss / gradient_accumulation_steps).backward()

        should_step = (
            step % gradient_accumulation_steps == 0 or step == len(train_loader)
        )
        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if log_every_steps > 0 and step % log_every_steps == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": round(float(loss.item()), 6),
                    },
                    ensure_ascii=False,
                )
            )

    return total_loss / len(train_loader)


def _evaluate_model(
    *,
    model: object,
    validation_loader: object,
    processor: object,
    device: str,
    torch_dtype: str,
    max_new_tokens: int,
    num_beams: int,
    ignore_index: int,
) -> dict[str, float]:
    import torch

    model.eval()
    total_loss = 0.0
    predictions: list[str] = []
    references: list[str] = []
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)

    if pad_token_id is None:
        raise ValueError("processor.tokenizer.pad_token_id가 필요합니다.")

    with torch.no_grad():
        for batch in validation_loader:
            moved_batch = _move_batch_to_device(
                batch=batch,
                device=device,
                torch_dtype=torch_dtype,
            )
            outputs = model(**moved_batch)
            total_loss += float(outputs.loss.item())

            generated_ids = model.generate(
                input_ids=moved_batch["input_ids"],
                pixel_values=moved_batch["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
            )

            label_ids = moved_batch["labels"].detach().clone()
            label_ids[label_ids == ignore_index] = pad_token_id

            decoded_predictions = processor.batch_decode(
                generated_ids.detach().cpu(),
                skip_special_tokens=True,
            )
            decoded_references = processor.tokenizer.batch_decode(
                label_ids.detach().cpu(),
                skip_special_tokens=True,
            )

            predictions.extend([text.strip() for text in decoded_predictions])
            references.extend([text.strip() for text in decoded_references])

    metrics = compute_tag_metrics(
        predictions=predictions,
        references=references,
    )
    metrics["validation_loss"] = total_loss / len(validation_loader)
    return metrics


def _move_batch_to_device(
    *,
    batch: dict[str, Any],
    device: str,
    torch_dtype: str,
) -> dict[str, Any]:
    import torch

    dtype = getattr(torch, torch_dtype)
    moved_batch: dict[str, Any] = {}
    for key, value in batch.items():
        if not hasattr(value, "to"):
            moved_batch[key] = value
            continue

        if key == "pixel_values":
            moved_batch[key] = value.to(device=device, dtype=dtype)
            continue

        moved_batch[key] = value.to(device=device)
    return moved_batch


def _freeze_vision_encoder(model: object) -> None:
    vision_tower = getattr(model, "vision_tower", None)
    if vision_tower is None:
        print(
            "vision_tower를 찾지 못해 freeze를 건너뜁니다. "
            "원격 코드 구조가 바뀌었는지 확인하세요."
        )
        return

    # 초기 smoke run에서는 시각 encoder를 잠가서 메모리와 과적합 리스크를 줄인다.
    for parameter in vision_tower.parameters():
        parameter.requires_grad = False
        setattr(parameter, "is_trainable", False)


def _serialize_config(config: TrainingConfig) -> dict[str, Any]:
    serialized = asdict(config)
    serialized["dataset_path"] = str(config.dataset_path)
    serialized["output_dir"] = str(config.output_dir)
    serialized["image_root"] = str(config.image_root) if config.image_root else None
    return serialized


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _default_torch_dtype(device: str) -> str:
    if device.startswith("cuda"):
        return "bfloat16"
    return "float32"


if __name__ == "__main__":
    raise SystemExit(main())
