"""Florence-2 SearchTag 튜닝용 보조 모듈."""

__all__ = [
    "FlorenceBatchCollator",
    "SearchTagExample",
    "SearchTagJsonlDataset",
    "SearchTagRecord",
    "GeoTagExample",
    "GeoTagJsonlDataset",
    "GeoTagRecord",
    "TrainingConfig",
    "compute_tag_metrics",
    "load_searchtag_records",
    "load_geotag_records",
    "train",
]


def __getattr__(name: str):
    # `python -m models.florence_2_tuning.training` 실행 시 선행 import 경고를 막기 위해
    # 패키지 루트에서는 필요한 시점에만 서브모듈을 불러온다.
    if name in {
        "SearchTagExample",
        "SearchTagJsonlDataset",
        "SearchTagRecord",
        "load_searchtag_records",
        "GeoTagExample",
        "GeoTagJsonlDataset",
        "GeoTagRecord",
        "load_geotag_records",
    }:
        from models.florence_2_tuning.dataset import (
            GeoTagExample,
            GeoTagJsonlDataset,
            GeoTagRecord,
            SearchTagExample,
            SearchTagJsonlDataset,
            SearchTagRecord,
            load_geotag_records,
            load_searchtag_records,
        )

        exports = {
            "SearchTagExample": SearchTagExample,
            "SearchTagJsonlDataset": SearchTagJsonlDataset,
            "SearchTagRecord": SearchTagRecord,
            "load_searchtag_records": load_searchtag_records,
            "GeoTagExample": GeoTagExample,
            "GeoTagJsonlDataset": GeoTagJsonlDataset,
            "GeoTagRecord": GeoTagRecord,
            "load_geotag_records": load_geotag_records,
        }
        return exports[name]

    if name == "compute_tag_metrics":
        from models.florence_2_tuning.metrics import compute_tag_metrics

        return compute_tag_metrics

    if name in {"FlorenceBatchCollator", "TrainingConfig", "train"}:
        from models.florence_2_tuning.training import FlorenceBatchCollator, TrainingConfig, train

        exports = {
            "FlorenceBatchCollator": FlorenceBatchCollator,
            "TrainingConfig": TrainingConfig,
            "train": train,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
