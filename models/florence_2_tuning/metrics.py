from __future__ import annotations

import re


def compute_tag_metrics(
    *,
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("예측 수와 정답 수가 다르면 지표를 계산할 수 없습니다.")

    if not predictions:
        return {
            "exact_match": 0.0,
            "tag_precision": 0.0,
            "tag_recall": 0.0,
            "tag_f1": 0.0,
        }

    exact_match_count = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for prediction, reference in zip(predictions, references, strict=True):
        predicted_tags = set(_normalize_tag_string(prediction))
        reference_tags = set(_normalize_tag_string(reference))

        if predicted_tags == reference_tags:
            exact_match_count += 1

        true_positive += len(predicted_tags & reference_tags)
        false_positive += len(predicted_tags - reference_tags)
        false_negative += len(reference_tags - predicted_tags)

    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)

    return {
        "exact_match": exact_match_count / len(predictions),
        "tag_precision": precision,
        "tag_recall": recall,
        "tag_f1": f1,
    }


def _normalize_tag_string(value: str) -> list[str]:
    # 세미콜론 중심 포맷을 기본으로 보되, 개행과 콤마가 섞인 데이터도 방어적으로 정리한다.
    tokens = [
        token.strip().lower()
        for token in re.split(r"[;,\n]+", value)
        if token.strip()
    ]
    return list(dict.fromkeys(tokens))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
