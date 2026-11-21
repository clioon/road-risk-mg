from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from road_accidents.config import PipelineError
from road_accidents.experiments.schemas import SEEDS

METRICS = ("auroc", "auprc", "f1", "precision", "recall")


def validated_scores(
    labels: ArrayLike, scores: ArrayLike
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    actual = np.asarray(labels)
    predicted = np.asarray(scores, dtype=np.float64)
    if (
        actual.ndim != 1
        or not actual.size
        or actual.shape != predicted.shape
        or not np.isin(actual, [0, 1]).all()
    ):
        raise PipelineError(
            "Metrics require aligned, nonempty binary labels and scores.", 6, "invalid_metrics"
        )
    if not np.isfinite(predicted).all() or not ((predicted >= 0) & (predicted <= 1)).all():
        raise PipelineError("Prediction scores must be finite probabilities.", 6, "invalid_scores")
    return actual.astype(np.int64), predicted


def select_threshold(labels: ArrayLike, scores: ArrayLike) -> float:
    actual, predicted = validated_scores(labels, scores)
    if len(np.unique(actual)) != 2:
        raise PipelineError(
            "Validation requires both classes for selection.", 6, "single_class_validation"
        )
    precision, recall, thresholds = precision_recall_curve(
        actual, predicted, drop_intermediate=False
    )
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(thresholds),
        where=denominator > 0,
    )
    return float(np.max(thresholds[f1 == np.max(f1)]))


def classification_metrics(
    labels: ArrayLike, scores: ArrayLike, threshold: float
) -> dict[str, dict[str, Any]]:
    actual, predicted = validated_scores(labels, scores)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise PipelineError("Invalid frozen decision threshold.", 5, "invalid_threshold")
    positive = predicted >= threshold
    true_positive = int(((actual == 1) & positive).sum())
    false_positive = int(((actual == 0) & positive).sum())
    false_negative = int(((actual == 1) & ~positive).sum())
    count = int(actual.sum())
    denominator = 2 * true_positive + false_positive + false_negative
    values: dict[str, float | None] = {
        "auroc": float(roc_auc_score(actual, predicted)) if 0 < count < len(actual) else None,
        "auprc": float(average_precision_score(actual, predicted)) if count else None,
        "precision": true_positive / int(positive.sum()) if positive.any() else None,
        "recall": true_positive / count if count else None,
        "f1": 2 * true_positive / denominator if denominator else None,
    }
    reasons = {
        "auroc": "missing_actual_class",
        "auprc": "no_actual_positives",
        "precision": "no_predicted_positives",
        "recall": "no_actual_positives",
        "f1": "zero_f1_denominator",
    }
    return {
        name: {
            "value": values[name],
            "undefined_reason": reasons[name] if values[name] is None else None,
            "observation_count": len(actual),
            "positive_count": count,
            "predicted_positive_count": int(positive.sum()),
            "decision_threshold": threshold,
        }
        for name in METRICS
    }


def aggregate_seeds(
    values: dict[int, float | None], expected_seeds: Sequence[int] = SEEDS
) -> dict[str, Any]:
    if set(values) != set(expected_seeds) or len(expected_seeds) != len(set(expected_seeds)):
        raise PipelineError("Metric aggregation is missing declared seeds.", 7, "incomplete_seeds")
    valid = [float(value) for value in values.values() if value is not None]
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in valid):
        raise PipelineError("Invalid seed metric.", 7, "invalid_metric")
    complete = len(valid) == len(expected_seeds)
    return {
        "expected_seeds": list(expected_seeds),
        "valid_seed_count": len(valid),
        "mean": float(np.mean(valid)) if complete else None,
        "std": float(np.std(valid, ddof=1)) if complete and len(valid) > 1 else None,
        "undefined_reason": "undefined_seed_metric"
        if not complete
        else "single_seed_smoke"
        if len(valid) == 1
        else None,
        "values": {str(seed): value for seed, value in sorted(values.items())},
    }
