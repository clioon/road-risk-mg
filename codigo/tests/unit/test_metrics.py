import numpy as np
import pytest
from sklearn.metrics import average_precision_score, precision_recall_curve


def test_metrics_distinguish_undefined_precision_from_zero_recall():
    from road_accidents.evaluation.metrics import classification_metrics

    metrics = classification_metrics([0, 1], [0.1, 0.2], 0.9)
    assert metrics["precision"]["value"] is None
    assert metrics["precision"]["undefined_reason"] == "no_predicted_positives"
    assert metrics["recall"]["value"] == 0.0
    assert metrics["f1"]["value"] == 0.0
    assert metrics["auprc"]["value"] == average_precision_score([0, 1], [0.1, 0.2])


@pytest.mark.parametrize(
    "labels,scores,expected",
    [
        ([0, 0], [0.1, 0.2], {"auroc", "auprc", "recall", "precision", "f1"}),
        ([1, 1], [0.8, 0.9], {"auroc"}),
    ],
)
def test_metric_nulls_are_specific_to_the_metric(labels, scores, expected):
    from road_accidents.evaluation.metrics import classification_metrics

    metrics = classification_metrics(labels, scores, 0.5)
    assert {name for name, result in metrics.items() if result["value"] is None} == expected


def test_threshold_uses_real_candidates_and_highest_tie():
    from road_accidents.evaluation.metrics import select_threshold

    labels = [0, 1, 0, 1, 1, 0]
    scores = [0.1, 0.2, 0.2, 0.6, 0.6, 0.9]
    precision, recall, thresholds = precision_recall_curve(labels, scores, drop_intermediate=False)
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(thresholds),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    assert select_threshold(labels, scores) == max(thresholds[f1 == f1.max()])
    assert select_threshold([0, 1], [0.5, 0.5]) == 0.5


def test_seed_aggregate_keeps_undefined_seed_and_sample_deviation():
    from road_accidents.evaluation.metrics import aggregate_seeds

    result = aggregate_seeds({101: 0.1, 202: 0.2, 303: 0.3, 404: 0.4, 505: 0.5})
    assert result["mean"] == pytest.approx(0.3)
    assert result["std"] == pytest.approx(np.std([0.1, 0.2, 0.3, 0.4, 0.5], ddof=1))
    missing = aggregate_seeds({101: 0.1, 202: None, 303: 0.3, 404: 0.4, 505: 0.5})
    assert missing["mean"] is None and missing["valid_seed_count"] == 4
