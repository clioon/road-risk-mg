import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from road_accidents.config import PipelineError


def training_arrays(
    features: ArrayLike, labels: ArrayLike
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    matrix = np.asarray(features, dtype=np.float32)
    target = np.asarray(labels)
    if (
        matrix.ndim != 2
        or len(matrix) != len(target)
        or target.ndim != 1
        or not np.isfinite(matrix).all()
    ):
        raise PipelineError("Training inputs are invalid.", 6, "invalid_training_data")
    if set(np.unique(target)) != {0, 1}:
        raise PipelineError("Training requires both binary classes.", 6, "single_class_training")
    return matrix, target.astype(np.int64)


def fit_model(
    family: str, features: ArrayLike, labels: ArrayLike, parameters: dict[str, Any], seed: int
) -> Any:
    matrix, target = training_arrays(features, labels)
    if family == "random_forest":
        model = RandomForestClassifier(
            random_state=seed, n_jobs=1, class_weight="balanced", **parameters
        )
    elif family == "xgboost":
        model = XGBClassifier(
            random_state=seed,
            n_jobs=1,
            device="cpu",
            tree_method="hist",
            eval_metric="logloss",
            scale_pos_weight=float((target == 0).sum() / (target == 1).sum()),
            **parameters,
        )
    elif family == "mlp":
        from road_accidents.models.mlp import fit_mlp

        return fit_mlp(matrix, target, parameters, seed)
    else:
        raise PipelineError("Unsupported model family.", 2, "usage_error")
    model.fit(matrix, target)
    return model


def score_model(model: Any, features: ArrayLike) -> NDArray[np.float64]:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise PipelineError("Invalid model input matrix.", 6, "invalid_features")
    if getattr(model, "family", None) == "mlp":
        from road_accidents.models.mlp import score_mlp

        scores = score_mlp(model, matrix)
    else:
        scores = np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    if (
        scores.shape != (len(matrix),)
        or not np.isfinite(scores).all()
        or not ((scores >= 0) & (scores <= 1)).all()
    ):
        raise PipelineError("Model did not produce valid probabilities.", 6, "invalid_scores")
    return scores


def save_model(model: Any, path: Path, family: str) -> None:
    if family in {"mlp", "graphsage", "gatv2"}:
        from road_accidents.models.mlp import save_neural

        save_neural(model, path)
        return
    if family == "xgboost":
        with path.open("xb") as output:
            output.write(model.get_booster().save_raw(raw_format="ubj"))
        return
    with path.open("xb") as output:
        pickle.dump({"family": family, "model": model}, output, protocol=5)


def load_model(path: Path, family: str) -> Any:
    if family in {"mlp", "graphsage", "gatv2"}:
        from road_accidents.models.mlp import load_neural

        return load_neural(path, family)
    if family == "xgboost":
        model = XGBClassifier(n_jobs=1, device="cpu")
        model.load_model(path)
        return model
    with path.open("rb") as source:
        payload = pickle.load(source)
    if payload["family"] != family:
        raise PipelineError("Model artifact family differs from its manifest.")
    return payload["model"]
