import pickle
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from road_accidents.config import PipelineError
from road_accidents.features.temporal import FEATURE_COLUMNS, validate_feature_columns
from road_accidents.provenance import hash_bytes


class SharedPreprocessor:
    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns if columns is not None else list(FEATURE_COLUMNS)
        validate_feature_columns(self.columns)
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise PipelineError("Preprocessing needs an ordered, unique feature list.")
        self.pipeline: ColumnTransformer | None = None
        self.feature_names: list[str] = []

    def fit(self, frame: pd.DataFrame) -> Self:
        if self.pipeline is not None:
            raise PipelineError(
                "A fitted preprocessor cannot be refitted.", 5, "temporal_integrity"
            )
        if "week_start" not in frame or frame.empty:
            raise PipelineError("Training dates are required for preprocessing.")
        dates = pd.to_datetime(frame.week_start)
        if (dates < "2019-01-01").any() or (dates >= "2024-01-01").any():
            raise PipelineError(
                "Preprocessing can only be fitted on 2019-2023.", 5, "temporal_integrity"
            )
        numeric = [column for column in self.columns if column != "road_type"]
        categorical = [column for column in self.columns if column == "road_type"]
        transformers = []
        if numeric:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        [
                            (
                                "impute",
                                SimpleImputer(
                                    strategy="median", add_indicator=True, keep_empty_features=True
                                ),
                            ),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    numeric,
                )
            )
        if categorical:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            (
                                "impute",
                                SimpleImputer(
                                    strategy="constant",
                                    fill_value="unknown",
                                    keep_empty_features=True,
                                ),
                            ),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    categorical,
                )
            )
        self.pipeline = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
        self.pipeline.fit(frame[self.columns])
        self.feature_names = list(self.pipeline.get_feature_names_out())
        return self

    def transform(self, frame: pd.DataFrame) -> NDArray[np.float32]:
        if self.pipeline is None:
            raise PipelineError("Preprocessor must be fitted on training data first.")
        if not set(self.columns) <= set(frame):
            raise PipelineError("Feature schema differs from the fitted preprocessor.")
        transformed = np.asarray(self.pipeline.transform(frame[self.columns]), dtype=np.float32)
        if (
            transformed.shape != (len(frame), len(self.feature_names))
            or not np.isfinite(transformed).all()
        ):
            raise PipelineError(
                "Preprocessing produced invalid model inputs.", 6, "invalid_features"
            )
        return transformed

    @property
    def state_hash(self) -> str:
        return hash_bytes(pickle.dumps(self, protocol=5))

    def save(self, path: Path) -> None:
        if self.pipeline is None:
            raise PipelineError("Cannot serialize an unfitted preprocessor.")
        with path.open("xb") as output:
            pickle.dump(self, output, protocol=5)

    @classmethod
    def load(cls, path: Path) -> Self:
        with path.open("rb") as source:
            processor = pickle.load(source)
        if not isinstance(processor, cls) or processor.pipeline is None:
            raise PipelineError("Invalid preprocessor artifact.")
        return processor
