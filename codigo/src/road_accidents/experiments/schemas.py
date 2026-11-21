from typing import Any, Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from road_accidents.config import StrictConfig
from road_accidents.data.schemas import Profile

type ModelFamily = Literal["random_forest", "xgboost", "mlp", "graphsage", "gatv2"]
type FeatureSet = Literal["full", "without_road", "without_temporal", "without_topology"]
FAMILIES: tuple[ModelFamily, ...] = ("random_forest", "xgboost", "mlp", "graphsage", "gatv2")
SEEDS = (101, 202, 303, 404, 505)
VARIANTS: tuple[FeatureSet, ...] = ("full", "without_road", "without_temporal", "without_topology")


class ModelConfig(StrictConfig):
    model_family: ModelFamily
    parameters: dict[str, Any] = Field(default_factory=dict)
    candidates: list[dict[str, Any]] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def allowed_parameters(self) -> Self:
        accepted = (
            {"n_estimators", "max_depth", "min_samples_leaf", "max_features"}
            if self.model_family == "random_forest"
            else {
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "reg_lambda",
            }
            if self.model_family == "xgboost"
            else {
                "epochs",
                "hidden_channels",
                "learning_rate",
                "weight_decay",
                "batch_size",
                "dropout",
                "heads",
            }
        )
        if any(not set(candidate) <= accepted for candidate in [self.parameters, *self.candidates]):
            raise ValueError("Model configuration contains undeclared parameters.")
        for candidate in [self.parameters, *self.candidates]:
            for name, value in candidate.items():
                if name in {
                    "epochs",
                    "hidden_channels",
                    "batch_size",
                    "n_estimators",
                    "heads",
                    "min_samples_leaf",
                } and (not isinstance(value, int) or value <= 0):
                    raise ValueError("Model dimensions and budgets must be positive integers.")
        return self


class ExperimentConfig(StrictConfig):
    profile: Profile = "confirmatory"
    dataset_version_id: str | None = None
    models: dict[ModelFamily, str]
    seeds: list[int] = Field(default_factory=lambda: list(SEEDS))
    feature_sets: list[FeatureSet] = Field(
        default=["full", "without_road", "without_temporal", "without_topology"]
    )
    ablation_test_2026: bool = False
    selection_metric: Literal["average_precision"] = "average_precision"
    seasonal_rule: Literal["common_iso_weeks_and_segments"] = "common_iso_weeks_and_segments"
    max_candidates_per_variant: int = Field(default=12, ge=1, le=12)

    @model_validator(mode="after")
    def complete_protocol(self) -> Self:
        if set(self.models) != set(FAMILIES):
            raise ValueError("All five model families are mandatory.")
        if self.profile == "confirmatory" and (
            self.seeds != list(SEEDS) or set(self.feature_sets) != set(VARIANTS)
        ):
            raise ValueError(
                "Confirmatory selection requires five seeds and three graph ablations."
            )
        if self.profile == "smoke_only" and (
            len(self.seeds) != 1 or self.feature_sets != ["full"] or self.ablation_test_2026
        ):
            raise ValueError("Smoke requires one seed, five full models and no ablations.")
        return self


class ExperimentRun(StrictConfig):
    schema_version: Literal["3.0.0"] = "3.0.0"
    run_id: str
    mlflow_run_id: str
    parent_run_id: str | None = None
    profile: Profile
    dataset_version_id: str
    development_digest: str
    model_family: ModelFamily
    feature_set: FeatureSet
    seed: int
    configuration_hash: str
    parameters: dict[str, Any]
    git_revision: str | None
    code_hash: str
    code_inventory: list[dict[str, Any]]
    environment_lock_hash: str
    started_at: AwareDatetime
    finished_at: AwareDatetime
    decision_threshold: float = Field(ge=0, le=1)
    status: Literal["completed", "failed", "invalidated"]
    failure_reason: str | None = None
    directory: str
    model_path: str
    preprocessor_path: str
    feature_names: list[str]
    feature_columns: list[str]
    node_order_hash: str
    validation_predictions: str
    validation_metrics: dict[str, dict[str, Any]]
    benchmark: dict[str, Any] = Field(default_factory=dict)
    artifact_hashes: dict[str, str]


class FrozenExperiment(StrictConfig):
    schema_version: Literal["3.0.0"] = "3.0.0"
    bundle_id: str
    profile: Profile
    parent_run_id: str
    development_dataset_version_id: str
    development_digest: str
    protocol_hash: str
    selected_run_ids: list[str]
    artifact_hashes: dict[str, str]
    baseline_reference: Literal["random_forest", "xgboost", "mlp"]
    seeds: list[int]
    ablation_test_2026: bool
    seasonal_rule: Literal["common_iso_weeks_and_segments"] = "common_iso_weeks_and_segments"
    code_hash: str
    code_inventory: list[dict[str, Any]]
    environment_lock_hash: str
    node_order_hash: str
    frozen_at: AwareDatetime
    status: Literal["frozen", "invalidated"] = "frozen"
    manifest_path: str

    @model_validator(mode="after")
    def matrix_size(self) -> Self:
        expected = 55 if self.profile == "confirmatory" else 5
        if len(self.selected_run_ids) != expected or len(set(self.selected_run_ids)) != expected:
            raise ValueError("Frozen matrix is incomplete or contains duplicate runs.")
        if self.profile == "confirmatory" and self.seeds != list(SEEDS):
            raise ValueError("Confirmatory seeds are fixed.")
        if self.profile == "smoke_only" and (len(self.seeds) != 1 or self.ablation_test_2026):
            raise ValueError("Synthetic bundles cannot claim confirmatory coverage.")
        return self


class EvaluationRun(StrictConfig):
    schema_version: Literal["3.0.0"] = "3.0.0"
    evaluation_id: str
    profile: Profile
    mlflow_run_id: str
    bundle_id: str
    dataset_version_id: str
    split: Literal["test_2025", "test_2026"]
    coverage_manifest_hash: str
    population_hash: str
    cohort_manifests: list[dict[str, Any]]
    model_run_ids: list[str]
    bundle_before_hash: str
    bundle_after_hash: str
    status: Literal["completed", "failed", "invalidated"]
    supersedes_evaluation_id: str | None = None
    manifest_path: str
    predictions: dict[str, str]
    results: dict[str, dict[str, dict[str, Any]]]
    aggregates: dict[str, Any]
    period: dict[str, Any]
    inference_benchmarks: dict[str, dict[str, Any]] = Field(default_factory=dict)
    artifact_hashes: dict[str, str]


class Prediction(StrictConfig):
    prediction_set_id: str
    run_id: str
    evaluation_id: str | None = None
    dataset_version_id: str
    segment_id: str
    week_start: str
    split: Literal["validation", "test_2025", "test_2026"]
    label: Literal[0, 1]
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    predicted_label: Literal[0, 1]


class EvaluationResult(StrictConfig):
    run_id: str
    prediction_set_id: str
    evaluation_id: str | None = None
    cohort_id: Literal["full_period", "matched_season"]
    split: Literal["validation", "test_2025", "test_2026"]
    metric_name: Literal["auroc", "auprc", "f1", "precision", "recall"]
    metric_value: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    decision_threshold: float | None = Field(default=None, ge=0, le=1)
    observation_count: int = Field(gt=0)
    positive_count: int = Field(ge=0)
    undefined_reason: str | None = None


class PackageComponent(StrictConfig):
    component_type: str
    logical_id: str
    path: str
    byte_count: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ReproductionPackage(StrictConfig):
    schema_version: Literal["3.0.0"] = "3.0.0"
    package_id: str
    profile: Profile
    dataset_version_id: str
    bundle_id: str
    evaluation_ids: list[str] = Field(min_length=2, max_length=2)
    report_manifest_hash: str
    report_manifest: str
    inventory: list[PackageComponent]
    created_at: AwareDatetime
    status: Literal["building", "sealed", "invalid"]
