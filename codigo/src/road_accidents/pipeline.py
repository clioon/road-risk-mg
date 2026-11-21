import argparse
import shutil
import time
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from road_accidents.config import PipelineError, StrictConfig, load_config
from road_accidents.data.coverage import CoverageLedger
from road_accidents.data.osm import ingest_osm
from road_accidents.data.pipeline import build_dataset, validate_dataset
from road_accidents.data.prf import ingest_prf
from road_accidents.data.schemas import DatasetConfig
from road_accidents.data.storage import read_json, safe_path, write_json
from road_accidents.data.synthetic import coverage_document, prf_bytes
from road_accidents.evaluation.runner import evaluate_bundle
from road_accidents.experiments.freeze import freeze_experiment
from road_accidents.experiments.schemas import FAMILIES, ExperimentConfig
from road_accidents.experiments.training import run_experiment
from road_accidents.provenance import MemorySampler as MemorySampler
from road_accidents.provenance import canonical_hash, capture_code_state
from road_accidents.reporting.packaging import export_package, verify_package
from road_accidents.reporting.reproduce import reproduce
from road_accidents.reporting.tables import generate_report, verify_report


class SmokeConfig(StrictConfig):
    profile: Literal["smoke_only"] = "smoke_only"
    seed: int = 101
    epochs: int = Field(default=1, ge=1, le=3)
    n_estimators: int = Field(default=8, ge=1, le=20)
    hidden_channels: int = Field(default=8, ge=1, le=32)
    max_seconds: float = Field(default=300, gt=0, le=300)
    max_peak_memory_mb: float = Field(default=4096, gt=0, le=4096)


def run_smoke(root: Path, config: SmokeConfig) -> dict[str, Any]:
    code_root = Path(__file__).resolve().parents[2]
    code = capture_code_state(code_root)
    identity = canonical_hash({"code_hash": code["code_hash"], "config": config.model_dump()})
    prefix = f"artifacts/smoke/{identity[7:]}"
    work = safe_path(root, prefix)
    receipt = work / "receipt.json"
    if receipt.exists():
        result: dict[str, Any] = read_json(receipt)
        verify_package(root / result["package_manifest"])
        verify_report(work, "artifacts/reports/smoke/manifest.json")
        return result
    started = time.perf_counter()
    memory = MemorySampler()
    memory.start()
    try:
        sources = [
            ingest_prf(
                prf_bytes(year),
                work,
                year=year,
                source_url=f"https://example.org/synthetic/{year}.csv",
                profile="smoke_only",
            )
            for year in range(2019, 2027)
        ]
        fixture_directory = code_root / "tests/fixtures"
        osm = ingest_osm(
            (fixture_directory / "network.osm").read_bytes(),
            work,
            effective_at="2018-12-31T23:59:59Z",
            profile="smoke_only",
        )
        boundary = work / "boundary.geojson"
        if not boundary.exists():
            shutil.copyfile(fixture_directory / "mg_boundary.geojson", boundary)
        coverage = CoverageLedger.from_documents(
            [
                coverage_document(year, source.snapshot_id)
                for year, source in zip(range(2019, 2027), sources, strict=True)
            ],
            profile="smoke_only",
        )
        dataset = build_dataset(
            work,
            DatasetConfig(profile="smoke_only", boundary_path="boundary.geojson"),
            sources,
            osm,
            coverage,
        )
        validate_dataset(work, dataset.dataset_version_id)
        models = {}
        for family in FAMILIES:
            parameters = (
                {"n_estimators": config.n_estimators, "max_depth": 2}
                if family in {"random_forest", "xgboost"}
                else {
                    "epochs": config.epochs,
                    "hidden_channels": config.hidden_channels,
                    "batch_size": 32,
                    "dropout": 0.0,
                }
            )
            model_path = work / f"{family}.yaml"
            if not model_path.exists():
                model_path.write_text(
                    yaml.safe_dump({"model_family": family, "parameters": parameters}),
                    encoding="utf-8",
                )
            models[family] = model_path.name
        experiment = ExperimentConfig(
            profile="smoke_only", models=models, seeds=[config.seed], feature_sets=["full"]
        )
        group = run_experiment(work, dataset.dataset_version_id, experiment)
        bundle = freeze_experiment(work, group["parent_run_id"])
        evaluations = [
            evaluate_bundle(work, bundle.bundle_id, dataset.dataset_version_id, split)
            for split in ("test_2025", "test_2026")
        ]
        report = generate_report(
            work,
            evaluations[0].evaluation_id,
            evaluations[1].evaluation_id,
            "artifacts/reports/smoke",
        )
        package = export_package(work, report["manifest"], "artifacts/packages/smoke")
        replay = reproduce(package, output=work / "artifacts/reproductions/smoke")
    finally:
        memory.stop()
    elapsed = time.perf_counter() - started
    peak_mb = memory.peak_bytes / 1024**2
    result = {
        "profile": "smoke_only",
        "dataset_version_id": dataset.dataset_version_id,
        "bundle_id": bundle.bundle_id,
        "evaluation_ids": [evaluation.evaluation_id for evaluation in evaluations],
        "manifest": f"{prefix}/{report['manifest']}",
        "package_manifest": package.relative_to(root.resolve()).as_posix(),
        "reproduction": replay,
        "elapsed_seconds": elapsed,
        "peak_memory_mb": peak_mb,
        "measurement": "RSS of parent and children sampled every 50 milliseconds",
    }
    if elapsed > config.max_seconds or peak_mb > config.max_peak_memory_mb:
        raise PipelineError(
            "Smoke exceeded its declared runtime or memory budget.", 6, "smoke_budget_exceeded"
        )
    write_json(receipt, result)
    return result


def run_dvc_stage(stage: str, root: Path) -> dict[str, Any]:
    if stage == "experiment":
        selected = read_json(root / "data/manifests/prepared.json")
        configuration = load_config(root / "configs/experiments/full.yaml", ExperimentConfig)
        group = run_experiment(root, selected["dataset_version_id"], configuration)
        bundle = freeze_experiment(root, group["parent_run_id"])
        result = {
            "dataset_version_id": selected["dataset_version_id"],
            "parent_run_id": group["parent_run_id"],
            "bundle_id": bundle.bundle_id,
        }
    elif stage == "report":
        selected = read_json(root / "artifacts/dvc/experiment.json")
        evaluations = [
            evaluate_bundle(root, selected["bundle_id"], selected["dataset_version_id"], split)
            for split in ("test_2025", "test_2026")
        ]
        suffix = selected["bundle_id"][7:]
        report = generate_report(
            root,
            evaluations[0].evaluation_id,
            evaluations[1].evaluation_id,
            f"artifacts/reports/{suffix}",
        )
        package = export_package(root, report["manifest"], f"artifacts/packages/{suffix}")
        result = {
            "manifest": report["manifest"],
            "package_manifest": package.relative_to(root).as_posix(),
            "evaluation_ids": [evaluation.evaluation_id for evaluation in evaluations],
        }
    else:
        raise PipelineError("Unknown DVC stage.", 2, "usage_error")
    receipt = root / f"artifacts/dvc/{stage}.json"
    if receipt.exists() and read_json(receipt) != result:
        raise PipelineError("DVC selector already identifies different artifacts.")
    if not receipt.exists():
        write_json(receipt, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["experiment", "report"])
    arguments = parser.parse_args()
    run_dvc_stage(arguments.stage, Path.cwd())
