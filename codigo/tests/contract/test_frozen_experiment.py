import pytest

from road_accidents.config import PipelineError


def test_task_progress_can_change_but_frozen_computation_cannot(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from road_accidents.experiments import freeze
    from road_accidents.provenance import canonical_hash, hash_file

    paths = ["src/model.py", "uv.lock", "specs/001-predicao-acidentes-rodovias/tasks.md"]
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original", encoding="utf-8")
    inventory = [{"path": relative, "sha256": hash_file(tmp_path / relative)} for relative in paths]
    bundle = SimpleNamespace(
        code_inventory=inventory,
        code_hash=canonical_hash(inventory),
        environment_lock_hash=hash_file(tmp_path / "uv.lock"),
    )
    monkeypatch.setattr(
        freeze, "__file__", str(tmp_path / "src/road_accidents/experiments/freeze.py")
    )
    (tmp_path / paths[-1]).write_text("completed task", encoding="utf-8")
    freeze.verify_code(bundle)
    (tmp_path / paths[0]).write_text("changed computation", encoding="utf-8")
    with pytest.raises(PipelineError, match="Source code changed"):
        freeze.verify_code(bundle)


def test_confirmatory_matrix_requires_five_seeds_and_all_ablations():
    from road_accidents.experiments.freeze import validate_selection
    from road_accidents.experiments.schemas import FAMILIES, SEEDS, VARIANTS

    runs = [
        {"model_family": family, "feature_set": variant, "seed": seed}
        for family in FAMILIES
        for variant in (VARIANTS if family in {"graphsage", "gatv2"} else ["full"])
        for seed in SEEDS
    ]
    assert len(runs) == 55
    validate_selection(runs, "confirmatory", list(SEEDS))
    with pytest.raises(PipelineError):
        validate_selection(runs[:-1], "confirmatory", list(SEEDS))
    with pytest.raises(PipelineError):
        validate_selection(runs, "smoke_only", [101])


def test_completed_development_group_freezes_idempotently(synthetic_bundle):
    from road_accidents.experiments.freeze import freeze_experiment, load_bundle

    inputs = synthetic_bundle
    repeated = freeze_experiment(inputs["root"], inputs["experiment"]["parent_run_id"])
    assert repeated.bundle_id == inputs["bundle"].bundle_id
    assert load_bundle(inputs["root"], repeated.bundle_id).status == "frozen"


def test_cli_freeze_and_evaluate_use_exact_bundle(synthetic_bundle, capsys):
    import json

    from road_accidents.cli import main

    inputs = synthetic_bundle
    global_args = ["--work-dir", str(inputs["root"]), "--result-format", "json"]
    capsys.readouterr()
    assert (
        main(
            global_args
            + ["experiment", "freeze", "--parent-run", inputs["experiment"]["parent_run_id"]]
        )
        == 0
    )
    bundle_id = json.loads(capsys.readouterr().out)["outputs"]["bundle_id"]
    assert bundle_id == inputs["bundle"].bundle_id
    for split in ("test_2025", "test_2026"):
        assert (
            main(
                global_args
                + [
                    "evaluate",
                    "--bundle",
                    bundle_id,
                    "--dataset-version",
                    inputs["version"].dataset_version_id,
                    "--split",
                    split,
                ]
            )
            == 0
        )
        result = json.loads(capsys.readouterr().out)
        assert result["outputs"]["bundle_id"] == bundle_id and result["outputs"]["split"] == split


def test_frozen_artifact_tampering_prevents_evaluation(synthetic_bundle):
    from road_accidents.evaluation.runner import evaluate_bundle
    from road_accidents.experiments.freeze import load_bundle

    inputs = synthetic_bundle
    bundle = load_bundle(inputs["root"], inputs["bundle"].bundle_id)
    weight = next(
        path
        for path in bundle.artifact_hashes
        if path.endswith(("model.pt", "model.pkl", "model.ubj"))
    )
    (inputs["root"] / weight).write_bytes(b"changed weights")
    with pytest.raises(PipelineError) as error:
        evaluate_bundle(
            inputs["root"], bundle.bundle_id, inputs["version"].dataset_version_id, "test_2025"
        )
    assert error.value.exit_code == 5


def test_evaluation_never_fits_and_keeps_bundle_in_both_years(synthetic_bundle, monkeypatch):
    from road_accidents.evaluation.runner import evaluate_bundle
    from road_accidents.features.preprocessing import SharedPreprocessor
    from road_accidents.provenance import hash_file

    inputs = synthetic_bundle

    def forbidden(*args, **kwargs):
        pytest.fail("Evaluation attempted to fit a preprocessor.")

    monkeypatch.setattr(SharedPreprocessor, "fit", forbidden)
    before = hash_file(inputs["root"] / inputs["bundle"].manifest_path)
    evaluations = [
        evaluate_bundle(
            inputs["root"], inputs["bundle"].bundle_id, inputs["version"].dataset_version_id, split
        )
        for split in ("test_2025", "test_2026")
    ]
    assert all(
        evaluation.bundle_before_hash == evaluation.bundle_after_hash for evaluation in evaluations
    )
    assert evaluations[0].bundle_id == evaluations[1].bundle_id
    assert before == hash_file(inputs["root"] / inputs["bundle"].manifest_path)
    assert all(
        benchmark["prediction_seconds"] > 0 and benchmark["observations_per_second"] > 0
        for evaluation in evaluations
        for benchmark in evaluation.inference_benchmarks.values()
    )
    assert all(
        set(evaluation.inference_benchmarks) == set(evaluation.results)
        for evaluation in evaluations
    )
    assert (inputs["root"] / "artifacts/protocol/test-access.jsonl").is_file()
