import shutil

import pytest

from road_accidents.config import PipelineError


def test_package_inventory_is_complete_and_portable(synthetic_evaluations):
    from road_accidents.reporting.packaging import export_package, verify_package
    from road_accidents.reporting.tables import generate_report

    inputs = synthetic_evaluations
    report = generate_report(
        inputs["root"],
        inputs["main_evaluation"].evaluation_id,
        inputs["temporal_evaluation"].evaluation_id,
        "artifacts/reports/test",
    )
    manifest = export_package(inputs["root"], report["manifest"], "artifacts/packages/test")
    package = verify_package(manifest)
    assert package.status == "sealed" and package.report_manifest_hash
    assert not list(manifest.parent.rglob("*.db"))
    assert all(
        not __import__("pathlib").Path(entry.path).is_absolute() for entry in package.inventory
    )


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_package_failure_precedes_reconstruction(synthetic_evaluations, damage):
    from road_accidents.reporting.packaging import export_package, verify_package
    from road_accidents.reporting.reproduce import reproduce
    from road_accidents.reporting.tables import generate_report

    inputs = synthetic_evaluations
    report = generate_report(
        inputs["root"],
        inputs["main_evaluation"].evaluation_id,
        inputs["temporal_evaluation"].evaluation_id,
        "artifacts/reports/test",
    )
    manifest = export_package(inputs["root"], report["manifest"], "artifacts/packages/test")
    package = verify_package(manifest)
    component = next(entry for entry in package.inventory if entry.component_type == "model")
    path = manifest.parent / component.path
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"corrupt fixture model")
    output = inputs["root"] / "must-not-exist"
    with pytest.raises(PipelineError) as error:
        reproduce(manifest, mode="full", output=output)
    assert error.value.exit_code == 7 and not output.exists()


def test_full_reproduction_uses_only_portable_package(synthetic_evaluations, tmp_path):
    from road_accidents.reporting.packaging import export_package
    from road_accidents.reporting.reproduce import reproduce
    from road_accidents.reporting.tables import generate_report

    inputs = synthetic_evaluations
    report = generate_report(
        inputs["root"],
        inputs["main_evaluation"].evaluation_id,
        inputs["temporal_evaluation"].evaluation_id,
        "artifacts/reports/test",
    )
    manifest = export_package(inputs["root"], report["manifest"], "artifacts/packages/test")
    copied = tmp_path / "portable"
    shutil.copytree(manifest.parent, copied)
    unavailable = tmp_path / "original-unavailable"
    unavailable.mkdir()
    (inputs["root"] / "data").rename(unavailable / "data")
    for directory in ("runs", "bundles", "evaluations", "reports", "code", "experiments"):
        (inputs["root"] / "artifacts" / directory).rename(unavailable / directory)
    result = reproduce(
        copied / "package-manifest.json", mode="full", output=tmp_path / "reproduction"
    )
    assert result["status"] == "succeeded"
    assert result["max_difference_pp"] <= 1.0
    assert result["reconstructed_dataset"] and result["inference_replayed"]
    assert result["training_calls"] == 0
    assert result["undefined_reasons_preserved"]


def test_report_only_manifest_cannot_claim_full_reproduction(synthetic_evaluations):
    from road_accidents.reporting.reproduce import reproduce
    from road_accidents.reporting.tables import generate_report

    inputs = synthetic_evaluations
    report = generate_report(
        inputs["root"],
        inputs["main_evaluation"].evaluation_id,
        inputs["temporal_evaluation"].evaluation_id,
        "artifacts/reports/test",
    )
    with pytest.raises(PipelineError) as error:
        reproduce(inputs["root"] / report["manifest"], mode="full")
    assert error.value.exit_code == 2
    result = reproduce(
        inputs["root"] / report["manifest"],
        mode="reports",
        output=inputs["root"] / "reports-reproduction",
    )
    assert result["status"] == "succeeded" and not result["reconstructed_dataset"]
