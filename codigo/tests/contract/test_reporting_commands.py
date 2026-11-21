import json

import pandas as pd
import pytest

from road_accidents.config import PipelineError


def test_report_keeps_periods_and_provenance_separate(synthetic_evaluations):
    from road_accidents.reporting.tables import generate_report

    inputs = synthetic_evaluations
    report = generate_report(
        inputs["root"],
        inputs["main_evaluation"].evaluation_id,
        inputs["temporal_evaluation"].evaluation_id,
        "artifacts/reports/test",
    )
    table = pd.read_csv(inputs["root"] / report["tables"]["metrics"])
    assert set(table.split) == {"test_2025", "test_2026"}
    assert set(table.metric) == {"auroc", "auprc", "f1", "precision", "recall"}
    assert table.evaluation_id.notna().all() and table.bundle_id.notna().all()
    assert report["periods"]["test_2026"]["partial"]
    assert report["profile"] == "smoke_only"
    markdown = (inputs["root"] / "artifacts/reports/test/report.md").read_text(encoding="utf-8")
    assert "2026-07-31" in markdown and "coverage_cutoff" in markdown
    assert "partial" in markdown and "2025-12-31" in markdown
    assert "Benchmark Computacional" in markdown and "Comparacao Sazonal" in markdown
    training = pd.read_csv(inputs["root"] / report["tables"]["training_costs"])
    inference = pd.read_csv(inputs["root"] / report["tables"]["inference_costs"])
    assert len(training) == 5 and len(inference) == 10
    assert (training.fit_seconds > 0).all()
    assert (inference.prediction_seconds > 0).all()
    assert set(inference.split) == {"test_2025", "test_2026"}
    for figure in report["figures"]:
        assert (inputs["root"] / figure).stat().st_size > 1000
    assert (
        generate_report(
            inputs["root"],
            inputs["main_evaluation"].evaluation_id,
            inputs["temporal_evaluation"].evaluation_id,
            "artifacts/reports/test",
        )["report_id"]
        == report["report_id"]
    )


def test_report_rejects_two_main_evaluations(synthetic_evaluations):
    from road_accidents.reporting.tables import generate_report

    inputs = synthetic_evaluations
    with pytest.raises(PipelineError):
        generate_report(
            inputs["root"],
            inputs["main_evaluation"].evaluation_id,
            inputs["main_evaluation"].evaluation_id,
            "artifacts/reports/wrong",
        )


def test_report_package_cli_returns_manifest_after_report(synthetic_evaluations, capsys):
    from road_accidents.cli import main

    inputs = synthetic_evaluations
    capsys.readouterr()
    args = [
        "--work-dir",
        str(inputs["root"]),
        "--result-format",
        "json",
        "report",
        "--main-evaluation",
        inputs["main_evaluation"].evaluation_id,
        "--temporal-evaluation",
        inputs["temporal_evaluation"].evaluation_id,
        "--output",
        "artifacts/reports/test",
        "--package-output",
        "artifacts/packages/test",
    ]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert (inputs["root"] / result["outputs"]["package_manifest"]).is_file()
    assert (inputs["root"] / result["outputs"]["manifest"]).is_file()
