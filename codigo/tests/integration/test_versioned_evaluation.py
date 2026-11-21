def test_revised_2026_preserves_frozen_bundle_and_prior_evaluation(synthetic_bundle):
    from road_accidents.data.coverage import CoverageLedger
    from road_accidents.data.pipeline import build_dataset, validate_dataset
    from road_accidents.data.prf import ingest_prf
    from road_accidents.evaluation.runner import evaluate_bundle
    from tests.fixtures.prf_factory import coverage_document, prf_bytes

    inputs = synthetic_bundle
    bundle = inputs["bundle"]
    evaluate_bundle(
        inputs["root"], bundle.bundle_id, inputs["version"].dataset_version_id, "test_2025"
    )
    first = evaluate_bundle(
        inputs["root"], bundle.bundle_id, inputs["version"].dataset_version_id, "test_2026"
    )
    revised = ingest_prf(
        prf_bytes(2026, revision=True),
        inputs["root"],
        year=2026,
        source_url="https://example.org/2026.csv",
        profile="smoke_only",
    )
    snapshots = inputs["snapshots"][:-1] + [revised]
    coverage = CoverageLedger.from_documents(
        [coverage_document(source.nominal_year, source.snapshot_id) for source in snapshots],
        profile="smoke_only",
    )
    version = build_dataset(inputs["root"], inputs["config"], snapshots, inputs["osm"], coverage)
    validate_dataset(inputs["root"], version.dataset_version_id)
    second = evaluate_bundle(
        inputs["root"], bundle.bundle_id, version.dataset_version_id, "test_2026"
    )
    assert first.evaluation_id != second.evaluation_id and first.bundle_id == second.bundle_id
    assert version.development_digest == bundle.development_digest
    assert (inputs["root"] / first.manifest_path).is_file()
