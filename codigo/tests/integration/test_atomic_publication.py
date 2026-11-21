import json
from pathlib import Path

import pytest


def test_successful_publication_is_immutable(tmp_path: Path) -> None:
    from road_accidents.data.storage import atomic_directory

    with atomic_directory(tmp_path, "versions/one") as staging:
        (staging / "content.txt").write_text("first", encoding="utf-8")
        assert not (tmp_path / "versions/one").exists()
    assert (tmp_path / "versions/one/content.txt").read_text() == "first"
    with pytest.raises(FileExistsError), atomic_directory(tmp_path, "versions/one"):
        pytest.fail("Existing immutable output must be rejected before writing.")
    assert (tmp_path / "versions/one/content.txt").read_text() == "first"


def test_interruption_keeps_failure_state_without_publishing_partial_output(tmp_path: Path) -> None:
    from road_accidents.data.storage import atomic_directory

    with pytest.raises(RuntimeError), atomic_directory(tmp_path, "versions/broken") as staging:
        (staging / "partial.txt").write_text("incomplete", encoding="utf-8")
        raise RuntimeError("secret-example-value")
    assert not (tmp_path / "versions/broken").exists()
    receipts = list((tmp_path / ".failures").glob("*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["status"] == "failed"
    assert "secret-example-value" not in receipts[0].read_text()


@pytest.mark.parametrize("relative", ["../outside", "/outside", "C:\\outside", "."])
def test_publication_cannot_escape_work_directory(tmp_path: Path, relative: str) -> None:
    from road_accidents.config import PipelineError
    from road_accidents.data.storage import atomic_directory

    with pytest.raises(PipelineError), atomic_directory(tmp_path, relative):
        pytest.fail("Unsafe paths must fail before staging.")


def test_tracking_retains_failed_run_state(tmp_path: Path) -> None:
    from road_accidents.provenance import RunTracker

    tracker = RunTracker(tmp_path)
    with pytest.raises(RuntimeError), tracker.run({"seed": "101"}) as run_id:
        raise RuntimeError("training failed")
    assert tracker.get_run(run_id).info.status == "FAILED"
    with tracker.run({"seed": "202"}) as completed:
        tracker.log_metrics(completed, {"score": 0.5})
    assert tracker.get_run(completed).info.status == "FINISHED"


def test_tabular_and_geometry_storage_preserve_content_and_crs(tmp_path: Path) -> None:
    import geopandas
    import pandas
    from shapely.geometry import Point

    from road_accidents.data.storage import (
        read_geoparquet,
        read_parquet,
        write_geoparquet,
        write_parquet,
    )

    table = pandas.DataFrame({"segment_id": ["segment-1"], "label": [0]})
    geometry = geopandas.GeoDataFrame(table, geometry=[Point(-44, -19)], crs="EPSG:4326")
    write_parquet(tmp_path / "table.parquet", table)
    write_geoparquet(tmp_path / "geometry.parquet", geometry)
    assert read_parquet(tmp_path / "table.parquet").equals(table)
    recovered = read_geoparquet(tmp_path / "geometry.parquet")
    assert recovered.equals(geometry)
    assert recovered.crs.to_epsg() == 4326
