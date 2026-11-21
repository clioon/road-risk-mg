import importlib
import importlib.metadata
import math
import socket
import sys
from pathlib import Path

import pytest
from pytest_socket import SocketBlockedError

from road_accidents import CONTRACT_VERSION, __version__


def test_package_is_installed_on_supported_python() -> None:
    assert sys.version_info[:2] == (3, 12)
    assert importlib.metadata.version("road-accidents") == __version__
    assert CONTRACT_VERSION == "3.0.0"


@pytest.mark.parametrize(
    "module",
    [
        "osmnx",
        "geopandas",
        "duckdb",
        "pyarrow",
        "sklearn",
        "xgboost",
        "torch_geometric",
        "mlflow",
        "dvc",
        "pandera",
        "typer",
    ],
)
def test_scientific_dependency_imports_without_network(module: str) -> None:
    assert importlib.import_module(module).__name__ == module


def test_cpu_tensor_backend() -> None:
    import torch

    values = torch.tensor([1.0, 2.0], device="cpu")
    assert torch.version.cuda is None
    assert values.sum().item() == 3.0


def test_metric_projection_is_available() -> None:
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5880", always_xy=True)
    easting, northing = transformer.transform(-44.0, -19.0)
    assert math.isfinite(easting)
    assert math.isfinite(northing)


def test_parquet_backend_round_trip(tmp_path: Path) -> None:
    import pyarrow as arrow
    import pyarrow.parquet as parquet

    original = arrow.table({"segment_id": ["segment-1"], "event_count": [0]})
    destination = tmp_path / "compatibility.parquet"
    parquet.write_table(original, destination)
    assert parquet.read_table(destination).equals(original)


def test_default_suite_cannot_open_network_socket() -> None:
    with pytest.raises(SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def test_checked_in_source_configs_parse_without_real_data(project_root: Path) -> None:
    from road_accidents.config import load_config
    from road_accidents.data.schemas import CoverageConfig, SourcesConfig

    sources = load_config(project_root / "configs/sources.yaml", SourcesConfig)
    coverage = load_config(project_root / "configs/coverage.yaml", CoverageConfig)
    assert sources.profile == coverage.profile == "confirmatory"
    assert set(sources.years) == {entry.nominal_year for entry in coverage.sources}
    assert len(coverage.sources) == 8
