import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict


class ExampleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    count: int


def test_help_exposes_global_options_without_creating_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from road_accidents.cli import main

    assert main(["--help"]) == 0
    assert "--result-format" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_unknown_command_returns_safe_json_error(capsys: pytest.CaptureFixture[str]) -> None:
    from road_accidents.cli import main

    assert main(["--result-format", "json", "secret-example-value"]) == 2
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["schema_version"] == "3.0.0"
    assert result["status"] == "failed"
    assert result["outputs"] == {}
    assert result["error"]["code"] == "usage_error"
    assert "secret-example-value" not in captured.out + captured.err


@pytest.mark.parametrize(
    "content",
    [
        "!!python/object/apply:os.system ['echo unsafe']",
        "count: 1\nunknown: secret-example-value",
        "count: 1\ncount: 2",
        "- not-a-mapping",
        "count: invalid",
    ],
)
def test_invalid_configuration_is_rejected_without_echoing_values(
    tmp_path: Path, content: str
) -> None:
    from road_accidents.config import ConfigError, load_config

    configuration = tmp_path / "configuration.yaml"
    configuration.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError) as error:
        load_config(configuration, ExampleConfig)
    assert "secret-example-value" not in str(error.value)


def test_safe_yaml_is_validated_against_requested_schema(tmp_path: Path) -> None:
    from road_accidents.config import load_config

    configuration = tmp_path / "configuration.yaml"
    configuration.write_text("count: 3", encoding="utf-8")
    assert load_config(configuration, ExampleConfig).count == 3
