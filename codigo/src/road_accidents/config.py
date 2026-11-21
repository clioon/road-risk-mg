from collections.abc import Hashable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError


class PipelineError(Exception):
    def __init__(self, message: str, exit_code: int = 4, code: str = "data_error") -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.code = code


class ConfigError(PipelineError):
    def __init__(self) -> None:
        super().__init__("Invalid configuration.", exit_code=2, code="configuration_error")


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UniqueSafeLoader(yaml.SafeLoader):
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        self.flatten_mapping(node)
        result: dict[Hashable, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ConfigError()
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def load_config[ModelType: BaseModel](path: Path, schema: type[ModelType]) -> ModelType:
    try:
        with path.open("rb") as source:
            payload = source.read(1_048_577)
        if len(payload) > 1_048_576:
            raise ConfigError()
        text = payload.decode("utf-8-sig")
        if any(isinstance(event, yaml.AliasEvent) for event in yaml.parse(text)):
            raise ConfigError()
        content = yaml.load(text, Loader=UniqueSafeLoader)
        if not isinstance(content, dict):
            raise ConfigError()
        return schema.model_validate(content)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError, ValueError, TypeError):
        raise ConfigError() from None
