"""Minimal config loading and validation for early smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a smoke-test config is missing required fields."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        data = _load_simple_yaml(text)
    else:
        data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ConfigError(f"Config must parse to a mapping: {config_path}")
    validate_config(data)
    return data


def validate_config(config: dict[str, Any]) -> None:
    required_sections = [
        "experiment",
        "data",
        "model",
        "flow_matching",
        "corruption",
        "train",
        "sample",
        "eval",
        "logging",
    ]
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ConfigError(f"Missing config sections: {', '.join(missing)}")

    _require(config, "experiment.name", str)
    _require(config, "experiment.seed", int)
    _require(config, "experiment.output_root", str)
    _require(config, "data.dataset_name", str)
    _require(config, "data.processed_root", str)
    _require(config, "data.split_root", str)
    split_type = _require(config, "data.split_type", str)
    if split_type not in {"random", "scaffold"}:
        raise ConfigError("data.split_type must be 'random' or 'scaffold'")

    backbone = _require(config, "model.backbone", str)
    if not backbone:
        raise ConfigError("model.backbone must be non-empty")

    objective = _require(config, "flow_matching.objective", str)
    interpolation = _require(config, "flow_matching.interpolation", str)
    if objective != "cartesian_linear" or interpolation != "linear":
        raise ConfigError("Only cartesian_linear / linear is supported in the smoke scaffold")

    mode = _require(config, "corruption.mode", str)
    if mode not in {"isotropic", "stiffness_structured"}:
        raise ConfigError("corruption.mode must be 'isotropic' or 'stiffness_structured'")

    for field in [
        "corruption.isotropic_sigma",
        "corruption.bond_sigma",
        "corruption.angle_sigma",
        "corruption.torsion_sigma",
    ]:
        value = _require(config, field, (int, float))
        if value < 0:
            raise ConfigError(f"{field} must be non-negative")

    _require(config, "train.enabled", bool)
    _require(config, "sample.enabled", bool)
    _require(config, "eval.enabled", bool)


def _require(config: dict[str, Any], dotted_key: str, expected_type: type | tuple[type, ...]) -> Any:
    cursor: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise ConfigError(f"Missing required config value: {dotted_key}")
        cursor = cursor[part]
    if not isinstance(cursor, expected_type):
        raise ConfigError(
            f"Config value {dotted_key} must be {expected_type}, got {type(cursor).__name__}"
        )
    return cursor


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by the bundled smoke config."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ConfigError(f"Unsupported YAML line: {raw_line}")
        key, raw_value = stripped.split(":", 1)
        value_text = raw_value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ConfigError(f"Invalid indentation near: {raw_line}")

        parent = stack[-1][1]
        if value_text == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value_text)

    return root


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        if any(marker in value for marker in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")
