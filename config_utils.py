import json
from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return [_parse_scalar(item) for item in inner.split(",")]
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _fallback_yaml_load(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith(" ") or line.startswith("\t"):
            raise ValueError("Fallback YAML parser only supports flat key-value mappings")
        if ":" not in line:
            raise ValueError(f"Invalid config line: {raw_line}")
        key, value = line.split(":", 1)
        result[key.strip()] = _parse_scalar(value)
    return result


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except ImportError:
        data = _fallback_yaml_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping at the top level")
    return dict(data)


def merge_config_with_cli(
    *,
    config_values: dict[str, Any],
    cli_values: dict[str, Any],
    cli_override_keys: set[str],
) -> dict[str, Any]:
    merged = dict(config_values)
    merged.update({key: value for key, value in cli_values.items() if key in cli_override_keys})
    return merged
