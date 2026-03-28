"""Configuration loading and validation for config-driven curation."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_CONFIG: Dict[str, Any] = {
    "sources": [],
    "cleaning": {
        "enabled": True,
        "fix_unicode": True,
        "strip_control_chars": True,
        "normalize_newlines": True,
        "collapse_whitespace": True,
        "collapse_blank_lines": True,
    },
    "language": {
        "enabled": True,
        "backend": "fasttext",
        "model_path": None,
        "allow_fallback": True,
        "allowed": ["en"],
        "min_score": 0.8,
    },
    "heuristics": {
        "enabled": True,
        "min_chars": 200,
        "max_symbol_ratio": 0.2,
        "max_line_repetition_ratio": 0.3,
        "max_ngram_repetition_ratio": 0.35,
        "min_alpha_ratio": 0.6,
        "min_english_word_ratio": 0.02,
    },
    "privacy": {
        "enabled": True,
        "backend": "regex",
        "replace_with_type": True,
        "entities": [
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "ADDRESS",
            "IP_ADDRESS",
            "URL",
            "US_SSN",
            "DATE_TIME",
            "LOCATION",
        ],
    },
    "classifiers": {
        "quality": {"enabled": True, "gate": False},
        "domain": {"enabled": True, "gate": False},
        "toxicity": {"enabled": True, "gate": False},
    },
    "dedup": {
        "exact": {"enabled": True},
        "fuzzy": {
            "enabled": False,
            "threshold": 0.9,
            "num_hashes": 128,
            "ngram_size": 5,
            "seed": 42,
            "require_linux": True,
        },
    },
    "outputs": {
        "include_final_metadata": False,
    },
}


class ConfigError(ValueError):
    """Raised when the curation config is invalid."""


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_scalar(raw: str) -> Any:
    value = raw.strip()
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none", "~"}:
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return json.loads(value) if value.startswith('"') else value[1:-1]
    if value.startswith(("{", "[")):
        normalized = re.sub(r"\btrue\b", "true", value, flags=re.IGNORECASE)
        normalized = re.sub(r"\bfalse\b", "false", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bnull\b", "null", normalized, flags=re.IGNORECASE)
        return json.loads(normalized)
    return value


def _strip_inline_comment(raw_line: str) -> str:
    in_single = False
    in_double = False
    escaped = False
    result: List[str] = []
    for char in raw_line:
        if escaped:
            result.append(char)
            escaped = False
            continue
        if char == "\\" and in_double:
            result.append(char)
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            result.append(char)
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            result.append(char)
            continue
        if char == "#" and not in_single and not in_double and (not result or result[-1].isspace()):
            break
        result.append(char)
    return "".join(result).rstrip()


def _preprocess_yaml(text: str) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    for raw_line in text.splitlines():
        stripped = _strip_inline_comment(raw_line).rstrip()
        if not stripped:
            continue
        content = stripped.lstrip()
        if content.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        rows.append((indent, content))
    return rows


def _parse_yaml_block(lines: List[Tuple[int, str]], index: int, indent: int) -> Tuple[Any, int]:
    if index >= len(lines):
        return {}, index
    if lines[index][1].startswith("- "):
        return _parse_yaml_list(lines, index, indent)
    return _parse_yaml_map(lines, index, indent)


def _parse_yaml_map(lines: List[Tuple[int, str]], index: int, indent: int) -> Tuple[Dict[str, Any], int]:
    mapping: Dict[str, Any] = {}
    while index < len(lines):
        cur_indent, content = lines[index]
        if cur_indent < indent:
            break
        if cur_indent > indent:
            raise ConfigError(f"Unexpected indentation near: {content}")
        if content.startswith("- "):
            break
        if ":" not in content:
            raise ConfigError(f"Expected key/value pair, got: {content}")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        index += 1
        if raw_value:
            mapping[key] = parse_scalar(raw_value)
            continue
        if index < len(lines) and lines[index][0] > cur_indent:
            mapping[key], index = _parse_yaml_block(lines, index, lines[index][0])
        else:
            mapping[key] = {}
    return mapping, index


def _parse_yaml_list(lines: List[Tuple[int, str]], index: int, indent: int) -> Tuple[List[Any], int]:
    items: List[Any] = []
    while index < len(lines):
        cur_indent, content = lines[index]
        if cur_indent < indent:
            break
        if cur_indent > indent:
            raise ConfigError(f"Unexpected indentation near: {content}")
        if not content.startswith("- "):
            break
        item_content = content[2:].strip()
        index += 1
        if not item_content:
            if index < len(lines) and lines[index][0] > cur_indent:
                value, index = _parse_yaml_block(lines, index, lines[index][0])
            else:
                value = None
            items.append(value)
            continue

        if ":" in item_content and not item_content.startswith(("{", "[", "'", '"')):
            key, raw_value = item_content.split(":", 1)
            entry: Dict[str, Any] = {}
            key = key.strip()
            raw_value = raw_value.strip()
            if raw_value:
                entry[key] = parse_scalar(raw_value)
            else:
                if index < len(lines) and lines[index][0] > cur_indent:
                    entry[key], index = _parse_yaml_block(lines, index, lines[index][0])
                else:
                    entry[key] = {}
            if index < len(lines) and lines[index][0] > cur_indent:
                extra, index = _parse_yaml_block(lines, index, lines[index][0])
                if not isinstance(extra, dict):
                    raise ConfigError("List item mapping must expand to a mapping")
                entry.update(extra)
            items.append(entry)
            continue

        items.append(parse_scalar(item_content))
    return items, index


def load_yaml_without_dependency(text: str) -> Dict[str, Any]:
    lines = _preprocess_yaml(text)
    if not lines:
        return {}
    value, index = _parse_yaml_block(lines, 0, lines[0][0])
    if index != len(lines):
        raise ConfigError("Could not parse the full config file")
    if not isinstance(value, dict):
        raise ConfigError("The top-level config must be a mapping")
    return value


def load_structured_file(path: str) -> Dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")
    stripped = raw.lstrip()
    if not stripped:
        return {}
    if stripped.startswith(("{", "[")):
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ConfigError("The top-level config must be a mapping")
        return data
    try:
        import yaml  # type: ignore
    except ImportError:
        return load_yaml_without_dependency(raw)
    data = yaml.safe_load(raw)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError("The top-level config must be a mapping")
    return data


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(config.get("sources"), list) or not config["sources"]:
        raise ConfigError("config.sources must be a non-empty list")
    for source in config["sources"]:
        if not isinstance(source, dict):
            raise ConfigError("each source entry must be a mapping")
        if "path" not in source and "source" not in source:
            raise ConfigError("each source must define either 'path' or 'source'")
    allowed_language = config["language"].get("allowed", [])
    if not isinstance(allowed_language, list) or not allowed_language:
        raise ConfigError("language.allowed must be a non-empty list")
    return config


def load_pipeline_config(path: str) -> Dict[str, Any]:
    user_config = load_structured_file(path)
    config = deep_merge(DEFAULT_CONFIG, user_config)
    return validate_config(config)
