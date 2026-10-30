"""Load experiment parameters without machine-specific defaults."""
import copy
import json
import math
from pathlib import Path


def load_config(path, overrides=()):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    for entry in overrides:
        key, value = entry.split("=", 1)
        cursor = settings
        segments = key.split(".")
        for segment in segments[:-1]:
            cursor = cursor[segment]
        if segments[-1] not in cursor:
            raise ValueError("Unknown configuration key")
        cursor[segments[-1]] = json.loads(value)
    if settings["version"] != 1:
        raise ValueError("Unsupported parameter schema")
    def finite(value):
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, list):
            return all(finite(v) for v in value)
        return not isinstance(value, float) or math.isfinite(value)
    if not finite(settings):
        raise ValueError("Parameters must be finite")
    return copy.deepcopy(settings)


def save_config(settings, path):
    Path(path).write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
