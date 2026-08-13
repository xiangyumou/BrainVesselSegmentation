from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def resolve_data_root(config: dict[str, Any]) -> Path:
    configured = config.get("data", {}).get("root")
    value = configured or os.environ.get("BVS_DATA_ROOT")
    if not value:
        raise RuntimeError("Set data.root in the config or define BVS_DATA_ROOT")
    return Path(os.path.expandvars(str(value))).expanduser().resolve()

