from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SPEC_FIELDS = {"filename", "directory", "pattern", "strategy"}


@dataclass(frozen=True)
class PatternCase:
    case_id: str
    modalities: dict[str, Path]
    label: Path


def unpack_pattern_spec(
    spec: str | dict[str, Any], default_directory: str, field: str
) -> tuple[str, str]:
    if isinstance(spec, str):
        pattern = spec
        directory = default_directory
    else:
        unknown = set(spec) - SPEC_FIELDS
        if unknown:
            raise ValueError(f"Unknown fields in {field}: {sorted(unknown)}")
        pattern = spec.get("pattern", spec.get("filename"))
        directory = spec.get("directory", default_directory)
    if not pattern:
        raise ValueError(f"{field} requires pattern")
    if str(pattern).count("{case_id}") != 1:
        raise ValueError(
            f"{field}.pattern must contain exactly one {{case_id}}: {pattern}"
        )
    if Path(str(pattern)).name != str(pattern):
        raise ValueError(f"{field}.pattern must be a filename, not a path: {pattern}")
    return str(directory), str(pattern)


def compile_case_pattern(pattern: str) -> re.Pattern[str]:
    expression = re.escape(pattern).replace(
        re.escape("{case_id}"), r"(?P<case_id>.+)"
    )
    return re.compile(f"^{expression}$")


def index_pattern_directory(directory: str | Path, pattern: str) -> dict[str, Path]:
    base = Path(directory).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Required directory does not exist: {base}")
    regex = compile_case_pattern(pattern)
    result: dict[str, Path] = {}
    for path in sorted(base.glob("*.nii.gz")):
        match = regex.match(path.name)
        if not match:
            continue
        case_id = match.group("case_id")
        if case_id in result:
            raise ValueError(f"Duplicate case ID '{case_id}' in {base}")
        result[case_id] = path
    if not result:
        raise FileNotFoundError(
            f"No NIfTI files in {base} match configured pattern: {pattern}"
        )
    return result


def resolve_input_directory(
    input_path: str | Path, configured_directory: str
) -> Path:
    source = Path(input_path).expanduser().resolve()
    nested = source / configured_directory
    return nested if nested.is_dir() else source


def render_pattern(pattern: str, case_id: str) -> str:
    return pattern.replace("{case_id}", case_id)


def discover_pattern_cases(
    data_root: str | Path,
    modality_specs: dict[str, str | dict[str, Any]],
    label_spec: str | dict[str, Any],
) -> list[PatternCase]:
    root = Path(data_root).expanduser().resolve()
    indexed_modalities: dict[str, dict[str, Path]] = {}
    for name, spec in modality_specs.items():
        directory, pattern = unpack_pattern_spec(
            spec, "imagesTr", f"data.modalities.{name}"
        )
        indexed_modalities[name] = index_pattern_directory(root / directory, pattern)
    label_directory, label_pattern = unpack_pattern_spec(
        label_spec, "cow_seg_labelsTr", "data.label"
    )
    labels = index_pattern_directory(root / label_directory, label_pattern)
    all_ids = set(labels)
    for values in indexed_modalities.values():
        all_ids |= set(values)
    errors: dict[str, list[str]] = {}
    for name, values in indexed_modalities.items():
        missing = sorted(all_ids - set(values))
        if missing:
            errors[f"missing_{name}"] = missing
    missing_labels = sorted(all_ids - set(labels))
    if missing_labels:
        errors["missing_labels"] = missing_labels
    if errors:
        raise ValueError(f"Pattern directory pairing failed: {errors}")
    return [
        PatternCase(
            case_id,
            {name: values[case_id] for name, values in indexed_modalities.items()},
            labels[case_id],
        )
        for case_id in sorted(all_ids)
    ]
