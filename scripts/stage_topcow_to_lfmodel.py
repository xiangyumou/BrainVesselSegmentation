#!/usr/bin/env python3
"""Copy a TopCoW release into the immutable LFModel raw workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


DATASET_NAME = "Dataset001_BrainVesselSegmentation"
MANIFEST_NAME = f"{DATASET_NAME}_raw.json"


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def stage(source: Path, workspace: Path, overwrite: bool = False) -> dict[str, object]:
    source = source.expanduser().resolve()
    workspace = workspace.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source release does not exist: {source}")
    if source == workspace or source in workspace.parents:
        raise ValueError("Workspace must not contain or equal the source release")

    destination = workspace / "raw" / DATASET_NAME
    manifest_path = workspace / "manifests" / MANIFEST_NAME
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Source release contains no files: {source}")

    records: list[dict[str, object]] = []
    copied = skipped = 0
    for index, source_path in enumerate(files, 1):
        relative = source_path.relative_to(source)
        target = destination / relative
        source_size = source_path.stat().st_size
        source_hash = sha256(source_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not target.is_file():
                raise FileExistsError(f"Destination is not a regular file: {target}")
            target_matches = target.stat().st_size == source_size and sha256(target) == source_hash
            if target_matches:
                skipped += 1
            elif not overwrite:
                raise FileExistsError(
                    f"Destination conflicts with source (use --overwrite to replace): {target}"
                )
            else:
                _copy_atomic(source_path, target)
                copied += 1
        else:
            _copy_atomic(source_path, target)
            copied += 1

        if target.stat().st_size != source_size or sha256(target) != source_hash:
            raise OSError(f"Post-copy SHA256 verification failed: {target}")
        records.append({"path": relative.as_posix(), "size": source_size, "sha256": source_hash})
        if index == 1 or index % 25 == 0 or index == len(files):
            print(f"Verified {index}/{len(files)} files", flush=True)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset": DATASET_NAME,
        "source": str(source),
        "destination": str(destination),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(records),
        "total_bytes": sum(int(record["size"]) for record in records),
        "files": records,
    }
    atomic_json(manifest_path, manifest)
    print(f"Staging complete: copied={copied}, unchanged={skipped}, manifest={manifest_path}")
    return manifest


def _copy_atomic(source: Path, target: Path) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        shutil.copy2(source, temporary_path)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        stage(args.source, args.workspace, args.overwrite)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
