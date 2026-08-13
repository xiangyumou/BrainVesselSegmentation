#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

MINIMUM_FREE_BYTES = 40 * 1024**3

# Supply release URLs and checksums explicitly because challenge mirrors can change.
DATASETS = {
    "topcow": {"size": "10.5 GB"},
    "ixi": {"size": "517.3 MB"},
    "lausanne": {"size": "1.3 GB"},
}


def checksum(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely download TopCoW-related datasets")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--url", required=True, help="Official or institutional archive URL")
    checksums = parser.add_mutually_exclusive_group(required=True)
    checksums.add_argument("--sha256", help="Expected archive SHA256")
    checksums.add_argument("--md5", help="Expected archive MD5")
    args = parser.parse_args()
    destination = Path(args.destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(destination).free
    if free < MINIMUM_FREE_BYTES:
        print(
            f"Refusing download: {free / 1024**3:.1f} GB free; at least 40 GB is required.",
            file=sys.stderr,
        )
        return 2
    archive = destination / Path(urllib.parse.urlparse(args.url).path).name
    if not archive.name:
        raise ValueError("URL must identify an archive filename")
    urllib.request.urlretrieve(args.url, archive)
    algorithm = "sha256" if args.sha256 else "md5"
    expected = args.sha256 or args.md5
    actual = checksum(archive, algorithm)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"{algorithm.upper()} mismatch for {archive}: expected {expected}, got {actual}"
        )
    print(f"Downloaded {args.dataset} ({DATASETS[args.dataset]['size']}) to {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
