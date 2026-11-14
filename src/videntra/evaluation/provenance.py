"""Hash immutable run inputs and verify exported files without copying media."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_paths(paths: Iterable[Path]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for path in paths:
        path = Path(path)
        if path.is_dir():
            for item in sorted(path.rglob("*")):
                if item.is_file():
                    result[str(item)] = sha256_file(item)
        elif path.is_file():
            result[str(path)] = sha256_file(path)
        else:
            result[str(path)] = None
    return result


def verify_hashes(hashes: dict[str, str | None]) -> bool:
    return all((sha256_file(Path(path)) if Path(path).is_file() else None) == digest
               for path, digest in hashes.items())
