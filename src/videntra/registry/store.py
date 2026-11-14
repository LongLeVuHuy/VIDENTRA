"""Append-only versioned registry; all batches commit atomically."""
import hashlib
import json
from pathlib import Path
import sqlite3

from videntra.contracts import validate


class RegistryConflict(ValueError):
    pass


def canonical(record: dict) -> str:
    return json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Registry:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS records(
                kind TEXT NOT NULL, record_id TEXT NOT NULL, revision TEXT NOT NULL,
                content TEXT NOT NULL, digest TEXT NOT NULL,
                PRIMARY KEY(kind,record_id,revision));
            CREATE TABLE IF NOT EXISTS media_identity(
                video_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS records_no_update BEFORE UPDATE ON records
                BEGIN SELECT RAISE(ABORT,'immutable registry'); END;
            CREATE TRIGGER IF NOT EXISTS records_no_delete BEFORE DELETE ON records
                BEGIN SELECT RAISE(ABORT,'immutable registry'); END;
            CREATE TRIGGER IF NOT EXISTS identity_no_update BEFORE UPDATE ON media_identity
                BEGIN SELECT RAISE(ABORT,'immutable identity'); END;
            CREATE TRIGGER IF NOT EXISTS identity_no_delete BEFORE DELETE ON media_identity
                BEGIN SELECT RAISE(ABORT,'immutable identity'); END;
        """)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.db.close()

    def register_batch(self, records: list[tuple[str, str, str, dict]]) -> int:
        added = 0
        with self.db:
            for kind, record_id, revision, record in records:
                if kind not in ("media_asset", "dataset"):
                    raise ValueError("Only media and dataset records belong in this registry")
                validate(kind, record)
                expected_id = record["video_id" if kind == "media_asset" else "dataset_id"]
                if record_id != expected_id or not revision:
                    raise ValueError("Record identity/revision mismatch")
                content = canonical(record)
                existing = self.db.execute(
                    "SELECT content FROM records WHERE kind=? AND record_id=? AND revision=?",
                    (kind, record_id, revision)).fetchone()
                if existing:
                    if existing[0] != content:
                        raise RegistryConflict("Immutable revision already contains different content")
                    continue
                if kind == "media_asset":
                    prior = self.db.execute("SELECT sha256 FROM media_identity WHERE video_id=?",
                                            (record_id,)).fetchone()
                    if prior and prior[0] != record["sha256"]:
                        raise RegistryConflict("A video ID cannot change content hash")
                    self.db.execute("INSERT OR IGNORE INTO media_identity VALUES (?,?)",
                                    (record_id, record["sha256"]))
                self.db.execute("INSERT INTO records VALUES (?,?,?,?,?)",
                                (kind, record_id, revision, content, hashlib.sha256(content.encode()).hexdigest()))
                added += 1
        return added

    def get(self, kind: str, record_id: str, revision: str) -> dict:
        row = self.db.execute(
            "SELECT content FROM records WHERE kind=? AND record_id=? AND revision=?",
            (kind, record_id, revision)).fetchone()
        if row is None:
            raise KeyError((kind, record_id, revision))
        return json.loads(row[0])

    def aliases(self, digest: str) -> list[str]:
        return [row[0] for row in self.db.execute(
            "SELECT video_id FROM media_identity WHERE sha256=? ORDER BY video_id", (digest,))]

