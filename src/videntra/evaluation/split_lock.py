"""Validate and freeze explicitly supplied source groups for evaluation splits."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def qvh_source_id(video_id: str) -> str:
    match = re.fullmatch(r"(.{11})_[0-9.]+_[0-9.]+", video_id)
    if not match:
        raise ValueError(f"Unknown QVHighlights video ID format: {video_id}")
    return match.group(1)


def freeze_split(records: list[dict], output: Path, *, metadata: dict | None = None) -> dict:
    owners: dict[str, str] = {}
    query_ids: set[str] = set()
    normalized = []
    for row in records:
        query_id, split, group = (row.get(k) for k in ("query_id", "split", "source_group"))
        if not all(isinstance(x, str) and x.strip() for x in (query_id, split, group)):
            raise ValueError("Every row requires query_id, split and source_group")
        if query_id in query_ids:
            raise ValueError(f"Duplicate query_id: {query_id}")
        if group in owners and owners[group] != split:
            raise ValueError(f"Source group occurs in multiple splits: {group}")
        query_ids.add(query_id)
        owners[group] = split
        normalized.append({"query_id": query_id, "split": split, "source_group": group})
    if not normalized:
        raise ValueError("Cannot lock an empty split")
    normalized.sort(key=lambda row: row["query_id"])
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    metadata_payload = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest = {"schema_version": 1, "records_sha256": hashlib.sha256(payload).hexdigest(),
                "grouping_status": "caller_supplied_unverified", "records": normalized,
                "metadata": metadata or {}, "metadata_sha256": hashlib.sha256(metadata_payload).hexdigest()}
    with Path(output).open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return manifest


def verify_split_lock(path: Path) -> bool:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    records = manifest.get("records")
    if manifest.get("schema_version") != 1 or not isinstance(records, list):
        return False
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(payload).hexdigest() != manifest.get("records_sha256"):
        return False
    metadata_payload = json.dumps(manifest.get("metadata", {}), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(metadata_payload).hexdigest() != manifest.get("metadata_sha256"):
        return False
    owners, seen = {}, set()
    for row in records:
        if not isinstance(row, dict):
            return False
        query_id, split, group = (row.get(k) for k in ("query_id", "split", "source_group"))
        if not all(isinstance(x, str) and x.strip() for x in (query_id, split, group)):
            return False
        if query_id in seen or (group in owners and owners[group] != split):
            return False
        seen.add(query_id)
        owners[group] = split
    return bool(records)
