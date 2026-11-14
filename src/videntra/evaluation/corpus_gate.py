"""Freeze a label-free corpus inventory and audit evaluation coverage separately.

This module does not construct an index or run retrieval.  A receipt hash proves
only that the referenced receipt is unchanged; it is not a legal determination.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


_FIELDS = frozenset({"video_id", "sha256", "rights_status", "rights_receipt_path",
                     "rights_receipt_sha256", "source_group", "duplicate_group"})
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def freeze_inventory(records: Iterable[Mapping[str, Any]], output: Path, *, snapshot_id: str) -> dict[str, Any]:
    """Write a new inventory; unknown/restricted rights are never marked eligible.

    Records come from media/source registries, not query or gold files. Extra
    fields are rejected to prevent accidental copying of labels into inference.
    ``allowed`` is an externally reviewed registry assertion, not inferred here.
    """
    if not _nonempty(snapshot_id):
        raise ValueError("snapshot_id is required")
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    entries = []
    for item in records:
        if set(item) != _FIELDS:
            raise ValueError("Corpus inventory must contain only the required label-free fields")
        record = dict(item)
        video_id, digest = record["video_id"], record["sha256"]
        if not _nonempty(video_id) or not isinstance(digest, str) or not _HEX.fullmatch(digest):
            raise ValueError("Invalid video_id or sha256")
        if video_id in seen_ids or digest in seen_hashes:
            raise ValueError("Duplicate video ID or byte hash must be resolved explicitly")
        seen_ids.add(video_id)
        seen_hashes.add(digest)
        rights = record["rights_status"]
        if rights not in {"allowed", "unknown", "restricted"}:
            raise ValueError("Invalid rights status")
        reasons: list[str] = []
        if rights != "allowed":
            reasons.append("rights_not_allowed")
        else:
            receipt_path, expected = record["rights_receipt_path"], record["rights_receipt_sha256"]
            if not _nonempty(receipt_path) or not isinstance(expected, str) or not _HEX.fullmatch(expected):
                reasons.append("receipt_missing_or_invalid")
            else:
                path = Path(receipt_path)
                if not path.is_absolute() or not path.is_file() or _digest(path) != expected:
                    reasons.append("receipt_unverified")
        if not _nonempty(record["source_group"]):
            reasons.append("source_group_unknown")
        if not _nonempty(record["duplicate_group"]):
            reasons.append("duplicate_group_unknown")
        record["eligibility"] = "blocked" if reasons else "eligible_pending_independent_review"
        record["reasons"] = reasons
        entries.append(record)
    if not entries:
        raise ValueError("Inventory must not be empty")
    entries.sort(key=lambda entry: entry["video_id"])
    payload = {"schema_version": 1, "snapshot_id": snapshot_id, "entries": entries,
               "independence_status": "unverified",
               "limitations": ["Grouping and rights review are external assertions.",
                               "Media bytes, index contents, duplicate completeness and labels were not verified."]}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    result = {**payload, "manifest_sha256": hashlib.sha256(canonical).hexdigest()}
    with Path(output).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
    return result


def verify_inventory(path: Path) -> bool:
    """Detect accidental manifest changes; the checksum is not a signature."""
    try:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        checksum = result.pop("manifest_sha256")
        canonical = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return result.get("schema_version") == 1 and checksum == hashlib.sha256(canonical).hexdigest()
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def evaluate_coverage(qrels: Iterable[Mapping[str, Any]], indexed_video_ids: Iterable[str]) -> dict[str, Any]:
    """Evaluator-only coverage audit, retaining every query including unlabeled.

    This function must not be used to select retrieval candidates or sampling.
    Indexed IDs are supplied from a pre-frozen snapshot, never inferred from qrels.
    """
    indexed = set(indexed_video_ids)
    seen: set[str] = set()
    per_dataset: dict[str, Counter[str]] = {}
    for row in qrels:
        query_id, dataset = row.get("query_id"), row.get("dataset")
        if not _nonempty(query_id) or not _nonempty(dataset) or query_id in seen:
            raise ValueError("Invalid or duplicate query ID/dataset")
        seen.add(query_id)
        targets = row.get("target_videos")
        counts = per_dataset.setdefault(dataset, Counter())
        counts["queries"] += 1
        if targets is None or targets == []:
            counts["unlabeled"] += 1
            continue
        if not isinstance(targets, (list, tuple)) or not all(_nonempty(target) for target in targets):
            raise ValueError("target_videos must be a list of video IDs")
        counts["labeled"] += 1
        covered_count = sum(target in indexed for target in set(targets))
        if covered_count:
            counts["covered"] += 1
            if covered_count < len(set(targets)):
                counts["partially_covered"] += 1
        else:
            counts["not_covered"] += 1
    if not seen:
        raise ValueError("No queries to audit")
    return {dataset: {"queries": c["queries"], "labeled": c["labeled"], "unlabeled": c["unlabeled"],
                      "covered": c["covered"], "not_covered": c["not_covered"],
                      "partially_covered": c["partially_covered"],
                      "coverage": c["covered"] / c["labeled"] if c["labeled"] else None}
            for dataset, c in sorted(per_dataset.items())}
