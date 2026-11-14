"""Download a frozen QVH evaluation inventory into a new quarantine directory.

This script is evaluation-only: its annotations must never be used to select
retrieval candidates or samples. Downloading does not authorize inference or
upload to a third party.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from videntra.evaluation.split_lock import verify_split_lock
from videntra.evaluation.provenance import sha256_file

REPO = "ayushsdev/qvhighlights-videos"
REVISION = "085b603a6b9fdf3db7508398614307c24ba5600d"


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def metadata(url: str) -> tuple[str, int]:
    try:
        response = build_opener(_NoRedirect).open(Request(url, method="HEAD"), timeout=30)
    except HTTPError as exc:
        if exc.code not in (301, 302, 303, 307, 308):
            raise
        response = exc
    with response:
        return (response.headers.get("x-linked-etag") or "").strip('"'), int(response.headers.get("x-linked-size", "0"))


def acquire(lock: Path, annotations: Path, output: Path) -> None:
    if not verify_split_lock(lock):
        raise ValueError("Invalid split manifest")
    data = json.loads(lock.read_text(encoding="utf-8"))
    expected_annotations = data.get("metadata", {}).get("source_files_sha256", {}).get("test")
    if not expected_annotations or sha256_file(annotations) != expected_annotations:
        raise ValueError("Annotation file does not match the frozen split source")
    rows = [json.loads(line) for line in annotations.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {f"qvh_{row['qid']}": row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate annotation query IDs")
    videos = sorted({by_id[row["query_id"]]["vid"] for row in data["records"] if row["split"] == "test"})
    if not videos or any(not re.fullmatch(r"[A-Za-z0-9_-]{11}_[0-9.]+_[0-9.]+", vid) for vid in videos):
        raise ValueError("Empty inventory or invalid video ID")
    output.mkdir(parents=True, exist_ok=False)
    receipt = output / "source_receipts.jsonl"
    with receipt.open("x", encoding="utf-8") as records:
        for vid in videos:
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{vid[0].lower()}/{vid}.mp4"
            expected, size = metadata(url)
            if not re.fullmatch(r"[a-f0-9]{64}", expected) or size <= 0:
                raise ValueError(f"Missing checksum/size for {vid}")
            target = output / f"{vid}.mp4"
            partial = output / f"{vid}.partial"
            digest = hashlib.sha256()
            written = 0
            try:
                with urlopen(url, timeout=120) as response, partial.open("xb") as stream:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                if digest.hexdigest() != expected or written != size:
                    raise ValueError(f"Checksum or size mismatch for {vid}")
                partial.rename(target)
            except Exception:
                if partial.exists():
                    partial.unlink()
                raise
            records.write(json.dumps({"video_id": vid, "source_url": url, "revision": REVISION,
                "sha256": expected, "bytes": size, "rights_status": "unknown",
                "usage_status": "quarantined_not_approved_for_inference_or_upload"}) + "\n")
            records.flush()
            print(f"Verified {len(list(output.glob('*.mp4')))}/{len(videos)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acquire(args.split_lock, args.annotations, args.output)
