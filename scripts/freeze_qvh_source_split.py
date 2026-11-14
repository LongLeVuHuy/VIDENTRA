"""Create a provisional source-disjoint QVH split from evaluation-only annotations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from videntra.evaluation.provenance import sha256_file
from videntra.evaluation.split_lock import freeze_split, qvh_source_id as source_id


def build(dev_path: Path, test_path: Path, output: Path) -> dict:
    def read(path):
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    dev, test = read(dev_path), read(test_path)
    dev_sources = {source_id(row["vid"]) for row in dev}
    kept_test = [row for row in test if source_id(row["vid"]) not in dev_sources]
    rows = [{"query_id": f"qvh_{row['qid']}", "split": split,
             "source_group": source_id(row["vid"])}
            for split, items in (("dev", dev), ("test", kept_test)) for row in items]
    manifest = freeze_split(rows, output, metadata={
        "source_files_sha256": {"dev": sha256_file(dev_path), "test": sha256_file(test_path)},
        "policy": "retain_dev_exclude_test_queries_sharing_youtube_source",
        "dev_count": len(dev), "test_count": len(kept_test), "test_excluded": len(test) - len(kept_test),
        "limitations": "Source ID grouping only; near duplicates, translations, labels and rights unverified.",
    })
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = build(args.dev, args.test, args.output)
    print(json.dumps(result["metadata"], ensure_ascii=False, indent=2))
