"""Acquire two explicitly licensed open movies independently of any query labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REVISION = "7d3459f790d83b918b1f6202cbc1f52070235280"
ASSETS = [
    {"video_id": "big-buck-bunny", "name": "big-buck-bunny.mov", "bytes": 249229883,
     "sha256": "b2acb9bddcb384f9762f919af8f6d8b4be781e40ad2f35a37cc12beef55b9a27",
     "license": "CC-BY-3.0", "rights_source": "https://peach.blender.org/about/",
     "attribution": "Copyright 2008, Blender Foundation / www.bigbuckbunny.org"},
    {"video_id": "elephants-dream", "name": "elephants-dream.mov", "bytes": 152496338,
     "sha256": "6fc295e0e92835316ec95929ccfc664b05ed2434f673e6f5e684f98f7558a547",
     "license": "CC-BY-2.5", "rights_source": "https://orange.blender.org/press/",
     "attribution": "Copyright 2006, Blender Foundation / Netherlands Media Art Institute / www.elephantsdream.org"},
]


def acquire(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    with (output / "source_receipts.jsonl").open("x", encoding="utf-8") as receipts:
        for item in ASSETS:
            url = f"https://huggingface.co/datasets/Mixpeek/momentbench/resolve/{REVISION}/videos/{item['name']}"
            partial, target = output / (item["name"] + ".partial"), output / item["name"]
            digest = hashlib.sha256()
            size = 0
            try:
                with urlopen(url, timeout=120) as response, partial.open("xb") as file:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        file.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                if digest.hexdigest() != item["sha256"] or size != item["bytes"]:
                    raise ValueError(f"Checksum or size mismatch: {item['video_id']}")
                partial.rename(target)
            except Exception:
                if partial.exists():
                    partial.unlink()
                raise
            record = {**item, "mirror_url": url, "mirror_revision": REVISION,
                      "source_group": item["video_id"], "labels_status": "not_collected",
                      "rights_note": "Attribution required; this receipt is provenance, not a legal determination."}
            receipts.write(json.dumps(record, ensure_ascii=False) + "\n")
            receipts.flush()
            print(f"Verified {item['video_id']}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    acquire(parser.parse_args().output)
