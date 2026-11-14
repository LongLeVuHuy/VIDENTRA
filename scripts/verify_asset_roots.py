"""Verify registered media after relocation; never reads annotation contents."""
import argparse
import json
from pathlib import Path
from videntra.cli import load_manifest
from videntra.registry.access import MediaResolver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = load_manifest(args.manifest)
    if manifest["kind"] != "media_asset":
        raise ValueError("A media manifest is required")
    resolver = MediaResolver(
        {key: Path(value) for key, value in config["roots"].items()},
        manifest["records"], tuple(Path(value) for value in config["denied_roots"]))
    verified, rights_denied = [], 0
    for record in manifest["records"]:
        resolver.resolve(record["video_id"], for_inference=False)
        verified.append(record["video_id"])
        if record["rights_status"] != "allowed":
            try:
                resolver.resolve(record["video_id"])
            except PermissionError:
                rights_denied += 1
            else:
                raise AssertionError("Unknown/restricted rights allowed inference")
    print(json.dumps({"media_hashes_verified": len(verified),
                      "inference_denied_by_rights": rights_denied,
                      "annotation_contents_read": False}))


if __name__ == "__main__":
    main()

