import json
from pathlib import Path
import pytest
from conftest import ROOT
from videntra.cli import load_manifest
from videntra.registry.access import MediaResolver


def test_asset_roots_configuration_relocation():
    config_path = ROOT / "configs" / "local" / "asset_roots.json"
    if not config_path.exists():
        pytest.skip("Local asset roots configuration not present")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert "roots" in config
    assert "msrvtt" in config["roots"]
    for ds_name, raw_root in config["roots"].items():
        if ds_name == "m1_fixtures":
            continue
        ds_root = Path(raw_root)
        # Must point to media directory (e.g. data/datasets/<id>/media), not annotations
        assert "datasets" in ds_root.parts or "data" in ds_root.parts
        assert "media" in ds_root.parts
        assert "annotations" not in ds_root.parts

    # Denied roots must contain evaluation paths
    denied = [Path(p) for p in config.get("denied_roots", [])]
    assert any("evaluation" in str(p).lower() for p in denied)

    # If legacy qvh assets directory is accessible on the test machine, verify MediaResolver
    qvh_root = Path(config["roots"].get("qvhighlights", ""))
    qvh_manifest_path = ROOT / "manifests" / "qvh24.media.json"
    if "qvhighlights" in config["roots"] and qvh_root.exists() and qvh_manifest_path.is_file():
        manifest = load_manifest(qvh_manifest_path)
        resolver = MediaResolver(
            {k: Path(v) for k, v in config["roots"].items()},
            manifest["records"],
            tuple(denied),
        )
        for record in manifest["records"]:
            resolved = resolver.resolve(record["video_id"], for_inference=False)
            assert resolved.exists()
            assert resolved.is_file()
            with pytest.raises(PermissionError):
                resolver.resolve(record["video_id"], for_inference=True)
