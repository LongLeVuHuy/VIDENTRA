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
    assert "qvhighlights" in config["roots"]
    qvh_root = Path(config["roots"]["qvhighlights"])

    # Must point to media directory (e.g. datasets/.../media or data/raw/.../media), not annotations
    assert "datasets" in qvh_root.parts or "data" in qvh_root.parts
    assert "media" in qvh_root.parts
    assert "annotations" not in qvh_root.parts

    # Denied roots must contain evaluation paths
    denied = [Path(p) for p in config.get("denied_roots", [])]
    assert any("evaluation" in str(p).lower() for p in denied)

    # If assets directory is accessible on the test machine, test MediaResolver behavior
    if qvh_root.exists():
        manifest = load_manifest(ROOT / "manifests" / "qvh24.media.json")
        resolver = MediaResolver(
            {k: Path(v) for k, v in config["roots"].items()},
            manifest["records"],
            tuple(denied),
        )
        for record in manifest["records"]:
            resolved = resolver.resolve(record["video_id"], for_inference=False)
            assert resolved.exists()
            assert resolved.is_file()
            # Unknown rights must be denied for inference
            with pytest.raises(PermissionError):
                resolver.resolve(record["video_id"], for_inference=True)
