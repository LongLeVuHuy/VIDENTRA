import json
import subprocess
import sys
from videntra.cli import validate_manifests
from conftest import ROOT


def test_frozen_local24_manifest():
    docs = validate_manifests(ROOT / "manifests")
    media = next(d for d in docs if d["kind"] == "media_asset")["records"]
    assert len(media) == 24
    assert all(r["exposure"] == "development_previously_used" and r["official_split"] == "val" for r in media)
    assert all(r["rights_status"] == "unknown" and r["source_offset_us"] is None for r in media)
    assert all(r["decode_status"] == "pending" for r in media)


def test_cli_from_other_working_directory(tmp_path):
    completed = subprocess.run([sys.executable, "-m", "videntra", "manifests", "validate", "--all",
                                "--directory", str(ROOT / "manifests")],
                               cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["records"] == 25


def test_cli_failure_is_not_empty_success(tmp_path):
    completed = subprocess.run([sys.executable, "-m", "videntra", "manifests", "validate", "--all",
                                "--directory", str(tmp_path)], capture_output=True, text=True)
    assert completed.returncode == 2
    assert "error" in json.loads(completed.stderr)

