import ast
import json
from pathlib import Path
import pytest

from videntra.registry.access import MediaResolver
from videntra.registry.store import sha256_file
from videntra.cli import load_manifest
from conftest import ROOT


def local_media(tmp_path, media):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"synthetic bytes, no decode claim")
    return dict(media, media_uri="asset://test/clip.mp4", sha256=sha256_file(path),
                rights_status="allowed", rights_reference="synthetic-test"), path


def test_media_allowlist_hash_and_unknown_rights(tmp_path, media):
    record, path = local_media(tmp_path, media)
    resolver = MediaResolver({"test": tmp_path}, [record])
    assert resolver.resolve(record["video_id"]) == path
    with pytest.raises(KeyError):
        resolver.resolve("gold-query-id")
    path.write_bytes(b"modified")
    with pytest.raises(ValueError):
        resolver.resolve(record["video_id"])
    record["rights_status"] = "unknown"
    with pytest.raises(PermissionError):
        MediaResolver({"test": tmp_path}, [record]).resolve(record["video_id"])


@pytest.mark.parametrize("locator", [
    "asset://test/labels.jsonl", "asset://test/../labels.jsonl",
    "asset://test/%2e%2e/labels.jsonl", "asset://test/C:/labels.mp4",
    "file:///secret.mp4", "asset://test/clip.mp4?gold=1",
])
def test_poison_label_and_traversal_denied(tmp_path, media, locator):
    record, _ = local_media(tmp_path, media)
    (tmp_path / "labels.jsonl").write_text('{"target_video":"secret"}')
    record["media_uri"] = locator
    with pytest.raises(PermissionError):
        MediaResolver({"test": tmp_path}, [record]).resolve(record["video_id"])


def test_denied_root_even_if_label_disguised_as_media(tmp_path, media):
    record, _ = local_media(tmp_path, media)
    with pytest.raises(PermissionError):
        MediaResolver({"test": tmp_path}, [record], denied_roots=(tmp_path,)).resolve(record["video_id"])


def test_manifest_cannot_embed_labels(tmp_path, media):
    media["gold_start_us"] = 123
    path = tmp_path / "poison.json"
    path.write_text(json.dumps({"kind": "media_asset", "revision": "r", "records": [media]}))
    with pytest.raises(ValueError):
        load_manifest(path)


def test_runtime_imports_no_evaluator_or_legacy():
    for path in (ROOT / "src/videntra").rglob("*.py"):
        if "evaluation" in path.parts:
            continue
        tree = ast.parse(path.read_text("utf-8"))
        for node in ast.walk(tree):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else (
                [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert all("evaluation" not in name and "lab." not in name for name in names), f"Forbidden import in {path}: {names}"

    # Verify benchmark scripts do not pass GT labels into runtime search() calls (E12)
    bench_script = ROOT / "scripts/run_validation_benchmark.py"
    if bench_script.exists():
        b_tree = ast.parse(bench_script.read_text("utf-8"))
        for node in ast.walk(b_tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "search":
                kw_names = {kw.arg for kw in node.keywords if kw.arg}
                forbidden_kw = {"target_vids", "gt_moments", "gold_video_ids", "ground_truth"}
                assert not (kw_names & forbidden_kw), f"GT leakage into search() call: {kw_names}"


