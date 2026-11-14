"""Comprehensive tests for Milestone 7-A FastAPI Search, Playback, and Export routes."""
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from videntra.api.app import create_app
from videntra.config.ingestion import IngestionConfig, SamplingConfig
from videntra.fixtures import build_fixtures
from videntra.ingestion.jobs import IngestionJobManager
from videntra.registry.access import MediaResolver
from videntra.registry.store import sha256_file
from videntra.retrieval.visual import VisualRetrievalEngine, RankedMoment
from videntra.encoders.siglip2 import MockSigLIP2Encoder


@pytest.fixture(scope="module")
def api_test_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("api_env")
    fixtures_dir = root / "fixtures"
    build_fixtures(fixtures_dir)

    ingest_out = root / "ingest"
    cfg = IngestionConfig(sampling=SamplingConfig(max_gap_seconds=0.25))
    mgr = IngestionJobManager(config=cfg, output_dir=ingest_out)

    cfr_path = fixtures_dir / "cfr.mkv"
    cfr_hash = sha256_file(cfr_path)
    video_id = f"sha256:{cfr_hash}"

    records = [{
        "schema_version": "1.0",
        "video_id": video_id,
        "source_video_id": "fixture:cfr",
        "sha256": cfr_hash,
        "media_uri": "asset://fixtures/cfr.mkv",
        "video_stream_index": 0,
        "audio_stream_index": None,
        "source_offset_us": 0,
        "source_mapping_kind": "constant_offset",
        "timeline_origin": {"num": 0, "den": 1},
        "duration_us": 1000000,
        "rights_status": "allowed",
        "rights_reference": "license://fixture-test",
        "original_dataset_id": "FixtureSuite",
        "official_split": "train",
        "exposure": "train",
        "decode_status": "pending",
        "timeline_revision": None,
        "modality_status": {"visual": "not_run", "ocr": "not_run", "asr": "not_run"},
    }]

    resolver = MediaResolver({"fixtures": fixtures_dir}, records)
    mgr.execute_manifest(records, resolver=resolver, for_inference=True)

    encoder = MockSigLIP2Encoder()
    engine = VisualRetrievalEngine(encoder=encoder)

    sample_path = list(ingest_out.glob("samples/*/*/samples.parquet"))[0]
    timeline_path = list(ingest_out.glob("timeline/*/*/frames.parquet"))[0]
    media_path = resolver.resolve(video_id, for_inference=True)
    engine.index_video(video_id, media_path, sample_path, timeline_path)

    app = create_app(
        engine=engine,
        resolver=resolver,
        ingest_dir=ingest_out,
        snapshot_id="test_snapshot_01",
    )
    client = TestClient(app)

    return {
        "client": client,
        "video_id": video_id,
        "resolver": resolver,
        "media_path": media_path,
        "ingest_dir": ingest_out,
    }


def test_health_check(api_test_env):
    client = api_test_env["client"]
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["engine_active"] is True
    assert data["resolver_active"] is True
    assert data["indexed_samples"] > 0
    assert data["snapshot_id"] == "test_snapshot_01"


def test_ui_inspector_html(api_test_env):
    client = api_test_env["client"]
    resp = client.get("/")
    assert resp.status_code == 200
    assert "VIDENTRA" in resp.text
    assert "<video id=\"player\"" in resp.text
    assert "Replay Moment" in resp.text


def test_search_endpoint(api_test_env):
    client = api_test_env["client"]
    video_id = api_test_env["video_id"]

    resp = client.post("/v1/search", json={
        "query": "màn hình kiểm tra màu sắc",
        "top_k": 5,
        "mode": "fast",
        "ranking_mode": "moment",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("complete", "partial")
    assert "request_id" in data
    assert "timings" in data
    assert data["timings"]["measured"] is True
    assert len(data["results"]) > 0

    first = data["results"][0]
    assert first["video_id"] == video_id
    assert "ranking_score" in first
    assert "timing_metadata" in first
    assert first["timing_metadata"]["frame_index_base"] == 0
    assert len(first["evidence"]) > 0


def test_search_diverse_video(api_test_env):
    client = api_test_env["client"]
    resp = client.post("/v1/search", json={
        "query": "color test pattern",
        "top_k": 5,
        "ranking_mode": "diverse_video",
    })
    assert resp.status_code == 200
    data = resp.json()
    vids = [r["video_id"] for r in data["results"]]
    assert len(vids) == len(set(vids))


def test_media_streaming_full_and_range_206(api_test_env):
    client = api_test_env["client"]
    video_id = api_test_env["video_id"]

    # Full media
    resp = client.get(f"/v1/media/{video_id}")
    assert resp.status_code == 200
    assert "Accept-Ranges" in resp.headers
    total_size = len(resp.content)
    assert total_size > 0

    # Range request for video player seek
    range_resp = client.get(f"/v1/media/{video_id}", headers={"Range": "bytes=0-499"})
    assert range_resp.status_code == 206
    assert range_resp.headers["Content-Range"].startswith("bytes 0-499/")
    assert len(range_resp.content) == 500


def test_media_security_and_404(api_test_env):
    client = api_test_env["client"]

    # Unknown video
    resp = client.get("/v1/media/unknown_nonexistent_id")
    assert resp.status_code == 404

    # Traversal attempt
    resp = client.get("/v1/media/../../windows/system32")
    assert resp.status_code in (404, 403)


def test_exact_frame_retrieval(api_test_env):
    client = api_test_env["client"]
    video_id = api_test_env["video_id"]

    resp = client.get(f"/v1/frames/{video_id}/0")
    assert resp.status_code == 200
    assert resp.headers["Content-Type"] == "image/jpeg"
    assert resp.headers["X-Frame-Index"] == "0"
    assert "X-PTS" in resp.headers
    assert len(resp.content) > 100


def test_export_csv_and_formula_safe_escaping(api_test_env):
    client = api_test_env["client"]
    video_id = api_test_env["video_id"]

    # Craft items with dangerous formula injection strings: =, +, -, @
    poisoned_results = [
        {
            "video_id": "=cmd|' /C calc'!A0",
            "source_video_id": "+cmd|' /C calc'!A0",
            "moment_start_seconds": 1.25,
            "moment_end_seconds": 3.50,
            "start_original_frame_index": 10,
            "end_original_frame_index": 25,
            "representative_original_frame_index": 15,
            "ranking_score": 0.88,
            "score_type": "uncalibrated_ranking",
            "match_status": "@malicious_formula",
            "timing_metadata": {"frame_index_base": 0, "coordinate_system": "clip_relative"},
        }
    ]

    resp = client.post("/v1/exports", json={
        "results": poisoned_results,
        "format": "csv",
        "snapshot_id": "snap_safe_01",
    })
    assert resp.status_code == 200
    assert resp.headers["Content-Type"].startswith("text/csv")
    csv_text = resp.text

    # Formulas must be escaped with single quote '
    assert "'=cmd|" in csv_text
    assert "'+cmd|" in csv_text
    assert "'@malicious_formula" in csv_text


def test_export_json(api_test_env):
    client = api_test_env["client"]
    test_results = [{"video_id": "v1", "ranking_score": 0.5}]
    resp = client.post("/v1/exports", json={
        "results": test_results,
        "format": "json",
        "snapshot_id": "snap_json_01",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["snapshot_id"] == "snap_json_01"
    assert data["count"] == 1
    assert data["results"] == test_results


def test_query_parse_endpoint(api_test_env):
    client = api_test_env["client"]
    resp = client.get("/v1/query/parse?query=người+áo+đỏ+nói+xin+chào")
    assert resp.status_code == 200
    data = resp.json()
    assert data["schema_version"] == "1.0"
    assert "entities" in data
    assert "predicates" in data
    assert data["original_query"] == "người áo đỏ nói xin chào"
    assert data["parser"]["status"] == "validated"


def test_search_accurate_mode_with_query_graph(api_test_env):
    client = api_test_env["client"]
    video_id = api_test_env["video_id"]

    resp = client.post("/v1/search", json={
        "query": "người mặc áo đỏ",
        "top_k": 3,
        "mode": "accurate",
        "ranking_mode": "moment",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("complete", "partial", "no_supported_match")
    assert data["parser_status"] == "validated"
    assert data["query_graph"] is not None
    assert len(data["results"]) > 0
    first = data["results"][0]
    assert "match_status" in first
    assert "unresolved_predicates" in first
