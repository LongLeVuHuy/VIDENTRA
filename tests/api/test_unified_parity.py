from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import sqlite3

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient

from videntra.api.app import create_app
from videntra.cli import handle_search
from videntra.retrieval.unified_pipeline import UnifiedMultimodalSearchPipeline, unified_result_dict


@pytest.fixture
def pipeline():
    p = UnifiedMultimodalSearchPipeline(lazy_load=True, use_disk_cache=False)
    p.augmented_embeddings = np.zeros((2, 864), dtype=np.float32)
    p.augmented_embeddings[:, 0] = [0.9, 0.3]
    p.aug_index = faiss.IndexFlatIP(864)
    p.aug_index.add(p.augmented_embeddings)
    p.index = faiss.IndexFlatIP(768)
    p.index.add(np.zeros((2, 768), dtype=np.float32))
    p.keyframe_map = {i: {"video_id": f"v{i}", "pts_time": 2.0,
                          "source_frame_index": i} for i in range(2)}
    p.video_keyframes = {f"v{i}": [{"faiss_id": i, "pts_time": 2.0}] for i in range(2)}
    p.video_durations = {"v0": 10.0, "v1": 10.0}
    def encode(texts):
        vectors = np.zeros((len(texts), 768), dtype=np.float32)
        vectors[:, 0] = 1
        return vectors
    p.encode_texts = encode
    return p


@pytest.mark.parametrize("mode", ["fast", "accurate"])
def test_core_api_cli_same_results(pipeline, monkeypatch, mode, tmp_path):
    import videntra.retrieval.unified_pipeline as module
    monkeypatch.setattr(module, "UnifiedMultimodalSearchPipeline", lambda **kwargs: pipeline)
    kwargs = dict(query="a person", top_k=2, mode=mode, ranking_mode="moment")
    expected = [unified_result_dict(m, "s1") for m in pipeline.search_with_metadata(snapshot_id="s1", **kwargs)["moments"]]
    client = TestClient(create_app(pipeline=pipeline, snapshot_id="s1"))
    response = client.post("/v1/search", json=kwargs)
    assert response.status_code == 200
    assert response.json()["snapshot_id"] == "s1"
    assert response.json()["results"] == expected
    assert all(r["provenance"]["snapshot_verified"] is False and r["timing_metadata"]["time_base"] is None for r in expected)
    db, idx, model = (tmp_path / name for name in ("db", "idx", "model"))
    db.touch()
    idx.touch()
    model.mkdir()
    args = SimpleNamespace(backend="unified", mock=False, db_path=db, faiss_path=idx,
                           model_path=model, device="cpu", snapshot_id="s1", query_id=None,
                           output=None, **kwargs)
    assert handle_search(args)["results"] == expected


def test_concurrent_metadata_does_not_leak(pipeline):
    def run(i):
        return pipeline.search_with_metadata(snapshot_id=str(i), query=f"query {i}", top_k=1,
                                             mode="fast", ranking_mode="moment")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(12)))
    assert [r["snapshot_id"] for r in results] == [str(i) for i in range(12)]
    assert all(r["timings"]["t_total_ms"] >= 0 and not r["diagnostics"] for r in results)


def test_lexical_database_error_is_not_reported_as_complete(pipeline):
    pipeline.db_conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(RuntimeError, match="Lexical retrieval failed"):
            pipeline.search_with_metadata(snapshot_id="s1", query='"unusualword"', top_k=1,
                                          mode="accurate", ranking_mode="moment")
    finally:
        pipeline.db_conn.close()


def test_relation_assessment_uses_only_evidence_in_window(pipeline):
    from videntra.retrieval.visual import RankedMoment
    pipeline.db_conn = sqlite3.connect(":memory:")
    pipeline.db_conn.execute("CREATE TABLE OCRObservations (observation_id TEXT, raw_text TEXT, video_id TEXT, timestamp_ms INTEGER, model_id TEXT)")
    pipeline.db_conn.execute("CREATE TABLE ASRSegments (segment_id TEXT, raw_text TEXT, video_id TEXT, start_ms INTEGER, end_ms INTEGER, model_id TEXT)")
    pipeline.db_conn.executemany("INSERT INTO OCRObservations VALUES (?, ?, ?, ?, ?)",
                                [("inside", "STOP", "v0", 2000, "ocr-fixture"), ("outside", "STOP", "v0", 9000, "ocr-fixture")])
    moment = RankedMoment("m", "v0", 1, 4, 0, 0, 0, 2, 0.5)
    try:
        result = pipeline._verify_relations('text "STOP"', [moment])[0]
        assert result["predicates"]["sale_ocr"]["status"] == "supported"
        assert result["predicates"]["sale_ocr"]["evidence_refs"] == ["inside"]
        assert result["evidence"][0]["model_id"] == "ocr-fixture"
        assert result["evidence"][0]["timestamp_seconds"] == 2.0
        assert result["evidence"][0]["generator_type"] == "model"
        result = pipeline._verify_relations('text "MISSING"', [moment])[0]
        assert result["predicates"]["sale_ocr"]["status"] == "unknown"
    finally:
        pipeline.db_conn.close()


def test_relation_asr_provenance_excludes_overlap_and_empty_segments(pipeline):
    from videntra.retrieval.visual import RankedMoment
    pipeline.db_conn = sqlite3.connect(":memory:")
    pipeline.db_conn.execute("CREATE TABLE OCRObservations (observation_id TEXT, raw_text TEXT, video_id TEXT, timestamp_ms INTEGER, model_id TEXT)")
    pipeline.db_conn.execute("CREATE TABLE ASRSegments (segment_id TEXT, raw_text TEXT, video_id TEXT, start_ms INTEGER, end_ms INTEGER, model_id TEXT)")
    pipeline.db_conn.executemany("INSERT INTO ASRSegments VALUES (?, ?, ?, ?, ?, ?)", [
        ("overlap", "hello", "v0", 500, 2000, "asr-model"),
        ("empty", "hello", "v0", 2000, 2000, "asr-model"),
        ("inside", "hello", "v0", 1500, 2500, "asr-model")])
    try:
        moment = RankedMoment("m", "v0", 1, 3, 0, 0, 0, 2, 0.5)
        result = pipeline._verify_relations("says hello", [moment])[0]
        assert result["predicates"]["speak_asr"]["evidence_refs"] == ["inside"]
        assert result["evidence"][0]["start_seconds"] == 1.5
        assert result["evidence"][0]["end_seconds"] == 2.5
        assert result["evidence"][0]["model_id"] == "asr-model"
    finally:
        pipeline.db_conn.close()


@pytest.mark.parametrize(("start_ms", "end_ms", "expected"), [
    (500, 2000, "unknown"), (1500, 2500, "supported"),
    (2500, 3500, "unknown"), (2000, 2000, "unknown"), (1000, 3000, "supported"),
])
def test_asr_interval_parity_with_relation_ranker(pipeline, start_ms, end_ms, expected):
    from videntra.query.rules import RuleBasedQueryParser
    from videntra.ranking.ranker import RelationAwareRanker
    from videntra.retrieval.visual import RankedMoment

    pipeline.db_conn = sqlite3.connect(":memory:")
    pipeline.db_conn.execute("CREATE TABLE OCRObservations (observation_id TEXT, raw_text TEXT, video_id TEXT, timestamp_ms INTEGER, model_id TEXT)")
    pipeline.db_conn.execute("CREATE TABLE ASRSegments (segment_id TEXT, raw_text TEXT, video_id TEXT, start_ms INTEGER, end_ms INTEGER, model_id TEXT)")
    pipeline.db_conn.execute("INSERT INTO ASRSegments VALUES (?, ?, ?, ?, ?, ?)",
                             ("e", "hello", "v0", start_ms, end_ms, "asr-fixture"))
    moment = RankedMoment("m", "v0", 1, 3, 0, 0, 0, 2, 0.5)
    try:
        unified = pipeline._verify_relations("says hello", [moment])[0]
        evidence = {"v0": [{"kind": "asr_segment", "text": "hello", "evidence_id": "e",
                             "start_seconds": start_ms / 1000, "end_seconds": end_ms / 1000}]}
        ranked = RelationAwareRanker().rank([moment], RuleBasedQueryParser().parse("says hello"), evidence)
        assert unified["status"] == ranked[0].match_status == expected
    finally:
        pipeline.db_conn.close()


def test_api_exposes_requested_relation_assessment(pipeline):
    response = TestClient(create_app(pipeline=pipeline, snapshot_id="s1")).post(
        "/v1/search", json={"query": 'text "STOP"', "mode": "accurate", "top_k": 1,
                            "verify_relations": True})
    assert response.status_code == 200
    assert response.json()["results"][0]["relation_assessment"]["predicates"]["sale_ocr"]["status"] == "unknown"


def test_cli_unified_verify_relations_attaches_assessment(pipeline, monkeypatch, tmp_path):
    import videntra.retrieval.unified_pipeline as module
    monkeypatch.setattr(module, "UnifiedMultimodalSearchPipeline", lambda **kwargs: pipeline)
    db, idx, model = (tmp_path / name for name in ("db", "idx", "model"))
    db.touch()
    idx.touch()
    model.mkdir()
    args = SimpleNamespace(
        backend="unified",
        mock=False,
        db_path=db,
        faiss_path=idx,
        model_path=model,
        device="cpu",
        snapshot_id="s1",
        query='text "STOP"',
        query_id="q1",
        top_k=1,
        mode="accurate",
        ranking_mode="moment",
        verify_relations=True,
        output=None,
    )
    out = handle_search(args)
    assert out["warnings"] == []
    assert len(out["results"]) == 1
    assert out["results"][0]["relation_assessment"]["predicates"]["sale_ocr"]["status"] == "unknown"

