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
    assert all(
        r["timings"]["t_total_ms"] >= 0
        and isinstance(r["diagnostics"], dict)
        and set(r["diagnostics"].keys()) <= {"ranked_videos"}
        for r in results
    )


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


def _make_lightweight_unified_pipeline(**kwargs):
    p = UnifiedMultimodalSearchPipeline(lazy_load=True, use_disk_cache=False, **kwargs)
    p.augmented_embeddings = np.zeros((2, 864), dtype=np.float32)
    p.augmented_embeddings[:, 0] = [0.9, 0.3]
    p.aug_index = faiss.IndexFlatIP(864)
    p.aug_index.add(p.augmented_embeddings)
    p.index = faiss.IndexFlatIP(768)
    p.index.add(np.zeros((2, 768), dtype=np.float32))
    p.keyframe_map = {
        i: {"video_id": f"v{i}", "pts_time": 2.0, "source_frame_index": i}
        for i in range(2)
    }
    p.video_keyframes = {f"v{i}": [{"faiss_id": i, "pts_time": 2.0}] for i in range(2)}
    p.video_durations = {"v0": 10.0, "v1": 10.0}

    def encode(texts):
        vectors = np.zeros((len(texts), 768), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors

    p.encode_texts = encode
    return p


def test_unified_cli_and_api_default_to_rules_even_with_cuda_and_local_model(monkeypatch, tmp_path):
    from videntra.query.llm import LocalSmallLLMBackend

    models_dir = tmp_path / "models" / "llm"
    (models_dir / "qwen2.5-1.5b-instruct").mkdir(parents=True)
    monkeypatch.delenv("VIDENTRA_LLM_ENABLED", raising=False)
    monkeypatch.setattr(LocalSmallLLMBackend, "_has_cuda", staticmethod(lambda: True))
    monkeypatch.setattr(LocalSmallLLMBackend, "DEFAULT_MODELS_DIR", models_dir)

    llm_calls = 0

    def spy_generate(self, prompt: str, system_prompt: str | None = None):
        nonlocal llm_calls
        llm_calls += 1
        return "{}"

    monkeypatch.setattr(LocalSmallLLMBackend, "generate", spy_generate)

    # Medium/long query (>4 words) that passes complexity gate into LLM/rules path
    query = "a dog is swimming in a pool in the backyard"

    # 1. API path
    api_pipeline = _make_lightweight_unified_pipeline()
    client = TestClient(create_app(pipeline=api_pipeline, snapshot_id="s1"))
    resp = client.post("/v1/search", json={"query": query, "top_k": 2, "mode": "accurate", "ranking_mode": "moment"})
    assert resp.status_code == 200
    assert llm_calls == 0
    assert api_pipeline.last_diagnostics["llm_enabled"] is False
    assert api_pipeline.last_diagnostics["query_plan_mode"] == "fallback"

    # 2. CLI path
    import videntra.retrieval.unified_pipeline as module

    cli_pipeline = _make_lightweight_unified_pipeline()
    monkeypatch.setattr(module, "UnifiedMultimodalSearchPipeline", lambda **kw: cli_pipeline)
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
        query=query,
        query_id=None,
        top_k=2,
        mode="accurate",
        ranking_mode="moment",
        verify_relations=False,
        output=None,
    )
    cli_out = handle_search(args)
    assert cli_out["results"] == resp.json()["results"]
    assert llm_calls == 0
    assert cli_pipeline.last_diagnostics["llm_enabled"] is False
    assert cli_pipeline.last_diagnostics["query_plan_mode"] == "fallback"


def test_unified_explicit_llm_enable_and_frozen_cache_hit_miss(monkeypatch, tmp_path):
    import hashlib
    import json
    import unicodedata
    from videntra.query.llm import AdaptiveQueryDecomposer, LocalSmallLLMBackend, SemanticDecomposerValidator

    monkeypatch.delenv("VIDENTRA_LLM_ENABLED", raising=False)
    llm_calls = 0

    def spy_generate(self, prompt: str, system_prompt: str | None = None):
        nonlocal llm_calls
        llm_calls += 1
        return json.dumps(
            {
                "lang": "en",
                "events": [{"id": "e1", "vis": "a dog is swimming in a pool", "ocr": [], "asr": [], "objs": [], "pol": "pos"}],
                "spatial": [],
                "temporal": [],
            }
        )

    monkeypatch.setattr(LocalSmallLLMBackend, "generate", spy_generate)

    # Explicit enable via parameter
    query = "a dog is swimming in a pool in the backyard"
    p_explicit = _make_lightweight_unified_pipeline(llm_enabled=True, query_plan_prompt_version="extractive-v2")
    p_explicit.search_with_metadata(snapshot_id="s1", query=query, top_k=2, mode="accurate", ranking_mode="moment")
    assert llm_calls == 1
    assert p_explicit.last_diagnostics["llm_enabled"] is True
    assert p_explicit.last_diagnostics["query_plan_mode"] == "llm_compact"
    assert p_explicit.last_diagnostics["query_plan_prompt_version"] == "extractive-v2"

    # Frozen cache hit and miss
    rev = "rev-frozen-1"
    pv = "extractive-v2"
    wire = {
        "lang": "en",
        "events": [{"id": "e1", "vis": "a dog is swimming in a pool", "ocr": [], "asr": [], "objs": [], "pol": "pos"}],
        "spatial": [],
        "temporal": [],
    }
    _, plan = SemanticDecomposerValidator().validate_and_expand(
        wire,
        original_query=query,
        query_length_tier="medium",
        routing_mode="llm_compact",
        model_revision=rev,
        prompt_version=pv,
    )
    nfc = unicodedata.normalize("NFC", query.strip())
    key = hashlib.sha256(f"{nfc}|{rev}|{pv}".encode("utf-8")).hexdigest()
    cache_file = tmp_path / "frozen_cache.jsonl"
    cache_file.write_text(
        json.dumps(
            {
                "key": key,
                "model_revision": rev,
                "prompt_version": pv,
                "query_text_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                "plan": plan.to_dict(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    p_frozen = _make_lightweight_unified_pipeline(
        query_plan_cache=cache_file,
        query_plan_revision=rev,
        query_plan_prompt_version=pv,
    )
    # Cache hit -> 0 live LLM calls, llm_compact
    p_frozen.search_with_metadata(snapshot_id="s1", query=query, top_k=2, mode="accurate", ranking_mode="moment")
    assert llm_calls == 1  # unchanged
    assert p_frozen.last_diagnostics["query_plan_mode"] == "llm_compact"
    assert p_frozen.last_diagnostics["query_plan_cache_hit"] is True

    # Cache miss -> falls back to rules with 0 live LLM calls
    p_frozen.search_with_metadata(
        snapshot_id="s1",
        query="a cat is sleeping on a sofa in the living room",
        top_k=2,
        mode="accurate",
        ranking_mode="moment",
    )
    assert llm_calls == 1  # unchanged
    assert p_frozen.last_diagnostics["query_plan_mode"] == "fallback"
    assert p_frozen.last_diagnostics["query_plan_cache_hit"] is False


def test_unified_ranking_falls_back_to_original_query_when_extractive_plan_rejected(tmp_path):
    import json
    from videntra.query.llm import AdaptiveQueryDecomposer, LocalSmallLLMBackend

    query = "a dog is swimming in a pool in the backyard"
    hallucinated_wire = {
        "lang": "en",
        "events": [{"id": "e1", "vis": "a dog flies an airplane", "ocr": [], "asr": [], "objs": [], "pol": "pos"}],
        "spatial": [],
        "temporal": [],
    }
    decomposer = AdaptiveQueryDecomposer(
        llm_backend=LocalSmallLLMBackend(generate_fn=lambda _q: json.dumps(hallucinated_wire)),
        cache_path=tmp_path / "cache.jsonl",
        enable_disk_cache=False,
        query_plan_prompt_version="extractive-v2",
    )
    encoded_texts: list[str] = []
    pipeline = _make_lightweight_unified_pipeline(query_decomposer=decomposer)

    def spy_encode(texts):
        encoded_texts.extend(texts)
        vectors = np.zeros((len(texts), 768), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors

    pipeline.encode_texts = spy_encode
    pipeline.search_with_metadata(snapshot_id="s1", query=query, top_k=2, mode="accurate", ranking_mode="moment")
    assert pipeline.last_diagnostics["query_plan_mode"] == "fallback"
    assert "a dog flies an airplane" not in encoded_texts
    assert query in encoded_texts


