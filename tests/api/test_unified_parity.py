from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
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


@pytest.mark.parametrize(
    "query,dropped_vis",
    [
        ("the dog is not really swimming in the pool", "swimming in the pool"),
        ("two brown dogs are swimming in the pool", "dogs are swimming in the pool"),
        ("hai con chó nâu đang bơi trong hồ", "chó nâu đang bơi trong hồ"),
    ],
)
def test_unified_pipeline_falls_back_and_excludes_constraint_dropping_clauses(tmp_path, query, dropped_vis):
    import json
    from videntra.query.llm import AdaptiveQueryDecomposer, LocalSmallLLMBackend

    bad_wire = {
        "lang": "en",
        "events": [{"id": "e1", "vis": dropped_vis, "ocr": [], "asr": [], "objs": [], "pol": "pos"}],
        "spatial": [],
        "temporal": [],
    }
    decomposer = AdaptiveQueryDecomposer(
        llm_backend=LocalSmallLLMBackend(generate_fn=lambda _q: json.dumps(bad_wire)),
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
    assert query in encoded_texts
    assert dropped_vis not in encoded_texts


def test_unified_filters_non_coco_obj_reqs_and_skips_redundant_single_clause(tmp_path):
    import json
    from videntra.query.llm import AdaptiveQueryDecomposer, LocalSmallLLMBackend, SemanticDecomposerValidator

    query = "a comedy show with a baseball player in a sony channel"
    wire = {
        "lang": "en",
        "events": [
            {
                "id": "e1",
                "vis": "comedy show with baseball player in sony channel",
                "ocr": [],
                "asr": [],
                "objs": [
                    {"id": "c1", "cls": "comedy show", "cnt": 1, "color": ""},
                    {"id": "p1", "cls": "baseball player", "cnt": 1, "color": ""},
                    {"id": "ch1", "cls": "sony channel", "cnt": 1, "color": ""},
                ],
                "pol": "pos",
            }
        ],
        "spatial": [],
        "temporal": [],
    }
    _, plan = SemanticDecomposerValidator().validate_and_expand(
        wire,
        original_query=query,
        prompt_version="extractive-v2",
    )
    assert [o["class"] for o in plan.object_requirements] == ["person"]

    # Also test pipeline filtering even if a legacy serialized cache contains non-COCO labels
    legacy_wire = {
        "lang": "en",
        "events": [
            {
                "id": "e1",
                "vis": "comedy show with baseball player in sony channel",
                "ocr": [],
                "asr": [],
                "objs": [],
                "pol": "pos",
            }
        ],
        "spatial": [],
        "temporal": [],
    }
    decomposer = AdaptiveQueryDecomposer(
        llm_backend=LocalSmallLLMBackend(generate_fn=lambda _q: json.dumps(legacy_wire)),
        cache_path=tmp_path / "cache.jsonl",
        enable_disk_cache=False,
        query_plan_prompt_version="extractive-v2",
    )
    decomposer.decompose(query)  # populate RAM cache cleanly
    encoded_texts: list[str] = []
    pipeline = _make_lightweight_unified_pipeline(query_decomposer=decomposer)

    def spy_encode(texts):
        encoded_texts.extend(texts)
        vectors = np.zeros((len(texts), 768), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors

    pipeline.encode_texts = spy_encode
    pipeline.search_with_metadata(snapshot_id="s1", query=query, top_k=2, mode="accurate", ranking_mode="moment")
    # Single redundant clause missing only articles must NOT trigger sub_clauses embedding
    assert len(encoded_texts) == 2
    assert encoded_texts[0] == query
    assert "comedy show with baseball player in sony channel" not in encoded_texts


def test_unified_clip_consensus_boost_promotes_unanimous_top1():
    from videntra.retrieval.visual import RankedMoment

    p = UnifiedMultimodalSearchPipeline(lazy_load=True, use_disk_cache=False)
    # Build 5 synthetic videos with 2 frames each in 768d space
    emb_768 = np.zeros((10, 768), dtype=np.float32)
    # v_win (fids 2,3): highest max frame (~0.90) AND highest clip mean (~0.90)
    emb_768[2, 0] = 0.90
    emb_768[2, 1] = 0.43589
    emb_768[3, 0] = 0.90
    emb_768[3, 1] = 0.43589
    # v_a5 (fids 0,1): lower max frame (0.85) & lower clip mean
    emb_768[0, 0] = 0.85
    emb_768[0, 2] = 0.52678
    emb_768[1, 0] = 0.70
    emb_768[1, 2] = 0.71414
    for fid in range(4, 10):
        emb_768[fid, 0] = 0.20
        emb_768[fid, 3] = 0.97980
    p.all_embeddings = emb_768
    p.augmented_embeddings = np.pad(emb_768, ((0, 0), (0, 96)))
    p.video_keyframes = {
        "v_a5": [{"faiss_id": 0, "pts_time": 1.0}, {"faiss_id": 1, "pts_time": 2.0}],
        "v_win": [{"faiss_id": 2, "pts_time": 1.0}, {"faiss_id": 3, "pts_time": 2.0}],
        "v_2": [{"faiss_id": 4, "pts_time": 1.0}, {"faiss_id": 5, "pts_time": 2.0}],
        "v_3": [{"faiss_id": 6, "pts_time": 1.0}, {"faiss_id": 7, "pts_time": 2.0}],
        "v_4": [{"faiss_id": 8, "pts_time": 1.0}, {"faiss_id": 9, "pts_time": 2.0}],
    }
    p.video_kf_fids = {vid: [k["faiss_id"] for k in kfs] for vid, kfs in p.video_keyframes.items()}

    q_768 = np.zeros((768,), dtype=np.float32)
    q_768[0] = 1.0
    cands = [
        RankedMoment("m_a5", "v_a5", 0.0, 2.0, 0, 0, 0, 1.0, ranking_score=0.520),
        RankedMoment("m_win", "v_win", 0.0, 2.0, 2, 2, 2, 1.0, ranking_score=0.500),
        RankedMoment("m_2", "v_2", 0.0, 2.0, 4, 4, 4, 1.0, ranking_score=0.200),
    ]
    updated, score_res, stats = p._apply_clip_consensus_boost(cands, q_768, filters=None)
    assert score_res is not None
    assert stats["applied"] is True
    assert stats["promoted_video_id"] == "v_win"
    updated.sort(key=lambda m: (-float(m.ranking_score), str(m.video_id)))
    assert updated[0].video_id == "v_win"
    assert updated[0].ranking_score == pytest.approx(0.522, abs=1e-6)


def test_unified_preserves_single_non_redundant_llm_clause_and_normalizes_flat_screen(tmp_path):
    import json
    from videntra.query.llm import AdaptiveQueryDecomposer, LocalSmallLLMBackend

    query = "a man holding a sign watches two flat screens"
    wire = {
        "lang": "en",
        "events": [
            {
                "id": "e1",
                "vis": "man holding a sign",
                "ocr": [],
                "asr": [],
                "objs": [{"id": "o1", "cls": "flat screen", "cnt": 2, "color": ""}],
                "pol": "pos",
            }
        ],
        "spatial": [],
        "temporal": [],
    }
    decomposer = AdaptiveQueryDecomposer(
        llm_backend=LocalSmallLLMBackend(generate_fn=lambda _q: json.dumps(wire)),
        cache_path=tmp_path / "cache.jsonl",
        enable_disk_cache=False,
        query_plan_prompt_version="extractive-v2",
    )
    decomposer.decompose(query)
    encoded_texts: list[str] = []
    pipeline = _make_lightweight_unified_pipeline(query_decomposer=decomposer)

    def spy_encode(texts):
        encoded_texts.extend(texts)
        vectors = np.zeros((len(texts), 768), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors

    pipeline.encode_texts = spy_encode
    out = pipeline.search_with_metadata(snapshot_id="s1", query=query, top_k=2, mode="accurate")
    assert "man holding a sign" in encoded_texts
    assert out["diagnostics"]["query_plan_mode"] == "llm_compact"


def test_unified_detector_unknown_vs_verified_empty_and_consensus_toggle():
    from videntra.relations.entities import compute_object_verification_boost
    from videntra.retrieval.visual import RankedMoment

    reqs = [{"class_name": "laptop", "canonical_class": "laptop", "cardinality": 1, "count": 1}]
    assert compute_object_verification_boost(reqs, [], detector_ran=False) == 0.0
    assert compute_object_verification_boost(reqs, [], detector_ran=True) == pytest.approx(-0.10, abs=1e-6)

    p_off = UnifiedMultimodalSearchPipeline(
        lazy_load=True,
        use_disk_cache=False,
        enable_clip_consensus_boost=False,
    )
    emb_768 = np.zeros((10, 768), dtype=np.float32)
    emb_768[2:4, 0] = 0.90
    emb_768[2:4, 1] = 0.43589
    emb_768[0, 0] = 0.85
    emb_768[0, 2] = 0.52678
    emb_768[1, 0] = 0.70
    emb_768[1, 2] = 0.71414
    for fid in range(4, 10):
        emb_768[fid, 0] = 0.20
        emb_768[fid, 3] = 0.97980
    p_off.all_embeddings = emb_768
    p_off.augmented_embeddings = np.pad(emb_768, ((0, 0), (0, 96)))
    p_off.video_keyframes = {
        "v_a5": [{"faiss_id": 0, "pts_time": 1.0}, {"faiss_id": 1, "pts_time": 2.0}],
        "v_win": [{"faiss_id": 2, "pts_time": 1.0}, {"faiss_id": 3, "pts_time": 2.0}],
        "v_2": [{"faiss_id": 4, "pts_time": 1.0}, {"faiss_id": 5, "pts_time": 2.0}],
        "v_3": [{"faiss_id": 6, "pts_time": 1.0}, {"faiss_id": 7, "pts_time": 2.0}],
        "v_4": [{"faiss_id": 8, "pts_time": 1.0}, {"faiss_id": 9, "pts_time": 2.0}],
    }
    p_off.video_kf_fids = {vid: [k["faiss_id"] for k in kfs] for vid, kfs in p_off.video_keyframes.items()}
    p_off.keyframe_map = {
        k["faiss_id"]: {"video_id": vid, "pts_time": k["pts_time"], "source_frame_index": i}
        for vid, kfs in p_off.video_keyframes.items()
        for i, k in enumerate(kfs)
    }

    p_off.encode_texts = lambda texts: np.tile(np.eye(1, 768, dtype=np.float32), (len(texts), 1))
    import faiss

    aug_idx = faiss.IndexFlatIP(864)
    aug_idx.add(p_off.augmented_embeddings.astype(np.float32))
    p_off.aug_index = aug_idx
    p_off.index = faiss.IndexFlatIP(768)
    p_off.index.add(emb_768)
    res_off = p_off.search_with_metadata(snapshot_id="s1", query="person playing guitar", top_k=3, ablation_config="A5")
    assert res_off["diagnostics"]["feature_flags"]["enable_clip_consensus_boost"] is False
    assert res_off["diagnostics"]["clip_consensus_stats"]["applied"] is False
    assert res_off["diagnostics"]["clip_consensus_stats"]["reason"] == "disabled_by_flag"
    assert "ltr_stats" in res_off["diagnostics"]
    assert "detector_evidence_status" in res_off["diagnostics"]


def _build_abc_blended_pipeline(*, stage2_selection_mode: str, search_depth: int = 1):
    from videntra.indexing.augmented_vector import build_query_864d_vector

    p = UnifiedMultimodalSearchPipeline(
        lazy_load=True,
        use_disk_cache=False,
        search_depth=search_depth,
        stage2_selection_mode=stage2_selection_mode,
        enable_spatial_patches=False,
        enable_vlm_caption_lexical=False,
        enable_vlm_caption_dense=False,
        enable_clip_consensus_boost=False,
        ltr_blend_weight=0.0,
    )
    e0 = np.zeros((768,), dtype=np.float32)
    e0[0] = 1.0
    p.encode_texts = lambda texts: np.tile(e0, (len(texts), 1))
    q_864 = build_query_864d_vector(e0, "quiet landscape", {})
    u_864 = q_864 / (float(np.linalg.norm(q_864)) + 1e-12)

    q_768_orth = np.zeros((768,), dtype=np.float32)
    q_768_orth[500] = 1.0
    q_864_orth = np.zeros((864,), dtype=np.float32)
    q_864_orth[500] = 1.0

    specs = [
        ("vA", 0.980, 0.600),
        ("vB", 0.800, 0.960),
        ("vC", 0.910, 0.890),
    ]
    emb_768 = np.zeros((3, 768), dtype=np.float32)
    emb_864 = np.zeros((3, 864), dtype=np.float32)
    for idx, (_vid, s768, s864) in enumerate(specs):
        emb_768[idx] = s768 * e0 + np.sqrt(1.0 - s768**2) * q_768_orth
        emb_864[idx] = s864 * u_864 + np.sqrt(1.0 - s864**2) * q_864_orth

    p.all_embeddings = emb_768
    p.augmented_embeddings = emb_864
    p.index = faiss.IndexFlatIP(768)
    p.index.add(emb_768)
    p.aug_index = faiss.IndexFlatIP(864)
    p.aug_index.add(emb_864)
    p.keyframe_map = {
        idx: {"video_id": vid, "pts_time": 1.0, "source_frame_index": 0}
        for idx, (vid, _, _) in enumerate(specs)
    }
    p.video_keyframes = {
        vid: [{"faiss_id": idx, "pts_time": 1.0}]
        for idx, (vid, _, _) in enumerate(specs)
    }
    p.video_durations = {vid: 5.0 for vid, _, _ in specs}
    return p


def test_p01_stage2_exact_blended_selection_and_edge_cases():
    # 1. Exact blend rescues C=0.887 over legacy K=1 union {A=0.886, B=0.840}
    p_legacy = _build_abc_blended_pipeline(stage2_selection_mode="legacy_union", search_depth=1)
    res_legacy = p_legacy.search_with_metadata(snapshot_id="s1", query="quiet landscape", top_k=3)
    assert res_legacy["moments"][0].video_id == "vA"
    assert res_legacy["diagnostics"]["stage2_retrieval_path"] == "legacy_union_numpy"

    p_exact = _build_abc_blended_pipeline(stage2_selection_mode="exact", search_depth=1)
    res_exact = p_exact.search_with_metadata(snapshot_id="s1", query="quiet landscape", top_k=3)
    assert res_exact["moments"][0].video_id == "vC"
    assert res_exact["diagnostics"]["stage2_retrieval_path"] == "exact_blended_numpy"

    # 2. Cutoff tie-breaking (-score, fid) & filter excluding Top-1 & < K valid frames
    res_filtered = p_exact.search_with_metadata(
        snapshot_id="s1",
        query="quiet landscape",
        top_k=3,
        filters={"video_ids": ["vA", "vB"]},
    )
    assert [m.video_id for m in res_filtered["moments"]] == ["vA"]  # search_depth=1 yields valid top-1 vA when vC filtered out!

    p_tie = _build_abc_blended_pipeline(stage2_selection_mode="exact", search_depth=1)
    p_tie.all_embeddings[1] = p_tie.all_embeddings[2]
    p_tie.augmented_embeddings[1] = p_tie.augmented_embeddings[2]
    valid_mask = p_tie._build_stage2_valid_mask(3, None)
    fids_tie, _ = p_tie._select_topk_valid_frames(np.array([0.80, 0.90, 0.90]), valid_mask, 1)
    assert fids_tie == [1]  # smaller fid wins tie
    fids_lt_k, _ = p_tie._select_topk_valid_frames(np.array([0.80, 0.90, 0.90]), np.array([False, True, False]), 5)
    assert fids_lt_k == [1]  # < K valid frames does not raise

    # 3. Missing 768d fallback & multi-clause exact parity
    p_fb = _build_abc_blended_pipeline(stage2_selection_mode="exact", search_depth=2)
    p_fb.all_embeddings = np.zeros((0, 768), dtype=np.float32)
    res_fb = p_fb.search_with_metadata(snapshot_id="s1", query="quiet landscape", top_k=3)
    assert res_fb["diagnostics"]["stage2_retrieval_path"] == "faiss_fallback"
    assert len(res_fb["moments"]) >= 1

    p_mc = _build_abc_blended_pipeline(stage2_selection_mode="exact", search_depth=3)
    p_mc.video_keyframes["vC"].append({"faiss_id": 2, "pts_time": 2.0})
    res_mc = p_mc.search_with_metadata(
        snapshot_id="s1",
        query="first a dog runs in the park, then a cat jumps over the fence",
        top_k=3,
        ablation_config="A5",
    )
    assert res_mc["diagnostics"]["stage2_retrieval_path"] == "multiclause_exact_numpy"


def test_p02_video_proposal_modes_and_clip_reserve():
    # Video vA has two high frames 1.5s apart; vB has only one spike at 0.92 and low 0.20 elsewhere
    p = UnifiedMultimodalSearchPipeline(
        lazy_load=True,
        use_disk_cache=False,
        search_depth=5,
        video_proposal_mode="video_mean_top2_pts",
        video_proposal_top_n=2,
        video_proposal_anchors_per_video=2,
        video_proposal_min_pts_gap=1.0,
        enable_spatial_patches=False,
        enable_vlm_caption_lexical=False,
        enable_vlm_caption_dense=False,
        enable_clip_consensus_boost=False,
        ltr_blend_weight=0.0,
    )
    e0 = np.zeros((768,), dtype=np.float32)
    e0[0] = 1.0
    p.encode_texts = lambda texts: np.tile(e0, (len(texts), 1))
    from videntra.indexing.augmented_vector import build_query_864d_vector

    q864 = build_query_864d_vector(e0, "quiet scene", {})
    o768 = np.zeros((768,), dtype=np.float32)
    o768[1] = 1.0
    o864 = np.zeros((864,), dtype=np.float32)
    o864[1] = 1.0

    # 5 frames: f0,f1 in vA (scores 0.89, 0.88 at 1.0s & 2.5s), f2,f3 in vB (0.92, 0.20 at 1.0s & 3.0s), f4 in vC (0.50)
    frame_scores = [0.89, 0.88, 0.92, 0.20, 0.50]
    emb_768 = np.vstack([s * e0 + np.sqrt(1.0 - s**2) * o768 for s in frame_scores]).astype(np.float32)
    emb_864 = np.vstack([s * q864 + np.sqrt(1.0 - s**2) * o864 for s in frame_scores]).astype(np.float32)
    p.all_embeddings = emb_768
    p.augmented_embeddings = emb_864
    p.index = faiss.IndexFlatIP(768)
    p.index.add(emb_768)
    p.aug_index = faiss.IndexFlatIP(864)
    p.aug_index.add(emb_864)
    p.keyframe_map = {
        0: {"video_id": "vA", "pts_time": 1.0, "source_frame_index": 0},
        1: {"video_id": "vA", "pts_time": 2.5, "source_frame_index": 1},
        2: {"video_id": "vB", "pts_time": 1.0, "source_frame_index": 0},
        3: {"video_id": "vB", "pts_time": 3.0, "source_frame_index": 1},
        4: {"video_id": "vC", "pts_time": 1.0, "source_frame_index": 0},
    }
    p.video_keyframes = {
        "vA": [{"faiss_id": 0, "pts_time": 1.0}, {"faiss_id": 1, "pts_time": 2.5}],
        "vB": [{"faiss_id": 2, "pts_time": 1.0}, {"faiss_id": 3, "pts_time": 3.0}],
        "vC": [{"faiss_id": 4, "pts_time": 1.0}],
    }
    p.video_kf_fids = {vid: [k["faiss_id"] for k in kfs] for vid, kfs in p.video_keyframes.items()}
    p.video_durations = {"vA": 5.0, "vB": 5.0, "vC": 5.0}

    res_top2 = p.search_with_metadata(snapshot_id="s1", query="quiet scene", top_k=3)
    vp_stats = res_top2["diagnostics"]["video_proposal_stats"]
    assert vp_stats["mode"] == "video_mean_top2_pts"
    assert vp_stats["selected_videos_count"] == 2
    assert vp_stats["top10_videos"][0] == "vA"  # mean(0.89,0.88)=0.885 > mean(0.92,0.20)=0.56

    for alt_mode in ("video_max", "video_rrf_max_top2"):
        p_alt = _build_abc_blended_pipeline(stage2_selection_mode="exact", search_depth=3)
        p_alt.video_proposal_mode = alt_mode
        p_alt.video_proposal_clip_reserve_ratio = 0.25 if alt_mode == "video_max" else 0.0
        res_alt = p_alt.search_with_metadata(snapshot_id="s1", query="quiet landscape", top_k=3)
        assert res_alt["diagnostics"]["video_proposal_stats"]["mode"] == alt_mode


def test_p03_p04_caption_provenance_pure_visual_flag_traces_and_hubness():
    from videntra.retrieval.visual import apply_querybank_norm, compute_hubness_report

    # 1. Caption-only provenance never conflated into visual branch + disable_pure_visual_lexical_bonus + stage_traces
    p = UnifiedMultimodalSearchPipeline(
        lazy_load=True,
        use_disk_cache=False,
        search_depth=1,  # Only v0 fits in Stage 2 visual top-1; v1 enters via caption/OCR!
        disable_pure_visual_lexical_bonus=True,
        enable_spatial_patches=False,
        enable_vlm_caption_lexical=True,
        enable_vlm_caption_dense=False,
        enable_clip_consensus_boost=False,
        ltr_blend_weight=0.0,
    )
    e0 = np.zeros((768,), dtype=np.float32)
    e0[0] = 1.0
    p.encode_texts = lambda texts: np.tile(e0, (len(texts), 1))
    emb_768 = np.array([[0.9, 0.43589] + [0.0] * 766, [0.8, 0.6] + [0.0] * 766], dtype=np.float32)
    p.all_embeddings = emb_768
    p.augmented_embeddings = np.pad(emb_768, ((0, 0), (0, 96)))
    p.index = faiss.IndexFlatIP(768)
    p.index.add(emb_768)
    p.aug_index = faiss.IndexFlatIP(864)
    p.aug_index.add(p.augmented_embeddings)
    p.keyframe_map = {
        0: {"video_id": "v0", "pts_time": 1.0, "source_frame_index": 0},
        1: {"video_id": "v1", "pts_time": 1.0, "source_frame_index": 0},
    }
    p.video_keyframes = {
        "v0": [{"faiss_id": 0, "pts_time": 1.0}],
        "v1": [{"faiss_id": 1, "pts_time": 1.0}],
    }
    p.video_durations = {"v0": 5.0, "v1": 5.0}
    p.db_conn = sqlite3.connect(":memory:")
    p.db_conn.execute("CREATE VIRTUAL TABLE fts_ocr USING fts5(video_id, timestamp_ms, normalized_text)")
    p.db_conn.execute("CREATE VIRTUAL TABLE fts_asr USING fts5(video_id, start_ms, end_ms, normalized_text)")
    p.db_conn.execute("CREATE VIRTUAL TABLE fts_vlm_caption USING fts5(video_id, faiss_id UNINDEXED, timestamp_ms UNINDEXED, normalized_text)")
    p.has_vlm_caption_fts = True
    p.db_conn.execute("INSERT INTO fts_asr VALUES ('v1', 0, 2000, 'dog playing ball')")
    p.db_conn.execute("INSERT INTO fts_vlm_caption VALUES ('v1', 1, 1000, 'dog playing ball')")
    try:
        res = p.search_with_metadata(snapshot_id="s1", query="dog playing ball", top_k=2, ablation_config="A5")
        diag = res["diagnostics"]
        assert diag["modality_intent"] == "pure_visual"
        traces = diag["stage_traces"]
        assert set(traces.keys()) == {"before_base_fusion", "before_rerank", "after_rerank"}
        # v1 entered only via caption/ASR, NOT Stage 2 visual: its ltr features have rrf_vis == 0 and "visual" not in branches!
        v1_ltr = diag["candidate_ltr_features"]["v1"]
        assert v1_ltr["rrf_vis"] == 0.0
        assert v1_ltr["rrf_vis_caption"] > 0.0
        v1_trace = next(r for r in traces["before_rerank"] if r["video_id"] == "v1")
        assert "visual" not in v1_trace["branches"]
        # With disable_pure_visual_lexical_bonus=True, v0 visual winner must stay Top-1 despite v1 ASR overlap
        assert res["moments"][0].video_id == "v0"
    finally:
        p.db_conn.close()

    # 2. Hubness report & QB-Norm
    preds = [
        {"query_id": "q1", "ranked_video_ids": ["hub_vid", "v2"], "target_videos": ["v2"]},
        {"query_id": "q2", "ranked_video_ids": ["hub_vid", "v3"], "target_videos": ["hub_vid"]},
        {"query_id": "q3", "ranked_video_ids": ["v3", "hub_vid"], "target_videos": ["v3"]},
    ]
    hub = compute_hubness_report(preds, top_k=2)
    assert hub["num_queries"] == 3
    assert hub["unique_top1_videos"] == 2
    assert hub["top1_hubs"][0]["video_id"] == "hub_vid"
    assert hub["top1_hubs"][0]["top1_wrong_wins"] == 1

    target_sims = np.array([[0.9, 0.8]], dtype=np.float32)
    qb_sims = np.array([[0.95, 0.2], [0.95, 0.2]], dtype=np.float32)
    normed = apply_querybank_norm(target_sims, qb_sims, temperature=20.0)
    assert normed[0, 1] > normed[0, 0]  # gallery hub item 0 strongly downweighted


def test_sec55_two_staging_snapshots_distinct_hashes_and_channel_status(tmp_path: Path, monkeypatch):
    import scripts.run_validation_benchmark as rvb
    from videntra.storage.database import OfflineMetadataStore

    def _make_snapshot(snap_dir: Path, *, with_caption_sidecar: bool) -> tuple[Path, Path]:
        snap_dir.mkdir(parents=True, exist_ok=True)
        db_p = snap_dir / "metadata.db"
        store = OfflineMetadataStore(db_p)
        store.register_video({
            "video_id": "v1",
            "dataset": "msrvtt",
            "file_path": "v1.mp4",
            "duration_ms": 5000,
            "fps": 1.0,
            "time_base_num": 1,
            "time_base_den": 1000,
            "width": 64,
            "height": 64,
            "total_frames": 5,
            "has_audio": 0,
            "status": "completed",
        })
        store.insert_keyframes([{
            "keyframe_id": "v1_0",
            "video_id": "v1",
            "keyframe_index": 0,
            "source_frame_index": 0,
            "pts": 1000,
            "time_base": "1/1000",
            "timestamp_ms": 1000,
            "keyframe_path": "",
            "thumbnail_path": "",
            "faiss_id": 0,
            "sampling_reason": "uniform",
        }])
        store.close() if hasattr(store, "close") else None

        idx_p = snap_dir / "faiss_index.bin"
        idx = faiss.IndexFlatIP(864)
        vec = np.zeros((1, 864), dtype=np.float32)
        vec[0, 0] = 1.0
        idx.add(vec)
        faiss.write_index(idx, str(idx_p))

        if with_caption_sidecar:
            c_idx = faiss.IndexFlatIP(384)
            c_vec = np.zeros((1, 384), dtype=np.float32)
            c_vec[0, 0] = 1.0
            c_idx.add(c_vec)
            cap_bin = snap_dir / "faiss_e5_caption_384.bin"
            faiss.write_index(c_idx, str(cap_bin))
            cap_man = snap_dir / "e5_caption_manifest.json"
            cap_man.write_text(json.dumps([{"faiss_id": 0, "video_id": "v1", "timestamp_ms": 1000, "caption_text": "test"}]), encoding="utf-8")
            (snap_dir / "p85_receipt.json").write_text(
                json.dumps({
                    "p85_coverage_mode": "subset",
                    "e5_caption_index_sha256": __import__("hashlib").sha256(cap_bin.read_bytes()).hexdigest(),
                    "e5_caption_manifest_sha256": __import__("hashlib").sha256(cap_man.read_bytes()).hexdigest(),
                }),
                encoding="utf-8",
            )
        return db_p, idx_p

    db1, faiss1 = _make_snapshot(tmp_path / "snap1", with_caption_sidecar=False)
    db2, faiss2 = _make_snapshot(tmp_path / "snap2", with_caption_sidecar=True)

    # Check SSOT loads caption artifact hash + p85_coverage_mode from same dir as db_path
    class _StubRunner:
        def __init__(self, db_path: Path, faiss_path: Path, **_kw):
            self.db_path = db_path
            self.faiss_path = faiss_path
            self.device = "cpu"
            self.indexed_videos = {"v1"}
            self.dataset_videos = {"msrvtt": {"v1"}, "avc": set(), "qvhighlights": set()}
            has_cap = (db_path.parent / "faiss_e5_caption_384.bin").is_file()
            self.channel_status = {
                "e5_caption_index": {"status": "ready" if has_cap else "missing_artifact", "reason": "test"}
            }
            self.pipeline = SimpleNamespace(
                ltr_ranker=None,
                load_optional_status=has_cap,
                close=lambda: None,
            )

        def search_multimodal(self, query, top_k=10, ablation_config="A5"):
            return [{"video_id": "v1", "start_sec": 0.0, "end_sec": 2.0, "score": 0.9, "modality_attribution": "visual"}]

        @property
        def timings(self):
            return {"t_total_ms": 1.0, "t_non_neural_ms": 1.0, "t_faiss_ms": 0.5, "t_fts5_ms": 0.1, "t_fusion_ms": 0.2, "t_boundary_ms": 0.2}

        @property
        def diagnostics(self):
            return {"degraded": self.channel_status, "all_candidate_moments": []}

    val_dir = tmp_path / "val"
    val_dir.mkdir()
    (val_dir / "dev_150.jsonl").write_text(
        json.dumps({"query_id": "q1", "query_text": "test", "dataset": "msrvtt", "relevant_videos": ["v1"]}) + "\n",
        encoding="utf-8",
    )
    test_dir = tmp_path / "test_q"
    test_dir.mkdir()

    monkeypatch.setattr(rvb, "MultimodalBenchmarkRunner", _StubRunner)
    monkeypatch.setattr(rvb, "load_validation_queries", lambda *_a, **_kw: [{"query_id": "q1", "query_text": "test", "dataset": "msrvtt", "target_videos": ["v1"], "target_windows": []}])

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    exp_root = Path("work") / "experiments"
    before_snaps = set(exp_root.glob("*_all_splits_*")) if exp_root.is_dir() else set()
    try:
        rvb.run_benchmark(val_dir=val_dir, test_dir=test_dir, out_dir=out1, db_path=db1, faiss_path=faiss1, model_dir=model_dir, splits="dev_150", ltr_model_path=None)
        rvb.run_benchmark(val_dir=val_dir, test_dir=test_dir, out_dir=out2, db_path=db2, faiss_path=faiss2, model_dir=model_dir, splits="dev_150", ltr_model_path=None)
    finally:
        if exp_root.is_dir():
            import shutil as _shutil
            for new_snap in set(exp_root.glob("*_all_splits_*")) - before_snaps:
                _shutil.rmtree(new_snap, ignore_errors=True)

    sum1 = json.loads((out1 / "benchmark_summary.json").read_text(encoding="utf-8"))
    sum2 = json.loads((out2 / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert sum1["channel_status"]["e5_caption_index"]["status"] == "missing_artifact"
    assert sum2["channel_status"]["e5_caption_index"]["status"] == "ready"
    assert any("p85_receipt.json" in k and v is not None for k, v in sum2["snapshot_artifact_hashes"].items())
    assert all(v is None for k, v in sum1["snapshot_artifact_hashes"].items() if "p85_receipt.json" in k)
    assert sum1["snapshot_artifact_hashes"] != sum2["snapshot_artifact_hashes"]

    # Also test real UnifiedMultimodalSearchPipeline p85_receipt mismatch check
    p_real = UnifiedMultimodalSearchPipeline(lazy_load=True, use_disk_cache=False, db_path=db2, faiss_path=faiss2)
    p_real.load_optional_phase74_artifacts(db2.parent)
    assert p_real.p85_coverage_mode == "subset"
    assert "e5_caption_index" not in p_real._channel_mismatches

    (db2.parent / "e5_caption_manifest.json").write_text("[]", encoding="utf-8")
    p_bad = UnifiedMultimodalSearchPipeline(lazy_load=True, use_disk_cache=False, db_path=db2, faiss_path=faiss2)
    p_bad.load_optional_phase74_artifacts(db2.parent)
    assert "e5_caption_index" in p_bad._channel_mismatches





