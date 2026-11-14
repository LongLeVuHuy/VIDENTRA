"""Unit and regression tests for UnifiedMultimodalSearchPipeline (SSOT) and fixes B1-B4, B7, B9, B10, C1, C2."""
from collections import defaultdict
import itertools
from pathlib import Path
import random
import sqlite3
import unicodedata
import numpy as np
import pytest

from videntra.retrieval.unified_pipeline import (
    UnifiedMultimodalSearchPipeline,
    TextEmbeddingDiskCache,
    compute_order_invariant_rrf,
    analyze_query_lexical_tokens,
)
from videntra.localization.refine import expand_boundary_contrast
from videntra.retrieval.visual import RankedMoment


class TestB3OrderInvariantRRF:
    """B3: Verification that RRF fusion is completely independent of candidate arrival order."""

    def test_order_invariance_permutations(self):
        """All permutations of the same RRF values must yield identical merged scores."""
        rrf_values = [0.01639, 0.01587, 0.01428, 0.01250]
        expected = rrf_values[0] + 0.35 * (rrf_values[1] + rrf_values[2])

        for perm in itertools.permutations(rrf_values):
            val = compute_order_invariant_rrf(list(perm))
            assert abs(val - expected) < 1e-7

    def test_order_invariance_random_shuffles(self):
        """Random shuffles of 6 candidates must yield identical results capped at top 3."""
        base_rrfs = [1.0 / (60 + r) for r in [1, 5, 8, 12, 19, 25]]
        canonical_val = compute_order_invariant_rrf(base_rrfs)

        rng = random.Random(42)
        shuffled = list(base_rrfs)
        for _ in range(20):
            rng.shuffle(shuffled)
            val = compute_order_invariant_rrf(shuffled)
            assert abs(val - canonical_val) < 1e-9

    def test_edge_cases_empty_and_single(self):
        assert compute_order_invariant_rrf([]) == 0.0
        assert compute_order_invariant_rrf([0.05]) == 0.05
        assert abs(compute_order_invariant_rrf([0.10, 0.04]) - (0.10 + 0.35 * 0.04)) < 1e-7


class TestC1DiskCacheAndPadding:
    """C1: Deterministic disk cache and dynamic token padding."""

    def test_cache_save_and_reload(self, tmp_path):
        cache_file = tmp_path / "cache" / "query_text_embeddings_768d.npz"
        cache = TextEmbeddingDiskCache(cache_file)
        assert len(cache) == 0

        vec1 = np.ones(768, dtype=np.float32) * 0.5
        cache.set("a man walking a dog", vec1)
        cache.save()

        assert cache_file.exists()

        # Reload from fresh instance
        reloaded = TextEmbeddingDiskCache(cache_file)
        assert len(reloaded) == 1
        retrieved = reloaded.get("a man walking a dog")
        assert retrieved is not None
        assert retrieved.shape == (768,)
        np.testing.assert_allclose(retrieved, vec1, atol=1e-6)

    def test_cache_nfc_normalization(self, tmp_path):
        """Verifies that NFC and NFD strings map to identical cache keys."""
        cache_file = tmp_path / "nfc_cache.npz"
        cache = TextEmbeddingDiskCache(cache_file)

        # Vietnamese text with diacritics
        text_precomposed = "người đi bộ trên đường"
        text_decomposed = unicodedata.normalize("NFD", text_precomposed)
        assert text_precomposed != text_decomposed  # Binary representations differ

        vec = np.random.randn(768).astype(np.float32)
        cache.set(text_precomposed, vec)

        # Retrieval using NFD string must hit the same key
        retrieved = cache.get(text_decomposed)
        assert retrieved is not None
        np.testing.assert_allclose(retrieved, vec, atol=1e-6)


class TestB10AnchorAwareBoundaryExpansion:
    """B10: Boundary expansion with anchor_idx and anchor_window."""

    def test_anchor_idx_priority_when_above_threshold(self):
        """When anchor_idx has score >= 0.65 * s_max, it must be chosen as peak_idx."""
        ts = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
        # Global argmax is at index 1 (5.0s, score 0.90)
        # Sequential alignment anchor is at index 5 (25.0s, score 0.70 >= 0.65 * 0.90 = 0.585)
        curve = [0.1, 0.90, 0.2, 0.1, 0.2, 0.70, 0.1]

        # Without anchor: expands around index 1 (5.0s)
        st_no_anchor, et_no_anchor = expand_boundary_contrast(ts, curve, v_dur=35.0)
        assert st_no_anchor <= 5.0 <= et_no_anchor
        assert et_no_anchor < 15.0

        # With anchor_idx=5: must expand around 25.0s
        st_anchored, et_anchored = expand_boundary_contrast(ts, curve, v_dur=35.0, anchor_idx=5)
        assert st_anchored <= 25.0 <= et_anchored
        assert st_anchored >= 15.0

    def test_anchor_idx_fallback_when_below_threshold(self):
        """When anchor_idx has score < 0.65 * s_max, it falls back to global argmax."""
        ts = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
        # Global argmax at index 1 (5.0s, score 0.95)
        # Anchor at index 4 (20.0s, score 0.20 < 0.65 * 0.95 = 0.6175)
        curve = [0.1, 0.95, 0.2, 0.1, 0.20, 0.05]

        st, et = expand_boundary_contrast(ts, curve, v_dur=30.0, anchor_idx=4)
        # Must fall back to global argmax at 5.0s
        assert st <= 5.0 <= et

    def test_anchor_window_initialization(self):
        """anchor_window initializes coverage window before hysteresis expansion."""
        ts = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0]
        curve = [0.2, 0.5, 0.6, 0.65, 0.6, 0.5, 0.2]

        anchor_win = (6.0, 14.0)
        st, et = expand_boundary_contrast(ts, curve, v_dur=30.0, anchor_window=anchor_win)
        # Boundary must fully envelop the anchor window
        assert st <= 6.0
        assert et >= 14.0


class TestB7SpecificitySortedTokens:
    """B7: Token sorting and ceiling at 14 tokens."""

    def test_specificity_token_ordering(self):
        pipeline = UnifiedMultimodalSearchPipeline(lazy_load=True, use_disk_cache=False)
        query = 'a presenter shows "Quarterly Financial Results" on widescreen display during Mike Pence press conference'
        fts_query, quoted, high, low, proper = pipeline.prepare_fts5_query_and_tokens(query)

        # Quoted phrase must be captured
        assert len(quoted) >= 1
        # Proper noun Pence must be recognized
        assert "pence" in proper

        # FTS5 query tokens should not exceed 14 extra tokens
        tokens = [t.strip() for t in fts_query.split("OR")]
        assert len(tokens) <= 15  # base query + at most 14 tokens


class TestB1TemporalLocalVisualLookup:
    """B1: Localized visual lookup."""

    def test_local_visual_lookup_behavior(self):
        pipeline = UnifiedMultimodalSearchPipeline(lazy_load=True, use_disk_cache=False)

        # Mock video keyframes: t = 2.0s and t = 100.0s
        vid = "test_vid_1"
        pipeline.video_keyframes[vid] = [
            {"faiss_id": 0, "pts_time": 2.0},
            {"faiss_id": 1, "pts_time": 100.0},
        ]
        # Frame 0 (t=2.0s) has high similarity 0.85
        # Frame 1 (t=100.0s) has low similarity 0.15
        dim = 864
        pipeline.augmented_embeddings = np.zeros((2, dim), dtype=np.float32)
        q_vec = np.zeros(dim, dtype=np.float32)
        q_vec[0] = 1.0

        pipeline.augmented_embeddings[0, 0] = 0.85
        pipeline.augmented_embeddings[1, 0] = 0.15

        # 1. Query near 2.0s -> gets 0.85
        sim_near_2 = pipeline.lookup_true_visual_sim(vid, ts=2.5, primary_q_864=q_vec.reshape(1, -1))
        assert abs(sim_near_2 - 0.85) < 1e-5

        # 2. Query near 100.0s -> gets 0.15 (does NOT borrow 0.85 from 2.0s!)
        sim_near_100 = pipeline.lookup_true_visual_sim(vid, ts=100.5, primary_q_864=q_vec.reshape(1, -1))
        assert abs(sim_near_100 - 0.15) < 1e-5

        # 3. Query far from both (e.g. ts=50.0s, > 6.0s away) -> gets 35th percentile baseline
        sim_far = pipeline.lookup_true_visual_sim(vid, ts=50.0, primary_q_864=q_vec.reshape(1, -1))
        expected_p35 = float(np.percentile([0.85, 0.15], 35))
        assert abs(sim_far - expected_p35) < 1e-5
        assert sim_far < 0.50  # Gate remains closed


class TestB9PureASRFalsePositiveGate:
    """B9: Suppression of false positive boosts for pure-ASR clusters lacking visual anchors."""

    def test_pure_asr_suppression_and_visual_anchor_difference(self):
        """Simulate clusters through Stage 5 fusion logic in search()."""
        pipeline = UnifiedMultimodalSearchPipeline(lazy_load=True, use_disk_cache=False)
        def fake_encode(texts):
            vectors = np.zeros((len(texts), 768), dtype=np.float32)
            vectors[:, 0] = 1.0
            return vectors
        pipeline.encode_texts = fake_encode

        # Setup 2 dummy videos
        # Video A: Target video with strong visual support
        # Video B: Distractor video with pure-ASR match but NO visual keyframe match
        dim = 864
        pipeline.augmented_embeddings = np.zeros((4, dim), dtype=np.float32)
        pipeline.augmented_embeddings[0, 0] = 0.40  # Vid A kf 0
        pipeline.augmented_embeddings[1, 0] = 0.42  # Vid A kf 1
        pipeline.augmented_embeddings[2, 0] = 0.08  # Vid B kf 0 (low visual similarity)
        pipeline.augmented_embeddings[3, 0] = 0.05  # Vid B kf 1

        pipeline.keyframe_map = {
            0: {"video_id": "vid_A", "pts_time": 5.0, "source_frame_index": 0},
            1: {"video_id": "vid_A", "pts_time": 10.0, "source_frame_index": 10},
            2: {"video_id": "vid_B", "pts_time": 5.0, "source_frame_index": 0},
            3: {"video_id": "vid_B", "pts_time": 10.0, "source_frame_index": 10},
        }
        pipeline.video_keyframes = {
            "vid_A": [{"faiss_id": 0, "pts_time": 5.0}, {"faiss_id": 1, "pts_time": 10.0}],
            "vid_B": [{"faiss_id": 2, "pts_time": 5.0}, {"faiss_id": 3, "pts_time": 10.0}],
        }
        pipeline.video_durations = {"vid_A": 20.0, "vid_B": 20.0}

        import faiss
        pipeline.aug_index = faiss.IndexFlatIP(dim)
        pipeline.aug_index.add(pipeline.augmented_embeddings)
        pipeline.index = faiss.IndexFlatIP(768)
        pipeline.index.add(np.zeros((4, 768), dtype=np.float32))

        # Perform search with a generic query
        # Vid A has visual match; Vid B has no visual anchor
        results = pipeline.search("talking person in office", top_k=2)
        assert len(results) > 0
        # Vid A should lead Vid B due to visual anchoring
        assert results[0].video_id == "vid_A"


class TestFrameObjectsSchemaIntegration:
    """Integration test verifying FrameObjects schema compliance and avoiding pts_sec regressions."""

    def test_schema_with_timestamp_ms_loads_correctly(self, tmp_path):
        """Pipeline must read timestamp_ms from FrameObjects and accurately convert to pts_sec in seconds."""
        db_file = tmp_path / "test_metadata.db"
        conn = sqlite3.connect(str(db_file))
        cur = conn.cursor()

        # Create schema matching production metadata.db (FrameObjects has timestamp_ms, NOT pts_sec)
        cur.execute("""
            CREATE TABLE Videos (
                video_id TEXT PRIMARY KEY,
                duration_ms INTEGER
            );
        """)
        cur.execute("""
            CREATE TABLE KeyframeMap (
                faiss_id INTEGER PRIMARY KEY,
                video_id TEXT,
                pts_time REAL,
                source_frame_index INTEGER,
                keyframe_path TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE FrameObjects (
                detection_id TEXT PRIMARY KEY,
                video_id TEXT,
                keyframe_id TEXT,
                timestamp_ms INTEGER,
                class_id INTEGER,
                class_name TEXT,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                confidence REAL,
                image_width INTEGER,
                image_height INTEGER,
                model_id TEXT,
                model_version TEXT,
                created_at TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE OCRObservations (
                video_id TEXT,
                timestamp_ms INTEGER
            );
        """)
        cur.execute("""
            CREATE TABLE ASRSegments (
                video_id TEXT,
                start_ms INTEGER,
                end_ms INTEGER
            );
        """)

        # Insert sample records
        cur.execute("INSERT INTO Videos VALUES ('vid_test_1', 30000);")
        cur.execute("INSERT INTO KeyframeMap VALUES (0, 'vid_test_1', 5.0, 150, '/path/kf0.jpg');")
        cur.execute("""
            INSERT INTO FrameObjects (detection_id, video_id, timestamp_ms, class_name, confidence)
            VALUES ('det_1', 'vid_test_1', 5000, 'person', 0.95),
                   ('det_2', 'vid_test_1', 12500, 'bicycle', 0.88);
        """)
        conn.commit()
        conn.close()

        # Initialize pipeline pointing to this DB with lazy_load=False (non-mocked _initialize)
        pipeline = UnifiedMultimodalSearchPipeline(
            db_path=db_file,
            faiss_path=tmp_path / "nonexistent_faiss.bin",
            model_dir=tmp_path / "nonexistent_model",
            lazy_load=False,
            use_disk_cache=False,
        )

        assert "vid_test_1" in pipeline.video_objects
        objs = pipeline.video_objects["vid_test_1"]
        assert len(objs) == 2
        # Crucial check: pts_sec must be converted from ms to seconds
        assert abs(objs[0]["pts_sec"] - 5.0) < 1e-4
        assert objs[0]["class_name"] == "person"
        assert abs(objs[0]["confidence"] - 0.95) < 1e-4

        assert abs(objs[1]["pts_sec"] - 12.5) < 1e-4
        assert objs[1]["class_name"] == "bicycle"

        pipeline.close()

    def test_production_metadata_db_schema_compliance(self):
        """Verify that actual data/processed/metadata.db initializes FrameObjects without errors."""
        prod_db = Path("data/processed/metadata.db")
        if not prod_db.exists():
            pytest.skip("data/processed/metadata.db not present in environment")

        pipeline = UnifiedMultimodalSearchPipeline(
            db_path=prod_db,
            faiss_path=Path("nonexistent_faiss.bin"),
            model_dir=Path("nonexistent_model"),
            lazy_load=False,
            use_disk_cache=False,
        )

        assert len(pipeline.video_objects) > 0
        # Check first object has pts_sec in seconds (e.g. <= 3600.0s), not milliseconds
        first_vid = next(iter(pipeline.video_objects))
        first_obj = pipeline.video_objects[first_vid][0]
        assert "pts_sec" in first_obj
        assert first_obj["pts_sec"] < 10000.0  # Seconds, not milliseconds

        pipeline.close()

