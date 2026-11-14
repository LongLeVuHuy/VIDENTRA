"""Comprehensive unit tests for 864-dimensional multimodal representation builder.

Tests:
- Vector concatenation: (N, 768) + (N, 80) + (N, 16) -> (N, 864)
- L2 normalization across full 864d vectors (||v||_2 approx 1.0)
- COCO global IDF calculation and YOLOv8 80-class mapping
- 16d Context Subspace: person cardinality, sin/cos temporal position, OCR/ASR density
- Edge cases: empty frame_objects, duration is None, duration <= 0, all-zero raw_embeddings, empty N=0
- Query 864d vector formulation: entity extraction, cardinality, temporal priors, modality indicators
"""
import math
from types import SimpleNamespace
import numpy as np
import pytest

from videntra.indexing.augmented_vector import (
    COCO_80_CLASSES,
    COCO_CLASS_TO_IDX,
    compute_coco_global_idf,
    build_augmented_864d_embeddings,
    build_query_864d_vector,
    rebuild_augmented_864d_cache,
)


class TestAugmented864dVectorBuilder:
    """Tests for build_augmented_864d_embeddings and related components."""

    def test_vector_concatenation_shapes_and_types(self):
        """Verifies (N, 768) + (N, 80) + (N, 16) -> (N, 864) shape and float32 dtype."""
        n_keyframes = 4
        rng = np.random.default_rng(42)
        raw_emb = rng.standard_normal((n_keyframes, 768)).astype(np.float32)

        kf_map = {
            i: {"video_id": "vid_1", "pts_time": float(i * 3.0)}
            for i in range(n_keyframes)
        }
        frame_objs = [
            {"video_id": "vid_1", "pts_sec": 0.0, "class_name": "person", "confidence": 0.9},
            {"video_id": "vid_1", "pts_sec": 3.0, "class_name": "car", "confidence": 0.8},
        ]
        durations = {"vid_1": 60.0}

        aug_emb, mean_vec, idf_map = build_augmented_864d_embeddings(
            raw_embeddings=raw_emb,
            keyframe_map=kf_map,
            frame_objects=frame_objs,
            videos_duration=durations,
            alpha_obj=0.25,
            alpha_meta=0.15,
            beta_hub=0.08,
        )

        assert isinstance(aug_emb, np.ndarray)
        assert aug_emb.shape == (n_keyframes, 864)
        assert aug_emb.dtype == np.float32

        assert isinstance(mean_vec, np.ndarray)
        assert mean_vec.shape == (768,)
        assert mean_vec.dtype == np.float32

        assert isinstance(idf_map, dict)
        assert len(idf_map) == 80
        assert set(idf_map.keys()) == set(COCO_80_CLASSES)

    def test_l2_normalization_strictness(self):
        """Verifies that every row in the augmented 864d matrix has L2 norm approx 1.0."""
        n_keyframes = 8
        rng = np.random.default_rng(123)
        raw_emb = rng.standard_normal((n_keyframes, 768)).astype(np.float32)

        kf_map = {
            i: SimpleNamespace(video_id="v_norm", pts_time=float(i * 2.5))
            for i in range(n_keyframes)
        }
        frame_objs = [
            {"video_id": "v_norm", "pts_sec": 2.5, "class_name": "dog", "confidence": 0.95},
            {"video_id": "v_norm", "pts_sec": 5.0, "class_name": "person", "confidence": 0.85},
            {"video_id": "v_norm", "pts_sec": 5.0, "class_name": "cat", "confidence": 0.75},
        ]
        durations = {"v_norm": 30.0}
        ocr_data = {"v_norm": [2.0, 2.5, 3.0]}
        asr_data = {"v_norm": [(4.0, 7.0)]}

        aug_emb, mean_vec, _ = build_augmented_864d_embeddings(
            raw_embeddings=raw_emb,
            keyframe_map=kf_map,
            frame_objects=frame_objs,
            videos_duration=durations,
            ocr_data=ocr_data,
            asr_data=asr_data,
        )

        # Check all keyframe vectors have L2 norm within 1e-5 of 1.0
        norms = np.linalg.norm(aug_emb, axis=1)
        for i, norm_val in enumerate(norms):
            assert abs(float(norm_val) - 1.0) < 1e-5, f"Vector at index {i} has norm {norm_val} != 1.0"

        # Check mean vector is also L2-normalized
        assert abs(float(np.linalg.norm(mean_vec)) - 1.0) < 1e-5

    def test_compute_coco_global_idf_mathematics(self):
        """Verifies exact formula idf_c = ln(1 + N_videos / (1 + df(c)))."""
        objs_by_vid = {
            "v1": [{"class_name": "person"}, {"class_name": "car"}],
            "v2": [{"class_name": "person"}, {"class_name": "dog"}],
            "v3": [{"class_name": "person"}],
        }
        total_videos = 3
        idf_map = compute_coco_global_idf(objs_by_vid, total_videos)

        assert len(idf_map) == 80
        # df(person) = 3 -> idf = ln(1 + 3 / (1 + 3)) = ln(1 + 0.75) = ln(1.75)
        expected_person_idf = math.log(1.0 + 3.0 / (1.0 + 3.0))
        assert abs(idf_map["person"] - expected_person_idf) < 1e-6

        # df(car) = 1 -> idf = ln(1 + 3 / (1 + 1)) = ln(1 + 1.5) = ln(2.5)
        expected_car_idf = math.log(1.0 + 3.0 / (1.0 + 1.0))
        assert abs(idf_map["car"] - expected_car_idf) < 1e-6

        # df(airplane) = 0 -> idf = ln(1 + 3 / (1 + 0)) = ln(4.0)
        expected_airplane_idf = math.log(1.0 + 3.0 / (1.0 + 0.0))
        assert abs(idf_map["airplane"] - expected_airplane_idf) < 1e-6

        # Rare classes must have strictly higher IDF than common classes
        assert idf_map["person"] < idf_map["car"] < idf_map["airplane"]

    def test_yolov8_object_mapping_and_timestamp_ms_conversion(self):
        """Verifies mapping of YOLOv8 detections (including timestamp_ms) to 80 COCO indices."""
        raw_emb = np.ones((2, 768), dtype=np.float32)
        kf_map = {
            0: {"video_id": "v_obj", "pts_time": 10.0},
            1: {"video_id": "v_obj", "pts_time": 50.0},
        }
        # First detection uses timestamp_ms=10200 (10.2s), matching keyframe 0 (10.0s) within 3.5s
        # Second detection uses pts_sec=51.0, matching keyframe 1 (50.0s) within 3.5s
        frame_objs = [
            {"video_id": "v_obj", "timestamp_ms": 10200, "class_name": "bicycle", "confidence": 0.8},
            {"video_id": "v_obj", "pts_sec": 51.0, "class_name": "horse", "confidence": 0.9},
        ]
        durations = {"v_obj": 100.0}

        aug_emb, _, _ = build_augmented_864d_embeddings(
            raw_embeddings=raw_emb,
            keyframe_map=kf_map,
            frame_objects=frame_objs,
            videos_duration=durations,
            alpha_obj=0.25,
            alpha_meta=0.15,
        )

        # Slice 80d object subspace: indices 768..848
        obj_slice_0 = aug_emb[0, 768:848]
        obj_slice_1 = aug_emb[1, 768:848]

        bicycle_idx = COCO_CLASS_TO_IDX["bicycle"]
        horse_idx = COCO_CLASS_TO_IDX["horse"]

        # Keyframe 0 should have non-zero bicycle feature and zero horse feature
        assert obj_slice_0[bicycle_idx] > 0.0
        assert obj_slice_0[horse_idx] == 0.0

        # Keyframe 1 should have non-zero horse feature and zero bicycle feature
        assert obj_slice_1[horse_idx] > 0.0
        assert obj_slice_1[bicycle_idx] == 0.0

    def test_16d_context_subspace_person_cardinality(self):
        """Verifies Person Cardinality in dims 0..3:
        1 person -> dim 0
        2 people -> dim 1
        >=3 people -> dim 2
        0 people (or only conf < 0.40) -> dim 3
        """
        raw_emb = np.ones((4, 768), dtype=np.float32)
        kf_map = {
            0: {"video_id": "v_card", "pts_time": 10.0},  # 1 person
            1: {"video_id": "v_card", "pts_time": 20.0},  # 2 people
            2: {"video_id": "v_card", "pts_time": 30.0},  # 3 people
            3: {"video_id": "v_card", "pts_time": 40.0},  # low conf only -> counts as 0
        }
        frame_objs = [
            # kf 0 (10.0s): 1 person conf 0.85
            {"video_id": "v_card", "pts_sec": 10.0, "class_name": "person", "confidence": 0.85},
            # kf 1 (20.0s): 2 people conf 0.70 and 0.60
            {"video_id": "v_card", "pts_sec": 19.5, "class_name": "person", "confidence": 0.70},
            {"video_id": "v_card", "pts_sec": 20.5, "class_name": "person", "confidence": 0.60},
            # kf 2 (30.0s): 3 people conf 0.80
            {"video_id": "v_card", "pts_sec": 29.0, "class_name": "person", "confidence": 0.80},
            {"video_id": "v_card", "pts_sec": 30.0, "class_name": "person", "confidence": 0.85},
            {"video_id": "v_card", "pts_sec": 31.0, "class_name": "person", "confidence": 0.90},
            # kf 3 (40.0s): 1 person but conf 0.35 (< 0.40 cutoff)
            {"video_id": "v_card", "pts_sec": 40.0, "class_name": "person", "confidence": 0.35},
        ]
        durations = {"v_card": 100.0}

        aug_emb, _, _ = build_augmented_864d_embeddings(
            raw_embeddings=raw_emb,
            keyframe_map=kf_map,
            frame_objects=frame_objs,
            videos_duration=durations,
        )

        # Context subspace is at indices 848..864
        meta_0 = aug_emb[0, 848:864]
        meta_1 = aug_emb[1, 848:864]
        meta_2 = aug_emb[2, 848:864]
        meta_3 = aug_emb[3, 848:864]

        # Check dim 0..3 relative ordering
        assert meta_0[0] > 0.0 and meta_0[1] == 0.0 and meta_0[2] == 0.0 and meta_0[3] == 0.0
        assert meta_1[1] > 0.0 and meta_1[0] == 0.0 and meta_1[2] == 0.0 and meta_1[3] == 0.0
        assert meta_2[2] > 0.0 and meta_2[0] == 0.0 and meta_2[1] == 0.0 and meta_2[3] == 0.0
        assert meta_3[3] > 0.0 and meta_3[0] == 0.0 and meta_3[1] == 0.0 and meta_3[2] == 0.0

    def test_16d_context_subspace_temporal_position_encoding(self):
        """Verifies sin/cos relative position encoding in dims 4..7."""
        raw_emb = np.ones((2, 768), dtype=np.float32)
        # kf 0 at t=0.0 (p_rel = 0.0) -> sin=0, cos=1
        # kf 1 at t=25.0 in 100s video (p_rel = 0.25) -> sin(2pi*0.25)=sin(pi/2)=1, cos=0
        kf_map = {
            0: {"video_id": "v_pos", "pts_time": 0.0},
            1: {"video_id": "v_pos", "pts_time": 25.0},
        }
        durations = {"v_pos": 100.0}

        aug_emb, _, _ = build_augmented_864d_embeddings(
            raw_embeddings=raw_emb,
            keyframe_map=kf_map,
            frame_objects=[],
            videos_duration=durations,
        )

        # Indices 848..864: dims 4..7 correspond to indices 852..856
        # At p_rel = 0.0: sin(0) = 0.0, cos(0) > 0.0
        assert abs(aug_emb[0, 852]) < 1e-6  # sin(2*pi*1.0*0) = 0
        assert aug_emb[0, 853] > 0.0         # cos(2*pi*1.0*0) > 0
        assert abs(aug_emb[0, 854]) < 1e-6  # sin(2*pi*2.0*0) = 0
        assert aug_emb[0, 855] > 0.0         # cos(2*pi*2.0*0) > 0

        # At p_rel = 0.25:
        # sin(2*pi*1.0*0.25) = sin(pi/2) = 1.0 > 0
        # cos(2*pi*1.0*0.25) = cos(pi/2) = 0.0
        # sin(2*pi*2.0*0.25) = sin(pi) = 0.0
        # cos(2*pi*2.0*0.25) = cos(pi) = -1.0 < 0
        assert aug_emb[1, 852] > 0.0
        assert abs(aug_emb[1, 853]) < 1e-4
        assert abs(aug_emb[1, 854]) < 1e-4
        assert aug_emb[1, 855] < 0.0

    def test_16d_context_subspace_ocr_and_asr_density(self):
        """Verifies OCR density (dims 8..11) and ASR density (dims 12..15)."""
        raw_emb = np.ones((1, 768), dtype=np.float32)
        kf_map = {0: {"video_id": "v_dense", "pts_time": 10.0}}
        # 3 OCR timestamps within +/- 2.0s: [9.0, 10.0, 11.5]
        ocr_data = {"v_dense": [9.0, 10.0, 11.5, 30.0]}
        # ASR span [8.0, 12.0] covers 10.0s; overlap in [5.0, 15.0] is 4.0s
        asr_data = {"v_dense": [(8.0, 12.0)]}
        durations = {"v_dense": 60.0}

        aug_emb, _, _ = build_augmented_864d_embeddings(
            raw_embeddings=raw_emb,
            keyframe_map=kf_map,
            frame_objects=[],
            videos_duration=durations,
            ocr_data=ocr_data,
            asr_data=asr_data,
        )

        meta = aug_emb[0, 848:864]
        # Dim 8: OCR presence > 0
        assert meta[8] > 0.0
        # Dim 9: OCR token density > 0
        assert meta[9] > 0.0
        # Dim 10, 11: unused / zero
        assert meta[10] == 0.0 and meta[11] == 0.0

        # Dim 12: ASR presence > 0
        assert meta[12] > 0.0
        # Dim 13: ASR duration ratio > 0
        assert meta[13] > 0.0
        # Dim 14: B5 auxiliary null-object indicator when frame has no objects; Dim 15: unused / zero
        assert meta[14] > 0.0 and meta[15] == 0.0

    def test_edge_case_duration_is_none(self):
        """Verifies defensive handling when video duration in dictionary is None."""
        raw_emb = np.ones((1, 768), dtype=np.float32)
        kf_map = {0: {"video_id": "v_none_dur", "pts_time": 5.0}}
        durations = {"v_none_dur": None}

        # Must not raise TypeError: float() argument must be a string or a real number, not 'NoneType'
        aug_emb, _, _ = build_augmented_864d_embeddings(
            raw_embeddings=raw_emb,
            keyframe_map=kf_map,
            frame_objects=[],
            videos_duration=durations,
        )
        assert aug_emb.shape == (1, 864)
        assert abs(float(np.linalg.norm(aug_emb[0])) - 1.0) < 1e-5

    def test_edge_case_duration_zero_or_negative(self):
        """Verifies safe handling when video duration is 0.0 or negative."""
        raw_emb = np.ones((2, 768), dtype=np.float32)
        kf_map = {
            0: {"video_id": "v_zero", "pts_time": 5.0},
            1: {"video_id": "v_neg", "pts_time": 5.0},
        }
        durations = {"v_zero": 0.0, "v_neg": -10.0}

        aug_emb, _, _ = build_augmented_864d_embeddings(
            raw_embeddings=raw_emb,
            keyframe_map=kf_map,
            frame_objects=[],
            videos_duration=durations,
        )
        assert aug_emb.shape == (2, 864)
        assert not np.isnan(aug_emb).any()
        assert not np.isinf(aug_emb).any()

    def test_edge_case_all_zero_raw_embeddings(self):
        """Verifies stability when raw_embeddings is completely zero."""
        raw_emb = np.zeros((3, 768), dtype=np.float32)
        kf_map = {i: {"video_id": "v_zero_emb", "pts_time": float(i)} for i in range(3)}
        durations = {"v_zero_emb": 10.0}

        aug_emb, mean_vec, _ = build_augmented_864d_embeddings(
            raw_embeddings=raw_emb,
            keyframe_map=kf_map,
            frame_objects=[],
            videos_duration=durations,
        )
        assert aug_emb.shape == (3, 864)
        assert not np.isnan(aug_emb).any()
        assert not np.isnan(mean_vec).any()

    def test_edge_case_empty_raw_embeddings(self):
        """Verifies empty matrix N=0 behavior."""
        raw_emb = np.zeros((0, 768), dtype=np.float32)
        aug_emb, mean_vec, idf_map = build_augmented_864d_embeddings(
            raw_embeddings=raw_emb,
            keyframe_map={},
            frame_objects=[],
            videos_duration={},
        )
        assert aug_emb.shape == (0, 864)
        assert mean_vec.shape == (768,)
        assert len(idf_map) == 80

    def test_edge_case_missing_keyframe_info(self):
        """Verifies that missing entries in keyframe_map do not cause index errors."""
        raw_emb = np.ones((3, 768), dtype=np.float32)
        # Only index 1 is present in map
        kf_map = {1: {"video_id": "v_sparse", "pts_time": 10.0}}
        durations = {"v_sparse": 20.0}

        aug_emb, _, _ = build_augmented_864d_embeddings(
            raw_embeddings=raw_emb,
            keyframe_map=kf_map,
            frame_objects=[],
            videos_duration=durations,
        )
        assert aug_emb.shape == (3, 864)
        assert not np.isnan(aug_emb).any()


class TestBuildQuery864dVector:
    """Tests for build_query_864d_vector query formulation."""

    def test_build_query_864d_output_shape_and_norm(self):
        """Verifies shape (864,) and unit L2 norm of query vector."""
        sem_vec = np.ones(768, dtype=np.float32)
        coco_idf = {c: 1.0 for c in COCO_80_CLASSES}
        q_vec = build_query_864d_vector(
            sem_vec=sem_vec,
            query_text="a man is walking a dog in the park",
            coco_idf_map=coco_idf,
            alpha_obj=0.25,
            alpha_meta=0.15,
        )
        assert isinstance(q_vec, np.ndarray)
        assert q_vec.shape == (864,)
        assert abs(float(np.linalg.norm(q_vec)) - 1.0) < 1e-5

    def test_query_object_subspace_activation(self):
        """Verifies that extracted objects activate corresponding COCO classes in 80d subspace."""
        sem_vec = np.zeros(768, dtype=np.float32)
        coco_idf = {c: 1.5 for c in COCO_80_CLASSES}
        q_vec = build_query_864d_vector(
            sem_vec=sem_vec,
            query_text="a person holding an umbrella near a car",
            coco_idf_map=coco_idf,
        )
        obj_slice = q_vec[768:848]

        person_idx = COCO_CLASS_TO_IDX["person"]
        umbrella_idx = COCO_CLASS_TO_IDX["umbrella"]
        car_idx = COCO_CLASS_TO_IDX["car"]
        airplane_idx = COCO_CLASS_TO_IDX["airplane"]

        assert obj_slice[person_idx] > 0.0
        assert obj_slice[umbrella_idx] > 0.0
        assert obj_slice[car_idx] > 0.0
        assert obj_slice[airplane_idx] == 0.0

    def test_query_context_cardinality_and_temporal_priors(self):
        """Verifies cardinality and temporal prior detection in query text."""
        sem_vec = np.zeros(768, dtype=np.float32)
        coco_idf = {c: 1.0 for c in COCO_80_CLASSES}

        # Query 1: Two people at the beginning
        q1 = build_query_864d_vector(
            sem_vec=sem_vec,
            query_text="two people talking at the beginning of the clip",
            coco_idf_map=coco_idf,
        )
        meta1 = q1[848:864]
        # Dim 1: 2 people cardinality
        assert meta1[1] > 0.0 and meta1[0] == 0.0
        # Dim 4..7: p=0.1 encoded
        assert meta1[4] > 0.0  # sin(2*pi*0.1) > 0
        assert meta1[5] > 0.0  # cos(2*pi*0.1) > 0

        # Query 2: Nobody at the end
        q2 = build_query_864d_vector(
            sem_vec=sem_vec,
            query_text="nobody seen at the end of the video",
            coco_idf_map=coco_idf,
        )
        meta2 = q2[848:864]
        # Dim 3: nobody cardinality
        assert meta2[3] > 0.0
        # Dim 4..7: p=0.9 encoded
        assert meta2[4] < 0.0  # sin(2*pi*0.9) < 0
        assert meta2[5] > 0.0  # cos(2*pi*0.9) > 0

    def test_query_modality_text_and_speech_indicators(self):
        """Verifies OCR (dim 8) and ASR (dim 12) triggers from query text."""
        sem_vec = np.zeros(768, dtype=np.float32)
        coco_idf = {c: 1.0 for c in COCO_80_CLASSES}

        # Query with text quote and speech indicator
        q = build_query_864d_vector(
            sem_vec=sem_vec,
            query_text="a man speaking and showing 'Sale 50%' sign",
            coco_idf_map=coco_idf,
        )
        meta = q[848:864]
        # Dim 8: text indicator ('Sale 50%', sign)
        assert meta[8] > 0.0
        # Dim 12: speech indicator (speaking)
        assert meta[12] > 0.0

    def test_b5_decoupled_subspace_scaling_semantic_invariance(self):
        """Verifies B5: Keyframe with objects vs keyframe without objects has identical
        semantic dot product with query that specifies no object requirements.
        Eliminates the +3.01% norm-coupling bias.
        """
        rng = np.random.default_rng(2026)
        raw_emb = rng.standard_normal((2, 768)).astype(np.float32)
        # Both keyframes have identical visual semantic vector
        raw_emb[1] = raw_emb[0]

        kf_map = {
            0: {"video_id": "v_b5", "pts_time": 5.0},
            1: {"video_id": "v_b5", "pts_time": 15.0},
        }
        # Keyframe 0 has NO objects, Keyframe 1 has person and car
        frame_objs = [
            {"video_id": "v_b5", "pts_sec": 15.0, "class_name": "person", "confidence": 0.9},
            {"video_id": "v_b5", "pts_sec": 15.0, "class_name": "car", "confidence": 0.85},
        ]
        durations = {"v_b5": 60.0}

        aug_emb, _, idf_map = build_augmented_864d_embeddings(
            raw_embeddings=raw_emb,
            keyframe_map=kf_map,
            frame_objects=frame_objs,
            videos_duration=durations,
        )

        # Both keyframes must have L2 norm within 1e-5 of 1.0
        norm_0 = float(np.linalg.norm(aug_emb[0]))
        norm_1 = float(np.linalg.norm(aug_emb[1]))
        assert abs(norm_0 - 1.0) < 1e-5
        assert abs(norm_1 - 1.0) < 1e-5

        # Query without object requirements (e.g. sunset landscape)
        sem_q = raw_emb[0].copy()
        q_vec = build_query_864d_vector(
            sem_vec=sem_q,
            query_text="a beautiful sunset over the mountains",
            coco_idf_map=idf_map,
        )

        sim_0 = float(np.dot(aug_emb[0, :768], q_vec[:768]))
        sim_1 = float(np.dot(aug_emb[1, :768], q_vec[:768]))
        # Semantic contribution must be strictly invariant across frames
        assert abs(sim_0 - sim_1) < 1e-6

    def test_b6_word_bounded_regex_false_positive_prevention(self):
        """Verifies B6: Word boundaries \\b prevent matching inside words like
        'context', 'design', 'assign', 'assignment', 'screening', 'widescreen'.
        """
        sem_vec = np.zeros(768, dtype=np.float32)
        coco_idf = {c: 1.0 for c in COCO_80_CLASSES}

        # Negative cases that previously false-positive matched without \b:
        negatives = [
            "discussing historical context in class",
            "modern system design and architecture",
            "assign tasks to team members",
            "completing homework assignment",
            "cancer screening in medical center",
        ]
        for query in negatives:
            q_vec = build_query_864d_vector(sem_vec, query, coco_idf)
            meta = q_vec[848:864]
            assert meta[8] == 0.0, f"Query '{query}' unexpectedly triggered OCR text flag (dim 8)!"

        # Positive cases that MUST match:
        positives = [
            "showing text on screen",
            "a large stop sign on the corner",
            "reading the title of the book",
            "store logo on the wall",
            "người đọc tiêu đề báo",
            "biển báo giao thông",
            'wearing shirt with "Videntra" print',
        ]
        for query in positives:
            q_vec = build_query_864d_vector(sem_vec, query, coco_idf)
            meta = q_vec[848:864]
            assert meta[8] > 0.0, f"Query '{query}' failed to trigger OCR text flag (dim 8)!"

    def test_p3_rebuild_augmented_864d_cache_validity(self, tmp_path):
        """Verifies P3: rebuild_augmented_864d_cache runs on CPU, produces normalized (N, 864) array."""
        out_file = tmp_path / "test_augmented.npy"
        aug, mean_vec, idf = rebuild_augmented_864d_cache(output_npy_path=out_file)

        assert isinstance(aug, np.ndarray)
        assert aug.shape[1] == 864
        assert aug.dtype == np.float32
        assert out_file.exists()

        loaded = np.load(out_file)
        assert loaded.shape == aug.shape
        # Check all vectors are normalized to 1.0
        norms = np.linalg.norm(loaded, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)
