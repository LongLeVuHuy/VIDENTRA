"""Regression tests for Phase 9 Section 2.1: OCR/ASR ingest safety, recipe isolation, and rollback."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts.ingest_kaggle_offline_results import ingest_ocr_asr_artifacts
from scripts.kaggle_multimodal_offline_worker import run_ocr_asr_extraction
from videntra.storage.database import (
    OfflineMetadataStore,
    PayloadConflictError,
    compute_recipe_hash,
    make_deterministic_record_id,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_two_videos(db_path: Path) -> OfflineMetadataStore:
    store = OfflineMetadataStore(db_path)
    for vid in ("vid_001", "vid_002"):
        store.register_video({
            "video_id": vid,
            "dataset": "test",
            "file_path": f"{vid}.mp4",
            "duration_ms": 10000,
            "fps": 25.0,
            "time_base_num": 1,
            "time_base_den": 1000,
            "width": 320,
            "height": 240,
            "total_frames": 250,
            "has_audio": 1,
            "status": "completed",
        })
        store.insert_keyframes([
            {
                "keyframe_id": f"{vid}__kf_00000",
                "video_id": vid,
                "keyframe_index": 0,
                "source_frame_index": 0,
                "pts": 0,
                "time_base": "1/1000",
                "timestamp_ms": 0,
                "keyframe_path": "",
                "thumbnail_path": "",
                "faiss_id": 0 if vid == "vid_001" else 1,
                "sampling_reason": "uniform",
            }
        ])
    return store


def _write_bundle(
    bundle_dir: Path,
    *,
    ocr_obs: list[dict] | None = None,
    asr_segs: list[dict] | None = None,
    ocr_model_prov: dict | None = None,
    asr_model_prov: dict | None = None,
    write_receipt: bool = True,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict] = {}
    if ocr_obs is not None:
        ocr_path = bundle_dir / "ocr_observations.jsonl"
        ocr_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ocr_obs),
            encoding="utf-8",
        )
        artifacts["ocr_observations"] = {
            "filename": "ocr_observations.jsonl",
            "sha256": _sha256_file(ocr_path),
            "line_count": len(ocr_obs),
        }
    if asr_segs is not None:
        asr_path = bundle_dir / "asr_segments.jsonl"
        asr_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in asr_segs),
            encoding="utf-8",
        )
        artifacts["asr_segments"] = {
            "filename": "asr_segments.jsonl",
            "sha256": _sha256_file(asr_path),
            "line_count": len(asr_segs),
        }
    if write_receipt:
        receipt = {
            "receipt_version": "2.0",
            "status": "success",
            "mode": "ocr_asr",
            "artifacts": artifacts,
            "ocr_models": ocr_model_prov or {"general": {"model_id": "RapidOCR", "revision": "1.4.4"}},
            "asr_models": asr_model_prov or {"backend": "phowhisper_small", "model_id": "vinai/PhoWhisper-small"},
        }
        (bundle_dir / "ocr_asr_receipt.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _logical_db_snapshot(db_path: Path) -> tuple[str, dict[str, int]]:
    with sqlite3.connect(str(db_path)) as conn:
        counts = {
            tbl: conn.execute(f"SELECT COUNT(*) FROM {tbl};").fetchone()[0]
            for tbl in (
                "OCRObservations",
                "OCREvidenceGroups",
                "OCREvidenceMembers",
                "ASRSegments",
                "ASRWords",
                "ASREvidenceGroups",
                "ASREvidenceMembers",
                "fts_ocr",
                "fts_asr",
            )
        }
        logical_rows = {
            "ocr_obs": conn.execute(
                "SELECT observation_id, video_id, keyframe_id, timestamp_ms, model_id, raw_text, "
                "bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence, language "
                "FROM OCRObservations ORDER BY observation_id;"
            ).fetchall(),
            "ocr_grp": conn.execute(
                "SELECT group_id, video_id, keyframe_id, consensus_text, status, models_involved "
                "FROM OCREvidenceGroups ORDER BY group_id;"
            ).fetchall(),
            "ocr_mem": conn.execute(
                "SELECT group_id, observation_id, model_id FROM OCREvidenceMembers ORDER BY group_id, observation_id;"
            ).fetchall(),
            "asr_seg": conn.execute(
                "SELECT segment_id, video_id, model_id, start_ms, end_ms, raw_text, confidence, language "
                "FROM ASRSegments ORDER BY segment_id;"
            ).fetchall(),
            "asr_word": conn.execute(
                "SELECT word_id, segment_id, word, start_ms, end_ms, confidence FROM ASRWords ORDER BY word_id;"
            ).fetchall(),
            "asr_grp": conn.execute(
                "SELECT group_id, video_id, start_ms, end_ms, consensus_text, status, models_involved "
                "FROM ASREvidenceGroups ORDER BY group_id;"
            ).fetchall(),
            "asr_mem": conn.execute(
                "SELECT group_id, segment_id, model_id FROM ASREvidenceMembers ORDER BY group_id, segment_id;"
            ).fetchall(),
        }
    digest = hashlib.sha256(json.dumps(logical_rows, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return digest, counts


def test_worker_emits_collision_free_recipe_ids_and_v2_receipt(tmp_path: Path):
    img_path = tmp_path / "kf0.png"
    Image.fromarray(np.full((32, 32, 3), 200, dtype=np.uint8)).save(img_path)
    manifest = tmp_path / "keyframes_manifest.jsonl"
    manifest.write_text(
        json.dumps({
            "video_id": "vid_001",
            "keyframe_id": "vid_001__kf_00000",
            "timestamp_ms": 0,
            "keyframe_path": str(img_path),
            "timeline_revision": "tl_v1",
        }) + "\n",
        encoding="utf-8",
    )

    class _FakeRouter:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def route_and_recognize(self, crop, hint_text=None):
            return SimpleNamespace(text="bảng hiệu Hà Nội", confidence=0.92, model_id=self.model_id), "test"

    def _fake_det(_img):
        return [((0.1, 0.1, 0.8, 0.5), "", 0.95)]

    out_rapid = tmp_path / "out_rapid"
    r1 = run_ocr_asr_extraction(
        keyframes_manifest=manifest,
        out_dir=out_rapid,
        ocr_backend="rapidocr",
        enable_vietocr=False,
        device="cpu",
        ocr_router=_FakeRouter("RapidOCR"),
        detector_fn=_fake_det,
    )
    out_paddle = tmp_path / "out_paddle"
    r2 = run_ocr_asr_extraction(
        keyframes_manifest=manifest,
        out_dir=out_paddle,
        ocr_backend="paddleocr_en",
        enable_vietocr=True,
        device="cpu",
        ocr_router=_FakeRouter("vietocr/vgg_transformer"),
        detector_fn=_fake_det,
    )

    assert r1["receipt_version"] == "2.0"
    assert r1["recipe_hashes"]["ocr"] != r2["recipe_hashes"]["ocr"]
    obs1 = json.loads((out_rapid / "ocr_observations.jsonl").read_text(encoding="utf-8").splitlines()[0])
    obs2 = json.loads((out_paddle / "ocr_observations.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert obs1["observation_id"] != obs2["observation_id"]
    assert obs1["observation_id"].startswith("vid_001__kf_00000__ocr_")
    assert obs2["observation_id"].startswith("vid_001__kf_00000__ocr_")
    assert r1["artifacts"]["ocr_observations"]["line_count"] == 1
    assert r1["artifacts"]["ocr_observations"]["sha256"] == _sha256_file(out_rapid / "ocr_observations.jsonl")


def test_two_recipes_same_video_and_ordinal_preserved_with_words_members_and_fts(tmp_path: Path):
    db_path = tmp_path / "metadata.db"
    _seed_two_videos(db_path)

    fw_hash = compute_recipe_hash({"record_type": "asr", "model_id": "faster_whisper_small", "rev": "r1"})
    pho_hash = compute_recipe_hash({"record_type": "asr", "model_id": "phowhisper_small", "rev": "r2"})
    fw_seg_id = make_deterministic_record_id("asr", "vid_001", fw_hash, 0)
    pho_seg_id = make_deterministic_record_id("asr", "vid_001", pho_hash, 0)
    assert fw_seg_id != pho_seg_id

    ocr_a_hash = compute_recipe_hash({"record_type": "ocr", "model_id": "RapidOCR"})
    ocr_b_hash = compute_recipe_hash({"record_type": "ocr", "model_id": "vietocr/vgg_transformer"})
    obs_a_id = make_deterministic_record_id("ocr", "vid_001__kf_00000", ocr_a_hash, 0)
    obs_b_id = make_deterministic_record_id("ocr", "vid_001__kf_00000", ocr_b_hash, 0)

    bundle_fw = tmp_path / "bundle_fw"
    _write_bundle(
        bundle_fw,
        ocr_obs=[{
            "observation_id": obs_a_id,
            "video_id": "vid_001",
            "keyframe_id": "vid_001__kf_00000",
            "timestamp_ms": 0,
            "model_id": "RapidOCR",
            "raw_text": "cà phê sữa",
            "bbox_x1": 0.1,
            "bbox_y1": 0.1,
            "bbox_x2": 0.6,
            "bbox_y2": 0.4,
            "confidence": 0.88,
        }],
        asr_segs=[{
            "segment_id": fw_seg_id,
            "video_id": "vid_001",
            "model_id": "faster_whisper_small",
            "start_ms": 500,
            "end_ms": 2500,
            "raw_text": "xin chào Hà Nội",
            "confidence": 0.82,
            "words": [
                {"word_id": f"{fw_seg_id}__w_0", "word": "xin", "start_ms": 500, "end_ms": 900, "confidence": 0.9},
                {"word_id": f"{fw_seg_id}__w_1", "word": "chào", "start_ms": 900, "end_ms": 1400, "confidence": 0.85},
            ],
        }],
        asr_model_prov={"backend": "faster_whisper", "model_id": "Systran/faster-whisper-small"},
    )

    bundle_pho = tmp_path / "bundle_pho"
    _write_bundle(
        bundle_pho,
        ocr_obs=[{
            "observation_id": obs_b_id,
            "video_id": "vid_001",
            "keyframe_id": "vid_001__kf_00000",
            "timestamp_ms": 0,
            "model_id": "vietocr/vgg_transformer",
            "raw_text": "cà phê sữa đá",
            "bbox_x1": 0.1,
            "bbox_y1": 0.1,
            "bbox_x2": 0.6,
            "bbox_y2": 0.4,
            "confidence": 0.95,
        }],
        asr_segs=[{
            "segment_id": pho_seg_id,
            "video_id": "vid_001",
            "model_id": "phowhisper_small",
            "start_ms": 500,
            "end_ms": 2500,
            "raw_text": "xin chào Hà Nội",
            "confidence": 0.94,
            "words": [
                {"word_id": f"{pho_seg_id}__w_0", "word": "xin", "start_ms": 500, "end_ms": 950, "confidence": 0.94},
            ],
        }],
        ocr_model_prov={"general": {"model_id": "vietocr/vgg_transformer"}},
        asr_model_prov={"backend": "phowhisper_small", "model_id": "vinai/PhoWhisper-small"},
    )

    res1 = ingest_ocr_asr_artifacts(bundle_fw, db_path)
    res2 = ingest_ocr_asr_artifacts(bundle_pho, db_path)
    assert res1["provenance_status"] == "verified"
    assert res2["provenance_status"] == "verified"

    with sqlite3.connect(str(db_path)) as conn:
        seg_models = {r[0] for r in conn.execute("SELECT model_id FROM ASRSegments;").fetchall()}
        assert seg_models == {"faster_whisper_small", "phowhisper_small"}

        word_count = conn.execute("SELECT COUNT(*) FROM ASRWords;").fetchone()[0]
        assert word_count == 3

        asr_members = conn.execute(
            "SELECT segment_id, model_id FROM ASREvidenceMembers ORDER BY model_id;"
        ).fetchall()
        assert asr_members == [
            (fw_seg_id, "faster_whisper_small"),
            (pho_seg_id, "phowhisper_small"),
        ]

        ocr_members = conn.execute(
            "SELECT observation_id, model_id FROM OCREvidenceMembers ORDER BY model_id;"
        ).fetchall()
        assert ocr_members == [
            (obs_a_id, "RapidOCR"),
            (obs_b_id, "vietocr/vgg_transformer"),
        ]

        fts_asr_hits = conn.execute(
            "SELECT video_id, model_id FROM fts_asr WHERE fts_asr MATCH 'ha noi';"
        ).fetchall()
        assert len(fts_asr_hits) == 1
        assert fts_asr_hits[0][0] == "vid_001"
        assert "faster_whisper_small" in fts_asr_hits[0][1]
        assert "phowhisper_small" in fts_asr_hits[0][1]

        fts_ocr_hits = conn.execute(
            "SELECT video_id FROM fts_ocr WHERE fts_ocr MATCH 'ca phe';"
        ).fetchall()
        assert len(fts_ocr_hits) == 1


def test_retry_same_bundle_is_idempotent_and_conflict_raises(tmp_path: Path):
    db_path = tmp_path / "metadata.db"
    _seed_two_videos(db_path)

    seg_id = make_deterministic_record_id(
        "asr",
        "vid_001",
        compute_recipe_hash({"record_type": "asr", "model_id": "phowhisper_small"}),
        0,
    )
    bundle = tmp_path / "bundle"
    _write_bundle(
        bundle,
        asr_segs=[{
            "segment_id": seg_id,
            "video_id": "vid_001",
            "model_id": "phowhisper_small",
            "start_ms": 0,
            "end_ms": 2000,
            "raw_text": "một hai ba",
            "confidence": 0.9,
        }],
    )

    ingest_ocr_asr_artifacts(bundle, db_path)
    hash_before, counts_before = _logical_db_snapshot(db_path)

    retry_res = ingest_ocr_asr_artifacts(bundle, db_path)
    hash_after, counts_after = _logical_db_snapshot(db_path)
    assert retry_res["asr_records_inserted"] == 0
    assert retry_res["fts_rebuilt"] is False
    assert hash_after == hash_before
    assert counts_after == counts_before

    # Same ID with different payload must raise PayloadConflictError and preserve DB
    conflict_bundle = tmp_path / "conflict_bundle"
    _write_bundle(
        conflict_bundle,
        asr_segs=[{
            "segment_id": seg_id,
            "video_id": "vid_001",
            "model_id": "phowhisper_small",
            "start_ms": 0,
            "end_ms": 2000,
            "raw_text": "nội dung bị thay đổi",
            "confidence": 0.9,
        }],
    )
    with pytest.raises(PayloadConflictError, match=seg_id):
        ingest_ocr_asr_artifacts(conflict_bundle, db_path)
    hash_conflict, counts_conflict = _logical_db_snapshot(db_path)
    assert hash_conflict == hash_before
    assert counts_conflict == counts_before


def test_mid_transaction_failure_before_fts_rolls_back_completely(tmp_path: Path):
    db_path = tmp_path / "metadata.db"
    _seed_two_videos(db_path)
    hash_initial, counts_initial = _logical_db_snapshot(db_path)

    seg_id = make_deterministic_record_id(
        "asr",
        "vid_001",
        compute_recipe_hash({"record_type": "asr", "model_id": "phowhisper_small"}),
        0,
    )
    bundle = tmp_path / "bundle"
    _write_bundle(
        bundle,
        asr_segs=[{
            "segment_id": seg_id,
            "video_id": "vid_001",
            "model_id": "phowhisper_small",
            "start_ms": 100,
            "end_ms": 1500,
            "raw_text": "dữ liệu sẽ rollback",
            "confidence": 0.91,
        }],
    )

    def _fail_before_fts(_conn: sqlite3.Connection) -> None:
        raise RuntimeError("simulated failure before FTS rebuild")

    with pytest.raises(RuntimeError, match="simulated failure"):
        ingest_ocr_asr_artifacts(bundle, db_path, _before_fts_hook=_fail_before_fts)

    hash_after, counts_after = _logical_db_snapshot(db_path)
    assert hash_after == hash_initial
    assert counts_after == counts_initial


def test_preflight_rejects_tampered_receipt_wrong_video_keyframe_and_marks_legacy_unverified(tmp_path: Path):
    db_path = tmp_path / "metadata.db"
    _seed_two_videos(db_path)

    # 1. Tampered receipt hash
    tampered = tmp_path / "tampered"
    seg_id = make_deterministic_record_id("asr", "vid_001", "abcd1234ef56", 0)
    _write_bundle(
        tampered,
        asr_segs=[{
            "segment_id": seg_id,
            "video_id": "vid_001",
            "model_id": "phowhisper_small",
            "start_ms": 0,
            "end_ms": 1000,
            "raw_text": "ok",
            "confidence": 0.9,
        }],
    )
    receipt_path = tampered / "ocr_asr_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifacts"]["asr_segments"]["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ingest_ocr_asr_artifacts(tampered, db_path, dry_run=True)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ingest_ocr_asr_artifacts(tampered, db_path, dry_run=False)

    # 2. Wrong-video keyframe reference
    wrong_kf = tmp_path / "wrong_kf"
    _write_bundle(
        wrong_kf,
        ocr_obs=[{
            "observation_id": make_deterministic_record_id("ocr", "vid_001__kf_00000", "abcd1234ef56", 0),
            "video_id": "vid_002",
            "keyframe_id": "vid_001__kf_00000",
            "timestamp_ms": 0,
            "model_id": "RapidOCR",
            "raw_text": "test",
            "bbox_x1": 0.1,
            "bbox_y1": 0.1,
            "bbox_x2": 0.5,
            "bbox_y2": 0.5,
            "confidence": 0.9,
        }],
    )
    with pytest.raises(ValueError, match="belongs to video 'vid_001', not 'vid_002'"):
        ingest_ocr_asr_artifacts(wrong_kf, db_path, dry_run=True)

    # 3. Invalid ASR interval (start >= end) and invalid OCR box
    bad_asr = tmp_path / "bad_asr"
    _write_bundle(
        bad_asr,
        asr_segs=[{
            "segment_id": seg_id,
            "video_id": "vid_001",
            "model_id": "phowhisper_small",
            "start_ms": 2000,
            "end_ms": 1000,
            "raw_text": "bad interval",
            "confidence": 0.9,
        }],
    )
    with pytest.raises(ValueError, match="invalid interval"):
        ingest_ocr_asr_artifacts(bad_asr, db_path, dry_run=True)

    bad_box = tmp_path / "bad_box"
    _write_bundle(
        bad_box,
        ocr_obs=[{
            "observation_id": make_deterministic_record_id("ocr", "vid_001__kf_00000", "abcd1234ef56", 0),
            "video_id": "vid_001",
            "keyframe_id": "vid_001__kf_00000",
            "timestamp_ms": 0,
            "model_id": "RapidOCR",
            "raw_text": "bad box",
            "bbox_x1": 0.6,
            "bbox_y1": 0.1,
            "bbox_x2": 0.4,
            "bbox_y2": 0.5,
            "confidence": 0.9,
        }],
    )
    with pytest.raises(ValueError, match="invalid normalized bbox"):
        ingest_ocr_asr_artifacts(bad_box, db_path, dry_run=True)

    # 4. Legacy artifact without receipt -> explicit unverified path
    legacy_dir = tmp_path / "legacy"
    _write_bundle(
        legacy_dir,
        asr_segs=[{
            "segment_id": "vid_001__asr_0",
            "video_id": "vid_001",
            "model_id": "faster_whisper_small",
            "start_ms": 0,
            "end_ms": 1000,
            "raw_text": "legacy transcript",
            "confidence": 0.85,
        }],
        write_receipt=False,
    )
    legacy_dry = ingest_ocr_asr_artifacts(legacy_dir, db_path, dry_run=True)
    assert legacy_dry["provenance_status"] == "unverified"
    assert legacy_dry["legacy_unverified"] is True
    legacy_res = ingest_ocr_asr_artifacts(legacy_dir, db_path, dry_run=False)
    assert legacy_res["provenance_status"] == "unverified"
    assert legacy_res["legacy_unverified"] is True
