"""API & CLI tests for P6 competition features (health, dataset filter, neighbors, similar search, exports)."""
from __future__ import annotations

from pathlib import Path
import sys

import faiss
from fastapi.testclient import TestClient
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from videntra.api.app import create_app
from videntra.retrieval.unified_pipeline import (
    UnifiedMultimodalSearchPipeline,
    infer_video_dataset_family,
)


def _make_p6_client() -> TestClient:
    rng = np.random.default_rng(19)
    raw_768 = rng.standard_normal((6, 768)).astype(np.float32)
    # Make L01_V001 frame 0 & video123 frame 0 highly similar
    raw_768[2] = raw_768[0] * 0.96 + 0.04 * raw_768[2]
    raw_768 /= np.linalg.norm(raw_768, axis=1, keepdims=True)
    aug_864 = np.zeros((6, 864), dtype=np.float32)
    aug_864[:, :768] = raw_768
    aug_864 /= np.linalg.norm(aug_864, axis=1, keepdims=True)

    kf_map = {
        0: {"video_id": "L01_V001", "pts_time": 1.0, "source_frame_index": 25},
        1: {"video_id": "L01_V001", "pts_time": 3.0, "source_frame_index": 75},
        2: {"video_id": "video123", "pts_time": 2.0, "source_frame_index": 50},
        3: {"video_id": "video123", "pts_time": 5.0, "source_frame_index": 125},
        4: {"video_id": "xY9_ab_10.0_40.0", "pts_time": 12.0, "source_frame_index": 300},
        5: {"video_id": "xY9_ab_10.0_40.0", "pts_time": 20.0, "source_frame_index": 500},
    }
    vid_kfs = {
        "L01_V001": [{"faiss_id": 0, "pts_time": 1.0, "source_frame_index": 25}, {"faiss_id": 1, "pts_time": 3.0, "source_frame_index": 75}],
        "video123": [{"faiss_id": 2, "pts_time": 2.0, "source_frame_index": 50}, {"faiss_id": 3, "pts_time": 5.0, "source_frame_index": 125}],
        "xY9_ab_10.0_40.0": [{"faiss_id": 4, "pts_time": 12.0, "source_frame_index": 300}, {"faiss_id": 5, "pts_time": 20.0, "source_frame_index": 500}],
    }
    p = UnifiedMultimodalSearchPipeline(lazy_load=True, enable_videoprism=False, use_disk_cache=False)
    p.all_embeddings = raw_768
    p.augmented_embeddings = aug_864
    p.keyframe_map = kf_map
    p.video_keyframes = vid_kfs
    p.video_durations = {"L01_V001": 10.0, "video123": 12.0, "xY9_ab_10.0_40.0": 40.0}
    p.aug_index = faiss.IndexFlatIP(864)
    p.aug_index.add(aug_864)
    p.index = faiss.IndexFlatIP(768)
    p.index.add(raw_768)
    p.encode_texts = lambda texts: np.tile(raw_768[0:1], (len(texts), 1))  # type: ignore[method-assign]
    app = create_app(pipeline=p, snapshot_id="p6_test")
    return TestClient(app)


def test_dataset_family_inference_and_filter() -> None:
    assert infer_video_dataset_family("video9999") == "msrvtt"
    assert infer_video_dataset_family("L02_V014") == "avc"
    assert infer_video_dataset_family("nOTjA_3.0_33.0") == "qvhighlights"

    client = _make_p6_client()
    r = client.post("/v1/search", json={"query": "street scene", "mode": "accurate", "filters": {"dataset": "avc"}})
    assert r.status_code == 200
    vids = [row["video_id"] for row in r.json()["results"]]
    assert vids == ["L01_V001"]


def test_health_neighbors_similar_and_exports() -> None:
    client = _make_p6_client()

    # /health reports unified telemetry accurately
    h = client.get("/health").json()
    assert h["engine_active"] is True
    assert h["backend"] == "unified"
    assert h["indexed_vectors"] == 6
    assert h["indexed_videos"] == 3

    # /v1/videos/{video_id}/neighbors
    nb = client.get("/v1/videos/L01_V001/neighbors?timestamp=1.1&window=10&limit=5")
    assert nb.status_code == 200
    nb_data = nb.json()
    assert nb_data["count"] == 2
    assert nb_data["neighbors"][0]["is_anchor"] is True
    assert nb_data["neighbors"][0]["source_frame_index"] == 25

    # /v1/search/similar
    sim = client.post(
        "/v1/search/similar",
        json={"video_id": "L01_V001", "timestamp_seconds": 1.0, "exclude_self": True, "top_k": 5},
    )
    assert sim.status_code == 200
    sim_rows = sim.json()["results"]
    assert sim_rows and sim_rows[0]["video_id"] == "video123"

    # Exports: AIC KIS + VBS CSV with formula injection escape
    sample_rows = [
        {
            "video_id": "=cmd|'bad'",
            "representative_original_frame_index": 25,
            "moment_start_seconds": 1.0,
            "moment_end_seconds": 4.0,
            "ranking_score": 0.912,
        }
    ]
    aic = client.post("/v1/exports", json={"results": sample_rows, "format": "aic_kis"})
    assert aic.status_code == 200
    assert aic.text.strip() == "'=cmd|'bad',25"

    vbs = client.post("/v1/exports", json={"results": sample_rows, "format": "vbs_csv"})
    assert vbs.status_code == 200
    lines = [l.strip() for l in vbs.text.strip().splitlines()]
    assert lines[0] == "video_id,start_ms,end_ms,frame_index,score"
    assert lines[1].startswith("'=cmd|'bad',1000,4000,25,0.912")
