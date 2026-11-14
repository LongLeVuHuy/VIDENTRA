"""Unit and integration tests for visual retrieval engine and moment ranking."""
from fractions import Fraction
from pathlib import Path
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from videntra.contracts import validate
from videntra.encoders.siglip2 import MockSigLIP2Encoder
from videntra.fixtures import build_fixtures
from videntra.ingestion.decode import decode_frames
from videntra.ingestion.probe import probe_media
from videntra.indexing.ids import IDMappingStore
from videntra.indexing.faiss_store import VisualIndexStore
from videntra.retrieval.visual import RankedMoment, VisualRetrievalEngine, temporal_iou
from videntra.timeline.ledger import write_timeline_ledger


def test_temporal_iou_edge_cases():
    # Identical intervals
    assert np.isclose(temporal_iou(0.0, 2.0, 0.0, 2.0), 1.0)
    # Disjoint intervals
    assert temporal_iou(0.0, 2.0, 3.0, 5.0) == 0.0
    # Overlapping intervals: [0, 3) and [1, 4) -> intersection=2, union=4 -> 0.5
    assert np.isclose(temporal_iou(0.0, 3.0, 1.0, 4.0), 0.5)
    # Subset: [1, 3) inside [0, 4) -> intersection=2, union=4 -> 0.5
    assert np.isclose(temporal_iou(1.0, 3.0, 0.0, 4.0), 0.5)


def test_visual_retrieval_end_to_end(tmp_path):
    # Create test media fixture
    fixtures_dir = tmp_path / "fixtures"
    build_fixtures(fixtures_dir)
    media_path = fixtures_dir / "cfr.mkv"

    # Decode and create ledger
    probe_res = probe_media(media_path)
    frames = list(decode_frames(media_path, probe_res, include_pixels=True))

    tl_path = tmp_path / "frames.parquet"
    write_timeline_ledger(tl_path, frames, video_id="video_cfr", timeline_revision="rev1")

    # Mock samples parquet
    samples_path = tmp_path / "samples.parquet"
    schema = pa.schema([
        ("sampled_frame_id", pa.string()),
        ("video_id", pa.string()),
        ("timeline_revision", pa.string()),
        ("sampler_revision", pa.string()),
        ("original_decoded_frame_index", pa.int64()),
        ("timestamp_us", pa.int64()),
        ("timestamp_seconds", pa.float64()),
        ("reasons", pa.list_(pa.string())),
        ("is_mandatory", pa.bool_()),
        ("phash", pa.string()),
        ("scene_id", pa.string()),
        ("epoch_id", pa.string()),
    ])
    data = {
        "sampled_frame_id": ["sample_0", "sample_2", "sample_4"],
        "video_id": ["video_cfr", "video_cfr", "video_cfr"],
        "timeline_revision": ["rev1", "rev1", "rev1"],
        "sampler_revision": ["v1", "v1", "v1"],
        "original_decoded_frame_index": [0, 2, 4],
        "timestamp_us": [0, 400_000, 800_000],
        "timestamp_seconds": [0.0, 0.4, 0.8],
        "reasons": [["first"], ["burst"], ["last"]],
        "is_mandatory": [True, False, True],
        "phash": ["0000000000000000", "1111111111111111", "2222222222222222"],
        "scene_id": ["scene_0000", "scene_0000", "scene_0000"],
        "epoch_id": ["epoch_0", "epoch_0", "epoch_0"],
    }
    pq.write_table(pa.Table.from_pydict(data, schema=schema), str(samples_path))

    # Initialize retrieval engine with Mock encoder
    engine = VisualRetrievalEngine(
        encoder=MockSigLIP2Encoder(),
        index_store=VisualIndexStore(dimension=768, prefer_faiss=True),
        id_mapping=IDMappingStore(),
    )

    indexed_count = engine.index_video(
        video_id="video_cfr",
        media_path=media_path,
        samples_parquet_path=samples_path,
        timeline_parquet_path=tl_path,
    )
    assert indexed_count == 3
    assert len(engine.index_store) == 3

    # Search
    moments = engine.search("test search query", top_k_moments=5)
    assert len(moments) > 0
    m0 = moments[0]
    assert m0.video_id == "video_cfr"
    assert m0.moment_start_seconds <= m0.moment_end_seconds
    assert m0.ranking_score > -10.0

    # Test conversion to Result schema and normative schema validation
    res_dict = m0.to_result_dict(media_uri="file:///test/cfr.mkv")
    # Must pass Result schema validation!
    validate("result", res_dict)
