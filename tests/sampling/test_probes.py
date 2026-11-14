"""Unit tests for independent OCR probes."""
from fractions import Fraction
from pathlib import Path
import pytest
import numpy as np

from videntra.ingestion.decode import DecodedFrame
from videntra.sampling.probes import select_ocr_probes


def _make_dummy_frame(ordinal: int, t_sec: float) -> DecodedFrame:
    pts = int(round(t_sec * 1000))
    return DecodedFrame(
        ordinal=ordinal,
        pts=pts,
        dts=pts,
        time_base_num=1,
        time_base_den=1000,
        origin=Fraction(0),
        is_keyframe=(ordinal == 0),
        width=64,
        height=48,
    )


def test_ocr_probes_grid_and_cuts(tmp_path):
    # 20 frames at 2 fps -> 10 seconds (0.0, 0.5, 1.0, 1.5, ..., 9.5)
    frames = [_make_dummy_frame(i, i * 0.5) for i in range(20)]
    # Cut at ordinal 7 (3.5s)
    manifest = select_ocr_probes(frames, "video_ocr", probe_fps=1.0, shot_cut_ordinals=[7])

    assert len(manifest.probes) >= 10
    ordinals = [p.original_decoded_frame_index for p in manifest.probes]
    assert 0 in ordinals
    assert 7 in ordinals

    pq_path = tmp_path / "ocr_probes.parquet"
    manifest.to_parquet(pq_path)
    assert pq_path.exists()
    assert pq_path.stat().st_size > 0
