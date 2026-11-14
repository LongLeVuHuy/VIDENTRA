from fractions import Fraction
import hashlib
from types import SimpleNamespace

import pytest

from scripts.kaggle_multimodal_offline_worker import ensure_output_files_absent, frame_time_seconds, verify_weights


def test_worker_uses_pts_and_rejects_unknown_mapping():
    stream = SimpleNamespace(time_base=Fraction(1, 12288))
    frame = SimpleNamespace(pts=24576, time_base=Fraction(1, 12288))
    assert frame_time_seconds(frame, stream) == 2.0
    assert frame_time_seconds(frame, stream, origin_pts=12288) == 1.0
    frame.pts = None
    with pytest.raises(ValueError, match="Missing frame PTS"):
        frame_time_seconds(frame, stream)
    frame.pts = 10
    frame.time_base = Fraction(1, 1000)
    with pytest.raises(ValueError, match="explicit mapping"):
        frame_time_seconds(frame, stream)


def test_worker_rejects_wrong_weights(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"fixture")
    expected = hashlib.sha256(b"fixture").hexdigest()
    assert verify_weights(path, expected) == expected
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_weights(path, "0" * 64)


def test_worker_refuses_existing_artifacts(tmp_path):
    (tmp_path / "summary.json").write_text("old")
    with pytest.raises(FileExistsError):
        ensure_output_files_absent(tmp_path)
    assert (tmp_path / "summary.json").read_text() == "old"
