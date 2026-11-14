import json
from fractions import Fraction
from types import SimpleNamespace

from scripts.propose_timeline_repairs import propose


def test_proposal_uses_ordinal_pts_and_explicit_unverified_origin(tmp_path, monkeypatch):
    outputs, media = tmp_path / "outputs", tmp_path / "media"
    outputs.mkdir()
    media.mkdir()
    (media / "v.mp4").write_bytes(b"synthetic")
    (outputs / "videos.json").write_text(json.dumps([{"video_id": "v", "time_base_num": 1, "time_base_den": 1000}]))
    (outputs / "keyframe_manifest.json").write_text(json.dumps([{"video_id": "v", "keyframe_id": "kf",
        "source_frame_index": 1, "pts": 2000, "timestamp_ms": 900}]))

    class Container:
        streams = SimpleNamespace(video=[object()])
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def decode(self, stream):
            return iter([SimpleNamespace(pts=1000, time_base=Fraction(1, 1000)),
                         SimpleNamespace(pts=2000, time_base=Fraction(1, 1000))])

    monkeypatch.setattr("scripts.propose_timeline_repairs.av.open", lambda path: Container())
    result = propose(outputs, media)
    assert result["status"] == "REVIEW_REQUIRED_NOT_APPLIED"
    assert result["videos_decoded"] == 1
    assert len(result["proposed_rows"]) == 1
    row = result["proposed_rows"][0]
    assert (row["old_timestamp_ms"], row["proposed_timestamp_ms"]) == (900, 1000)
    assert row["first_pts_seconds_num"] == 1
    assert row["status"] == "REVIEW_REQUIRED_NOT_APPLIED"


def test_missing_media_blocks_proposal(tmp_path):
    outputs, media = tmp_path / "outputs", tmp_path / "media"
    outputs.mkdir()
    media.mkdir()
    (outputs / "videos.json").write_text(json.dumps([{"video_id": "v", "time_base_num": 1, "time_base_den": 1000}]))
    (outputs / "keyframe_manifest.json").write_text(json.dumps([{"video_id": "v", "keyframe_id": "kf",
        "source_frame_index": 1, "pts": 2000, "timestamp_ms": 900}]))
    result = propose(outputs, media)
    assert result["proposed_rows"] == []
    assert result["blocked_videos"][0]["reason"] == "missing_or_ambiguous_media"
