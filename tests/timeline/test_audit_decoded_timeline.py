import json
from fractions import Fraction
from types import SimpleNamespace

from scripts.audit_decoded_timeline import audit


def test_audit_compares_decoded_frame_pts_and_reports_missing_files(tmp_path, monkeypatch):
    outputs, media = tmp_path / "outputs", tmp_path / "media"
    outputs.mkdir()
    media.mkdir()
    videos = [{"video_id": name, "time_base_num": 1, "time_base_den": 1000} for name in ("v1", "v2")]
    frames = [{"video_id": name, "source_frame_index": 1, "pts": 1000, "timestamp_ms": 900}
              for name in ("v1", "v2")]
    (outputs / "videos.json").write_text(json.dumps(videos), encoding="utf-8")
    (outputs / "keyframe_manifest.json").write_text(json.dumps(frames), encoding="utf-8")
    (media / "v1.mp4").touch()

    class Container:
        streams = SimpleNamespace(video=[object()])
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def decode(self, stream):
            return iter([SimpleNamespace(pts=0, time_base=Fraction(1, 1000)),
                         SimpleNamespace(pts=1000, time_base=Fraction(1, 1000))])

    monkeypatch.setattr("scripts.audit_decoded_timeline.av.open", lambda path: Container())
    result = audit(outputs, media)
    assert result["affected_videos_in_artifact"] == 2
    assert result["missing_or_ambiguous_files"] == 1
    assert result["artifact_pts_match_decoded"] == 1
    assert result["saved_timestamp_differs_clip_pts_gt_1ms"] == 1
    assert result["max_timestamp_difference_ms"] == 100


def test_audit_subtracts_nonzero_first_pts(tmp_path, monkeypatch):
    outputs, media = tmp_path / "outputs", tmp_path / "media"
    outputs.mkdir()
    media.mkdir()
    (outputs / "videos.json").write_text(json.dumps([{"video_id": "v", "time_base_num": 1, "time_base_den": 1000}]), encoding="utf-8")
    (outputs / "keyframe_manifest.json").write_text(json.dumps([{"video_id": "v", "source_frame_index": 1,
        "pts": 1100, "timestamp_ms": 1000}]), encoding="utf-8")
    (media / "v.mp4").touch()

    class Container:
        streams = SimpleNamespace(video=[object()])
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def decode(self, stream):
            return iter([SimpleNamespace(pts=100, time_base=Fraction(1, 1000)),
                         SimpleNamespace(pts=1100, time_base=Fraction(1, 1000))])

    monkeypatch.setattr("scripts.audit_decoded_timeline.av.open", lambda path: Container())
    result = audit(outputs, media)
    assert result["videos_first_pts_nonzero_or_missing"] == 1
    assert result["artifact_pts_match_decoded"] == 1
    assert result["saved_timestamp_differs_clip_pts_gt_1ms"] == 0
