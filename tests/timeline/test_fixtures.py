from fractions import Fraction
import subprocess
import pytest
from videntra.fixtures import build_fixtures
from videntra.contracts.timing import frame_endpoints


def test_real_ffmpeg_fixtures(tmp_path):
    receipt = build_fixtures(tmp_path / "media")
    assert len(receipt["fixtures"]) == 4
    by_name = {f["name"]: f for f in receipt["fixtures"]}
    times = [Fraction(t) for t in by_name["vfr"]["observed_video_times"]]
    assert len(set(b-a for a,b in zip(times,times[1:]))) > 1
    assert frame_endpoints(times, Fraction(3,5), Fraction(1)) == (2,3)
    assert by_name["nonzero_pts"]["origin"] == "2"
    assert by_name["audio_delay"]["observed_audio_delay"] == "7/20"
    with pytest.raises(FileExistsError):
        build_fixtures(tmp_path / "media")

