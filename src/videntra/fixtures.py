"""Small FFmpeg-generated test media with independently specified rational times."""
from fractions import Fraction
import json
from pathlib import Path
import subprocess

from .registry.store import sha256_file


def run(command):
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=120).stdout


def probe(path: Path, *, frames=True):
    args = ["ffprobe", "-v", "error", "-show_streams", "-of", "json"]
    if frames:
        args += ["-show_frames"]
    return json.loads(run(args + [str(path)]))


def build_fixtures(destination: Path, include_extended: bool = False):
    """Build canonical and extended test fixtures."""
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError("Fixture destination must be empty; existing artifacts are preserved")

    specs = {
        "cfr": ("N/(5*TB)", [Fraction(i, 5) for i in range(5)], False),
        "vfr": ("(N+floor(N/2))/(5*TB)", [Fraction(i, 5) for i in [0, 1, 3, 4, 6]], False),
        "nonzero_pts": ("N/(5*TB)+2/TB", [Fraction(2) + Fraction(i, 5) for i in range(5)], False),
        "audio_delay": ("N/(5*TB)", [Fraction(i, 5) for i in range(5)], True),
    }

    reports = []
    for name, (expression, expected, audio) in specs.items():
        path = destination / (name + ".mkv")
        command = [
            "ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "lavfi",
            "-i", "testsrc2=size=64x48:rate=5:duration=1",
        ]
        if audio:
            command += [
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=0.4",
                "-filter_complex", "[0:v]setpts=" + expression + "[v];[1:a]asetpts=PTS+0.35/TB[a]",
                "-map", "[v]", "-map", "[a]", "-c:a", "pcm_s16le",
            ]
        else:
            command += ["-vf", "setpts=" + expression, "-an"]
        command += ["-c:v", "ffv1", "-fps_mode", "passthrough", "-avoid_negative_ts", "disabled", str(path)]
        run(command)
        observed = probe(path)
        streams = {s["index"]: s for s in observed["streams"]}
        video = [f for f in observed["frames"] if f["media_type"] == "video"]
        actual = [int(f["pts"]) * Fraction(streams[f["stream_index"]]["time_base"]) for f in video]
        if actual != expected:
            raise ValueError(f"Fixture {name} changed timestamps: {actual} != {expected}")
        audio_frames = [f for f in observed["frames"] if f["media_type"] == "audio"]
        delay = None
        if audio:
            first = audio_frames[0]
            delay = int(first["pts"]) * Fraction(streams[first["stream_index"]]["time_base"]) - expected[0]
            if delay != Fraction(35, 100):
                raise ValueError("Audio delay was normalized away")
        reports.append({
            "name": name,
            "file": path.name,
            "sha256": sha256_file(path),
            "expected_video_times": [str(t) for t in expected],
            "observed_video_times": [str(t) for t in actual],
            "origin": str(expected[0]),
            "expected_audio_delay": "7/20" if audio else None,
            "observed_audio_delay": str(delay) if delay is not None else None,
            "command": command,
        })

    if include_extended:
        # 1. B-frames fixture with H.264
        bf_path = destination / "b_frames.mkv"
        bf_cmd = [
            "ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "lavfi",
            "-i", "testsrc2=size=64x48:rate=10:duration=1",
            "-c:v", "libx264", "-bf", "2", "-g", "5",
            str(bf_path),
        ]
        run(bf_cmd)
        reports.append({
            "name": "b_frames",
            "file": bf_path.name,
            "sha256": sha256_file(bf_path),
            "command": bf_cmd,
            "has_b_frames": True,
        })

        # 2. Audio preroll (negative start timestamp)
        preroll_path = destination / "audio_preroll.mkv"
        preroll_cmd = [
            "ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "lavfi",
            "-i", "testsrc2=size=64x48:rate=5:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=1.2",
            "-filter_complex", "[0:v]setpts=PTS[v];[1:a]asetpts=PTS-0.2/TB[a]",
            "-map", "[v]", "-map", "[a]", "-c:v", "ffv1", "-c:a", "pcm_s16le",
            "-avoid_negative_ts", "disabled", str(preroll_path),
        ]
        run(preroll_cmd)
        reports.append({
            "name": "audio_preroll",
            "file": preroll_path.name,
            "sha256": sha256_file(preroll_path),
            "command": preroll_cmd,
            "negative_audio_start": True,
        })

        # 3. No audio fixture
        no_audio_path = destination / "no_audio.mkv"
        no_audio_cmd = [
            "ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "lavfi",
            "-i", "testsrc2=size=64x48:rate=5:duration=0.6",
            "-c:v", "ffv1", "-an", str(no_audio_path),
        ]
        run(no_audio_cmd)
        reports.append({
            "name": "no_audio",
            "file": no_audio_path.name,
            "sha256": sha256_file(no_audio_path),
            "command": no_audio_cmd,
            "has_audio": False,
        })

        # 4. Truncated / corrupt tail file
        corrupt_path = destination / "corrupt_tail.mkv"
        cfr_bytes = (destination / "cfr.mkv").read_bytes()
        # Truncate at 60% of file size
        corrupt_path.write_bytes(cfr_bytes[: int(len(cfr_bytes) * 0.6)])
        reports.append({
            "name": "corrupt_tail",
            "file": corrupt_path.name,
            "sha256": sha256_file(corrupt_path),
            "truncated": True,
        })

        # 5. Repeated PTS fixture
        rep_path = destination / "repeated_pts.mkv"
        rep_cmd = [
            "ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "lavfi",
            "-i", "testsrc2=size=64x48:rate=5:duration=1",
            "-vf", "setpts=if(eq(N\\,1)\\,0\\,N)/(5*TB)",
            "-c:v", "ffv1", "-fps_mode", "passthrough", "-avoid_negative_ts", "disabled",
            "-an", str(rep_path),
        ]
        run(rep_cmd)
        reports.append({
            "name": "repeated_pts",
            "file": rep_path.name,
            "sha256": sha256_file(rep_path),
            "command": rep_cmd,
            "repeated_pts": True,
        })

    receipt = {
        "synthetic": True,
        "ffmpeg": run(["ffmpeg", "-version"]).splitlines()[0],
        "fixtures": reports,
    }
    with (destination / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    return receipt
