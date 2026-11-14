"""Voice Activity Detection (VAD) with energy thresholds distinguishing silence from missing audio."""
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import av
import numpy as np


@dataclass(frozen=True)
class VADSegment:
    start_us: int
    end_us: int
    energy: float

    @property
    def duration_seconds(self) -> float:
        return (self.end_us - self.start_us) / 1_000_000.0


def extract_audio_mono(media_path: Path | str, target_sr: int = 16000) -> tuple[np.ndarray | None, bool]:
    """Decode audio stream to single-channel float32 array normalized in [-1.0, 1.0].
    
    Returns (samples, has_audio_stream).
    """
    path = Path(media_path)
    if not path.is_file():
        return None, False

    try:
        with av.open(str(path)) as container:
            audio_streams = [s for s in container.streams if s.type == "audio"]
            if not audio_streams:
                return None, False

            stream = audio_streams[0]
            resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sr)
            chunks = []

            for frame in container.decode(stream):
                resampled_frames = resampler.resample(frame)
                for rf in resampled_frames:
                    arr = rf.to_ndarray().flatten()
                    chunks.append(arr)

            if not chunks:
                return np.empty(0, dtype=np.float32), True

            full_audio = np.concatenate(chunks).astype(np.float32)
            return full_audio, True
    except Exception:
        return None, False


class VoiceActivityDetector:
    """Detects continuous speech intervals based on Short-Time Energy and frame clustering."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        energy_threshold: float = 0.02,
        min_speech_ms: int = 250,
        max_silence_gap_ms: int = 300,
    ):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.energy_threshold = energy_threshold
        self.min_speech_ms = min_speech_ms
        self.max_silence_gap_ms = max_silence_gap_ms

    def detect_segments(self, samples: np.ndarray) -> list[VADSegment]:
        """Process 1D float32 audio samples and return speech segments in microseconds."""
        if samples is None or len(samples) == 0:
            return []

        frame_len = int(self.sample_rate * (self.frame_ms / 1000.0))
        if frame_len <= 0 or len(samples) < frame_len:
            return []

        num_frames = len(samples) // frame_len
        frame_energies = np.empty(num_frames, dtype=np.float32)

        for i in range(num_frames):
            frame = samples[i * frame_len: (i + 1) * frame_len]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            frame_energies[i] = rms

        # Speech mask
        is_speech = frame_energies > self.energy_threshold

        segments: list[VADSegment] = []
        in_segment = False
        start_frame = 0
        silence_count = 0
        max_silence_frames = int((self.max_silence_gap_ms / 1000.0) / (self.frame_ms / 1000.0))

        for i, flag in enumerate(is_speech):
            if flag:
                if not in_segment:
                    in_segment = True
                    start_frame = i
                silence_count = 0
            else:
                if in_segment:
                    silence_count += 1
                    if silence_count > max_silence_frames:
                        # Close segment
                        end_frame = i - silence_count + 1
                        dur_ms = (end_frame - start_frame) * self.frame_ms
                        if dur_ms >= self.min_speech_ms:
                            start_us = int(start_frame * self.frame_ms * 1000)
                            end_us = int(end_frame * self.frame_ms * 1000)
                            avg_energy = float(np.mean(frame_energies[start_frame:end_frame]))
                            segments.append(VADSegment(start_us=start_us, end_us=end_us, energy=avg_energy))
                        in_segment = False
                        silence_count = 0

        # Trailing segment
        if in_segment:
            end_frame = num_frames
            dur_ms = (end_frame - start_frame) * self.frame_ms
            if dur_ms >= self.min_speech_ms:
                start_us = int(start_frame * self.frame_ms * 1000)
                end_us = int(end_frame * self.frame_ms * 1000)
                avg_energy = float(np.mean(frame_energies[start_frame:end_frame]))
                segments.append(VADSegment(start_us=start_us, end_us=end_us, energy=avg_energy))

        return segments
