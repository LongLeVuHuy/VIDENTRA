import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def media():
    return {
        "schema_version": "1.0",
        "video_id": "sha256:c12895f168c00faf3fa7bab1c3a69a62e3ce3a1f5e735baee9f003928489b93a",
        "source_video_id": "msrvtt:clip:video0",
        "sha256": "c12895f168c00faf3fa7bab1c3a69a62e3ce3a1f5e735baee9f003928489b93a",
        "media_uri": "asset://msrvtt/video0.mp4",
        "video_stream_index": 0,
        "audio_stream_index": 1,
        "source_offset_us": None,
        "source_mapping_kind": "unknown",
        "timeline_origin": {"num": 0, "den": 1},
        "duration_us": 15000000,
        "rights_status": "unknown",
        "rights_reference": None,
        "original_dataset_id": "MSR-VTT",
        "official_split": "test",
        "exposure": "development_previously_used",
        "decode_status": "pending",
        "timeline_revision": None,
        "modality_status": {"visual": "not_run", "ocr": "not_run", "asr": "not_run"},
    }


@pytest.fixture
def query():
    return json.loads((ROOT / "docs/schemas/query_vi.synthetic.json").read_text("utf-8"))


@pytest.fixture
def result():
    return json.loads((ROOT / "docs/schemas/result.synthetic.json").read_text("utf-8"))

