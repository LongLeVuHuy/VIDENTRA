import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def media():
    return json.loads((ROOT / "manifests/qvh24.media.json").read_text("utf-8"))["records"][0]


@pytest.fixture
def query():
    return json.loads((ROOT / "docs/schemas/query_vi.synthetic.json").read_text("utf-8"))


@pytest.fixture
def result():
    return json.loads((ROOT / "docs/schemas/result.synthetic.json").read_text("utf-8"))

