import pyarrow as pa
import pyarrow.parquet as pq

from scripts.audit_blender_ledger import verify_ledger
from videntra.fixtures import build_fixtures
from videntra.ingestion.decode import decode_frames
from videntra.timeline.ledger import write_timeline_ledger


def test_independent_decode_detects_altered_timestamp(tmp_path):
    fixtures = tmp_path / "fixtures"
    build_fixtures(fixtures)
    video = fixtures / "vfr.mkv"
    ledger = tmp_path / "frames.parquet"
    write_timeline_ledger(ledger, decode_frames(video, include_pixels=False), "v", "rev")
    result = verify_ledger(video, ledger)
    assert result["decoded_frames"] == 5
    assert result["mismatched_rows"] == 0
    table = pq.read_table(ledger)
    pos = table.schema.get_field_index("timestamp_us")
    values = table["timestamp_us"].to_pylist()
    values[2] += 1
    table = table.set_column(pos, "timestamp_us", pa.array(values, type=pa.int64()))
    pq.write_table(table, ledger)
    assert verify_ledger(video, ledger)["mismatched_rows"] == 1
