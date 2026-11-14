import copy
from fractions import Fraction
import json
from pathlib import Path
import pytest

from videntra.contracts import validate, ContractError
from videntra.contracts.timing import frame_endpoints, microseconds
from videntra.contracts.evidence import validate_evidence
from conftest import ROOT


@pytest.mark.parametrize("filename", ["query_vi.synthetic.json", "query_en.synthetic.json"])
def test_query_examples(filename):
    validate("query_graph", json.loads((ROOT / "docs/schemas" / filename).read_text("utf-8")))


def test_result_example(result):
    validate("result", result)
    with pytest.raises(ContractError):
        validate("result", result, production=True)


@pytest.mark.parametrize("field,value", [
    ("duration_us", -1), ("rights_status", "allowed"), ("source_mapping_kind", "constant_offset"),
    ("video_stream_index", -1), ("decode_status", "ready"), ("video_id", ""),
])
def test_media_invalid_cross_fields(media, field, value):
    media[field] = value
    with pytest.raises(ContractError):
        validate("media_asset", media)


@pytest.mark.parametrize("failure", ["duplicate", "role", "span", "logic_cycle", "coref_cycle", "temporal_cycle", "gap"])
def test_graph_invalid(query, failure):
    if failure == "duplicate":
        query["entities"][1]["id"] = "a"
    elif failure == "role":
        query["predicates"][0]["roles"]["recipient"] = "missing"
    elif failure == "span":
        query["predicates"][0]["source_span"] = "not in the user's query"
    elif failure == "logic_cycle":
        query["logic"][0]["children"].append("all")
    elif failure == "coref_cycle":
        query["entities"][0]["coreference_of"] = "b"
        query["entities"][1]["coreference_of"] = "a"
    elif failure == "temporal_cycle":
        extra = copy.deepcopy(query["predicates"][2])
        extra["id"] = "opposite"
        extra["roles"] = {"event_a": "speak", "event_b": "give"}
        query["predicates"].append(extra)
    elif failure == "gap":
        query["predicates"][2]["scope"].update(min_gap_ms=100, max_gap_ms=1)
    with pytest.raises(ContractError):
        validate("query_graph", query)


@pytest.mark.parametrize("failure", ["empty", "frame", "pts", "source", "nan", "missing_payload", "dangling"])
def test_result_invalid(result, failure):
    if failure == "empty":
        result["moment_end_seconds"] = result["moment_start_seconds"]
    elif failure == "frame":
        result["representative_original_frame_index"] = 999
    elif failure == "pts":
        result["timing_metadata"]["representative_pts"] += 10
    elif failure == "source":
        result["timing_metadata"]["source_end_seconds"] = 999
    elif failure == "nan":
        result["ranking_score"] = float("nan")
    elif failure == "missing_payload":
        del result["evidence"][0]["roles"]
    elif failure == "dangling":
        result["evidence"][-2]["left"] = "absent"
    with pytest.raises(ValueError):
        validate("result", result)


def test_vfr_half_open_and_duplicate_pts():
    times = [Fraction(x, 1000) for x in [0, 40, 130, 130, 140]]
    assert frame_endpoints(times, Fraction(13, 100), Fraction(14, 100)) == (2, 3)
    assert microseconds(Fraction(1, 2_000_000)) == 0
    assert microseconds(Fraction(3, 2_000_000)) == 2
    with pytest.raises(ValueError):
        frame_endpoints(times, Fraction(1), Fraction(2))


def test_frame_arithmetic_and_origin():
    frame = dict(video_id="v", source_video_id="s", timeline_revision="t", epoch_id="e",
                 original_decoded_frame_index=2, presentation_timestamp=1330, best_effort_timestamp=None,
                 time_base={"num": 1, "den": 1000}, timestamp_us=130000, timestamp_seconds=.13,
                 duration_us=None, pts_occurrence=0, source_offset_seconds=210,
                 source_timestamp_us=210130000, source_original_frame_index=None, sampled_frame_id=None,
                 scene_id=None, sampling_reason=[], image_dimensions={"width": 64, "height": 48},
                 fingerprint=None, codec_key_frame=False, timestamp_quality="exact_pts", decoder_recipe="test")
    validate("frame_record", frame, origin=Fraction(6, 5))
    with pytest.raises(ContractError):
        validate("frame_record", frame)
    frame["timestamp_us"] = 140000
    with pytest.raises(ContractError):
        validate("frame_record", frame, origin=Fraction(6, 5))


def test_packaged_schemas_match_blueprint():
    for path in (ROOT / "docs/schemas").glob("*.schema.json"):
        assert json.loads(path.read_text("utf-8")) == json.loads(
            (ROOT / "src/videntra/contracts/schemas" / path.name).read_text("utf-8"))


def evidence():
    return dict(evidence_id="obs", video_id="v", source_video_id="s", epoch_id="e",
                timeline_revision="t", recipe_hash="a"*64, model_id="ocr", model_revision="r",
                input_hashes=["b"*64], generator_type="model", created_at="2026-11-05T00:00:00Z",
                raw_scores={"recognition": .8}, score_semantics="uncalibrated", quality_flags=[],
                status="ready", kind="OCRObservation", frame_id="f", timestamp_us=0,
                polygon_px=[[0,0],[10,0],[10,10],[0,10]], text_original="SALE", text_nfc="SALE",
                language_hypotheses=["en"], detector_revision="d", recognizer_revision="r",
                crop_hash="c"*64)


def test_evidence_requires_kind_payload_and_refs():
    record = evidence()
    validate_evidence(record, known_refs={"f"}, frame_dimensions=(64,48))
    with pytest.raises(ValueError):
        validate_evidence(record, known_refs=set())
    record["polygon_px"][1][0] = 100
    with pytest.raises(ValueError):
        validate_evidence(record, known_refs={"f"}, frame_dimensions=(64,48))


@pytest.mark.parametrize("mutation", ["human", "missing_model", "invented_interval", "wrong_type", "invalid_date"])
def test_evidence_rejects_invalid_provenance(mutation):
    record = evidence()
    if mutation == "human":
        record["generator_type"] = "human"
    elif mutation == "missing_model":
        record["model_revision"] = None
    elif mutation == "invented_interval":
        record["end_us"] = 100
    elif mutation == "wrong_type":
        record["timestamp_us"] = "zero"
    elif mutation == "invalid_date":
        record["created_at"] = "yesterday"
    with pytest.raises(ValueError):
        validate_evidence(record, known_refs={"f"})

