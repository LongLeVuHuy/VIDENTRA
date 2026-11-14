"""Typed archival evidence and deliberately smaller API projections."""
import unicodedata
import re
from datetime import datetime
from jsonschema import Draft202012Validator
from .validation import require, finite, ContractError
from .timing import interval

S = {"type": "string", "minLength": 1}
I = {"type": "integer"}
N = {"type": "number"}
A = {"type": "array"}
O = {"type": "object"}
REFS = {"type": "array", "items": S, "uniqueItems": True}
STATE = {"enum": ["supported", "contradicted", "unknown"]}
COMMON = {
    **{k: S for k in ("evidence_id", "video_id", "source_video_id", "epoch_id", "timeline_revision",
                      "score_semantics")},
    "recipe_hash": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
    "model_id": {"type": ["string", "null"]}, "model_revision": {"type": ["string", "null"]},
    "input_hashes": {"type": "array", "items": {"type": "string", "pattern": "^[a-f0-9]{64}$"}},
    "generator_type": {"enum": ["model", "rule", "human"]},
    "created_at": {"type": "string", "format": "date-time"},
    "raw_scores": {"type": "object", "additionalProperties": N},
    "quality_flags": REFS,
    "status": {"enum": ["ready", "failed", "missing", "not_run", "empty_observed"]},
}
PAYLOADS = {
    "FrameObservation": {"frame_id": S, "original_decoded_frame_index": {"type": "integer", "minimum": 0},
                         "timestamp_us": I, "dimensions": O, "fingerprint": S},
    "OCRObservation": {"frame_id": S, "timestamp_us": I, "polygon_px": A, "text_original": S,
                       "text_nfc": S, "language_hypotheses": A, "detector_revision": S,
                       "recognizer_revision": S, "crop_hash": S},
    "OCRTrack": {"observation_ids": REFS, "first_observed_us": I, "last_observed_us": I,
                 "association_method": S, "association_scores": A, "holes": A,
                 "persistence_kind": {"enum": ["instant", "tracked_interval", "tracked_with_gaps"]},
                 "boundary_uncertainty_us": {"type": "integer", "minimum": 0}},
    "ASRSegment": {"start_us": I, "end_us": I, "text_original": S, "text_nfc": S, "language": S,
                   "chunk_id": S, "audio_sample_start": I, "audio_sample_end": I, "timestamp_method": S,
                   "word_records": A, "speaker_cluster": {"type": ["string", "null"]}},
    "EntityTrack": {"track_id": S, "entity_type": S, "frame_boxes": A, "attribute_observations": A,
                    "track_links": A, "identity_confidence": {"type": ["number", "null"]}},
    "EventHypothesis": {"event_id": S, "predicate": S, "roles": {"type": "object", "additionalProperties": S},
                        "start_us": I, "end_us": I, "support_refs": REFS, "assessment_state": STATE,
                        "method": S, "boundary_uncertainty": O},
    "RelationAssessment": {"query_predicate_id": S, "event_or_track_refs": REFS, "state": STATE,
                           "features": O, "reason_codes": REFS, "support_refs": REFS, "contradict_refs": REFS},
    "SpeakerBinding": {"speaker_cluster": S, "actor_track_id": S, "start_us": I, "end_us": I,
                        "audio_visual_model_revision": S, "support_frame_refs": REFS, "sync_error_us": I,
                        "state": STATE},
    "CandidateMoment": {"candidate_id": S, "seed_branch_ids": REFS, "start_us": I, "end_us": I,
                        "proposal_method": S, "seed_evidence_ids": REFS,
                        "original_query_rank": {"type": ["integer", "null"], "minimum": 1}},
}


def validate_evidence(record, *, known_refs: set[str], frame_dimensions=None, production=True):
    from jsonschema import FormatChecker, ValidationError
    finite(record)
    kind = record.get("kind")
    require(kind in PAYLOADS, "Unknown archival evidence kind; FinalMoment uses result contract")
    props = {**COMMON, "kind": {"const": kind}, **PAYLOADS[kind]}
    try:
        Draft202012Validator({"type": "object", "properties": props, "required": list(props),
                              "additionalProperties": False}, format_checker=FormatChecker()).validate(record)
    except ValidationError as exc:
        raise ContractError(str(exc)) from exc
    require(not production or record["generator_type"] != "human", "Human labels forbidden in runtime evidence")
    require(bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})",
                              record["created_at"])), "Expected RFC3339 creation timestamp")
    datetime.fromisoformat(record["created_at"])
    if record["generator_type"] == "model":
        require(bool(record["model_id"]) and bool(record["model_revision"]), "Model provenance missing")
    if "start_us" in record:
        interval(record["start_us"], record["end_us"])
    if "text_nfc" in record:
        require(record["text_nfc"] == unicodedata.normalize("NFC", record["text_original"]), "Incorrect NFC text")
    refs = []
    for key in ("support_refs", "contradict_refs", "observation_ids", "support_frame_refs",
                "event_or_track_refs", "seed_evidence_ids"):
        refs += record.get(key, [])
    for key in ("frame_id", "actor_track_id"):
        if key in record:
            refs.append(record[key])
    refs += list(record.get("roles", {}).values())
    require(all(ref in known_refs for ref in refs), "Dangling evidence reference")
    if kind == "OCRObservation":
        polygon(record["polygon_px"], frame_dimensions)
    if kind == "OCRTrack":
        track(record)
    if kind == "ASRSegment":
        interval(record["audio_sample_start"], record["audio_sample_end"])
        for word in record["word_records"]:
            require(set(word) == {"text", "start_us", "end_us"} and isinstance(word["text"], str), "Invalid word record")
            require((word["start_us"] is None) == (word["end_us"] is None), "Partial word timing")
            if word["start_us"] is not None:
                interval(word["start_us"], word["end_us"])
                require(record["start_us"] <= word["start_us"] < word["end_us"] <= record["end_us"], "Word outside segment")
    if kind == "SpeakerBinding" and record["state"] == "supported":
        require(bool(record["support_frame_refs"]), "Speaker binding needs visual support")


def polygon(points, dimensions=None):
    require(isinstance(points, list) and len(points) >= 4, "Polygon requires at least four points")
    for point in points:
        require(isinstance(point, list) and len(point) == 2 and all(type(v) in (int, float) for v in point),
                "Invalid polygon point")
        require(all(v >= 0 for v in point), "Negative polygon coordinate")
        if dimensions:
            require(point[0] <= dimensions[0] and point[1] <= dimensions[1], "Polygon outside source frame")


def track(record):
    refs = record["observation_ids"]
    require(len(refs) == len(set(refs)) and len(refs) >= 1, "Invalid OCR observations")
    first, last = record["first_observed_us"], record["last_observed_us"]
    if record["persistence_kind"] == "instant":
        require(len(refs) == 1 and first == last, "Singleton OCR must be instant")
    else:
        require(len(refs) >= 2 and first < last, "Tracked OCR needs multiple observations")


def validate_projection(records):
    required = {
        "frame": {"original_decoded_frame_index", "timestamp_us"},
        "event": {"start_us", "end_us", "roles", "predicate", "state"},
        "asr_segment": {"start_us", "end_us", "text_original", "timestamp_method"},
        "speaker_binding": {"start_us", "end_us", "speaker_cluster", "actor_track", "state", "basis"},
        "ocr_observation": {"timestamp_us", "text_original", "polygon", "original_decoded_frame_index"},
        "ocr_track": {"first_observed_us", "last_observed_us", "observation_ids", "persistence_kind"},
        "relation": {"predicate", "state"},
    }
    ids = {x["evidence_id"] for x in records}
    require(len(ids) == len(records), "Duplicate result evidence ID")
    for rec in records:
        kind = rec["kind"]
        require(kind in required and required[kind] <= rec.keys(), "Incomplete evidence projection")
        for key in ("timestamp_us", "start_us", "end_us", "first_observed_us", "last_observed_us",
                    "original_decoded_frame_index"):
            if key in rec:
                require(type(rec[key]) is int, "Evidence time/frame fields must be integers")
        for key in ("text_original", "predicate", "timestamp_method", "speaker_cluster", "actor_track", "basis"):
            if key in rec:
                require(isinstance(rec[key], str) and bool(rec[key]), "Invalid evidence text field")
        if "roles" in rec:
            require(isinstance(rec["roles"], dict) and bool(rec["roles"])
                    and all(isinstance(v, str) and v for v in rec["roles"].values()), "Invalid event roles")
        if "start_us" in rec:
            interval(rec["start_us"], rec["end_us"])
        if "state" in rec:
            require(rec["state"] in STATE["enum"], "Invalid assessment state")
        if kind == "ocr_observation":
            require("end_us" not in rec, "An OCR observation is an instant")
            polygon(rec["polygon"])
        if kind == "ocr_track":
            track(rec)
            require(all(ref in ids for ref in rec["observation_ids"]), "Dangling OCR observation")
        if kind == "relation" and rec["predicate"] in ("before", "after", "during", "overlap"):
            require(rec.get("left") in ids and rec.get("right") in ids, "Dangling temporal relation")

