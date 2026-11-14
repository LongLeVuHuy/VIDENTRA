"""Shape and cross-field validation for the normative blueprint contracts."""
from functools import lru_cache
from fractions import Fraction
from importlib.resources import files
import json
import math

from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from .timing import interval, microseconds


class ContractError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Non-finite numeric value")
    elif isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)


@lru_cache
def schema_validator(kind):
    allowed = {"media_asset", "frame_record", "query_graph", "result", "dataset"}
    require(kind in allowed, "Unknown contract")
    schema = json.loads(files("videntra.contracts").joinpath("schemas", kind + ".schema.json").read_text("utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def acyclic(edges: dict[str, list[str]]) -> None:
    active, visited = set(), set()

    def visit(node):
        require(node not in active, "Cyclic references")
        if node in visited:
            return
        active.add(node)
        for child in edges.get(node, []):
            visit(child)
        active.remove(node)
        visited.add(node)

    for node in edges:
        visit(node)


def validate(kind: str, record: dict, *, origin: Fraction | None = None, production: bool = False) -> None:
    finite(record)
    try:
        schema_validator(kind).validate(record)
        if kind == "media_asset":
            require(bool(record["video_id"]) and bool(record["source_video_id"]), "Empty identity")
            require(record["source_mapping_kind"] != "constant_offset" or record["source_offset_us"] is not None,
                    "Constant mapping needs verified source offset")
            require(record["source_mapping_kind"] == "constant_offset" or record["source_offset_us"] is None,
                    "Offset cannot stand in for unknown/piecewise mapping")
            require(record["duration_us"] is None or record["duration_us"] > 0, "Invalid media duration")
            require(record["rights_status"] != "allowed" or bool(record["rights_reference"]), "Allowed rights need receipt")
            require(record["decode_status"] != "ready" or bool(record["timeline_revision"]), "Ready decode needs timeline")
            for field in ("video_stream_index", "audio_stream_index"):
                require(record[field] is None or record[field] >= 0, "Invalid stream index")
        elif kind == "frame_record":
            quality = record["timestamp_quality"]
            pts = record["presentation_timestamp"] if quality == "exact_pts" else record["best_effort_timestamp"]
            if quality == "missing":
                require(record["timestamp_us"] is None and record["timestamp_seconds"] is None, "Missing PTS must not fabricate time")
            elif quality in ("exact_pts", "best_effort"):
                require(pts is not None and record["timestamp_us"] is not None, "Timestamp quality lacks evidence")
                require(origin is not None, "Frame arithmetic validation requires shared origin")
                base = record["time_base"]
                seconds = Fraction(pts * base["num"], base["den"]) - origin
                require(record["timestamp_us"] == microseconds(seconds), "Frame PTS conversion mismatch")
                require(record["timestamp_seconds"] is not None and abs(record["timestamp_seconds"] - float(seconds)) <= 1e-6,
                        "Frame display timestamp mismatch")
            offset, source = record["source_offset_seconds"], record["source_timestamp_us"]
            if offset is not None:
                require(record["timestamp_us"] is not None and source == record["timestamp_us"] + microseconds(Fraction(str(offset))),
                        "Source offset mismatch")
            require(not record["sampling_reason"] or record["sampled_frame_id"] is not None, "Sample reason needs sample ID")
        elif kind == "query_graph":
            validate_graph(record)
        elif kind == "result":
            validate_result(record, production)
        elif kind == "dataset":
            require(len(record["media_ids"]) == len(set(record["media_ids"])), "Duplicate dataset members")
            require(record["rights_status"] != "allowed" or bool(record["rights_reference"]), "Dataset rights receipt missing")
    except (ValidationError, ValueError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError(str(exc)) from exc


def validate_graph(q):
    entities = {x["id"]: x for x in q["entities"]}
    predicates = {x["id"]: x for x in q["predicates"]}
    logic = {x["id"]: x for x in q["logic"]}
    ids = set(entities) | set(predicates) | set(logic)
    require(len(ids) == len(q["entities"]) + len(q["predicates"]) + len(q["logic"]), "Duplicate graph IDs")
    require(q["root_id"] in predicates or q["root_id"] in logic, "Unknown graph root")
    coref, ordering = {}, {}
    for key, entity in entities.items():
        target = entity["coreference_of"]
        require(target is None or target in entities, "Missing coreference")
        coref[key] = [] if target is None else [target]
        require(entity["mention"] in q["original_query"], "Entity mention not in query")
        for attr in entity["attributes"]:
            require(attr["source_span"] in q["original_query"], "Attribute not grounded in original query")
    acyclic(coref)
    for pred in predicates.values():
        require(pred["source_span"] in q["original_query"], "Predicate span not in original query")
        for role, target in pred["roles"].items():
            require(target in (predicates if role in ("event_a", "event_b") else entities), "Dangling role")
        scope = pred["scope"]
        if scope:
            anchor = scope["anchor_predicate_id"]
            require(anchor is None or anchor in predicates, "Dangling scope")
            lo, hi = scope["min_gap_ms"], scope["max_gap_ms"]
            require(lo is None or hi is None or lo <= hi, "Invalid gap bounds")
        if pred["kind"] == "temporal" and pred["operator"] in ("before", "after"):
            roles = pred["roles"]
            require("event_a" in roles and "event_b" in roles, "Ordering requires both events")
            a, b = roles["event_a"], roles["event_b"]
            if pred["operator"] == "after":
                a, b = b, a
            ordering.setdefault(a, []).append(b)
    acyclic(ordering)
    edges = {}
    for key, node in logic.items():
        require(all(x in predicates or x in logic for x in node["children"]), "Dangling logic child")
        require(node["op"] != "NOT" or len(node["children"]) == 1, "NOT requires exactly one operand")
        edges[key] = node["children"]
    acyclic(edges)


def validate_result(record, production):
    start, end = record["moment_start_seconds"], record["moment_end_seconds"]
    interval(start, end)
    require(not production or not record["provenance"]["synthetic"], "Synthetic results forbidden in production")
    first, last, representative = (record[k] for k in (
        "start_original_frame_index", "end_original_frame_index", "representative_original_frame_index"))
    require((first is None) == (last is None), "Frame endpoints must be paired")
    if first is not None:
        require(0 <= first <= last, "Invalid frame endpoints")
        require(representative is None or first <= representative <= last, "Representative outside frame interval")
    t = record["representative_timestamp_seconds"]
    require(t is None or start <= t < end, "Representative outside moment")
    timing = record["timing_metadata"]
    pts, origin = timing["representative_pts"], timing["first_video_pts"]
    if pts is not None and origin is not None:
        base = timing["time_base"]
        require(base["num"] > 0, "Nonpositive time base")
        require(t is not None and abs(float(Fraction((pts-origin)*base["num"], base["den"])) - t) <= 1e-6,
                "Representative PTS mismatch")
    offset = timing["source_offset_seconds"]
    if offset is not None:
        for bound in ("start", "end"):
            source = timing["source_" + bound + "_seconds"]
            require(source is not None and abs(source - (record["moment_" + bound + "_seconds"] + offset)) <= 1e-6,
                    "Source conversion mismatch")
    from .evidence import validate_projection
    validate_projection(record["evidence"])

