"""Typed structures for VIDENTRA QueryGraph v1."""
from dataclasses import dataclass, field, asdict
from typing import Any, Sequence

from videntra.contracts import validate


@dataclass
class EntityAttribute:
    name: str
    value: str
    source_span: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Entity:
    id: str
    type: str  # "person", "object", "text_region", "vehicle", "location", "animal"
    mention: str
    attributes: list[EntityAttribute] = field(default_factory=list)
    coreference_of: str | None = None
    uncertainty: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["attributes"] = [a.to_dict() if isinstance(a, EntityAttribute) else a for a in self.attributes]
        return d


@dataclass
class PredicateScope:
    anchor_predicate_id: str | None = None
    min_gap_ms: int | None = None
    max_gap_ms: int | None = None
    quantifier: str = "exists"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Predicate:
    id: str
    kind: str  # "action", "ocr", "asr", "spatial", "temporal", "attribute", "state"
    operator: str
    roles: dict[str, str]
    value: str | None = None
    source_span: str = ""
    polarity: str = "positive"
    modality_options: list[list[str]] = field(default_factory=lambda: [["visual"]])
    scope: PredicateScope | None = None
    confidence: float | None = None
    uncertainty: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.scope is not None:
            d["scope"] = self.scope.to_dict() if isinstance(self.scope, PredicateScope) else self.scope
        return d


@dataclass
class LogicNode:
    id: str
    op: str  # "AND", "OR", "NOT"
    children: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParserMetadata:
    status: str  # "validated" or "fallback"
    model_revision: str | None = None
    prompt_version: str = "qg-v1"
    confidence_kind: str = "uncalibrated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryGraph:
    original_query: str
    language: str
    entities: list[Entity]
    predicates: list[Predicate]
    logic: list[LogicNode]
    root_id: str
    ambiguities: list[str] = field(default_factory=list)
    residual_constraints: list[str] = field(default_factory=list)
    parser: ParserMetadata = field(default_factory=lambda: ParserMetadata(status="validated"))
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "original_query": self.original_query,
            "language": self.language,
            "entities": [e.to_dict() if isinstance(e, Entity) else e for e in self.entities],
            "predicates": [p.to_dict() if isinstance(p, Predicate) else p for p in self.predicates],
            "logic": [l.to_dict() if isinstance(l, LogicNode) else l for l in self.logic],
            "root_id": self.root_id,
            "ambiguities": self.ambiguities,
            "residual_constraints": self.residual_constraints,
            "parser": self.parser.to_dict() if isinstance(self.parser, ParserMetadata) else self.parser,
        }

    def validate_schema(self) -> None:
        """Validate structure against official JSON schema."""
        validate("query_graph", self.to_dict())


@dataclass(frozen=True)
class CompactDecomposedQuery:
    """Compact Wire JSON structure (~60-100 tokens) emitted by Small-LLM before deterministic expansion."""

    lang: str = "unknown"
    events: tuple[dict[str, Any], ...] = ()
    spatial: tuple[dict[str, Any], ...] = ()
    temporal: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompactDecomposedQuery":
        return cls(
            lang=str(data.get("lang", "unknown")),
            events=tuple(dict(e) for e in (data.get("events") or []) if isinstance(e, dict)),
            spatial=tuple(dict(s) for s in (data.get("spatial") or []) if isinstance(s, dict)),
            temporal=tuple(dict(t) for t in (data.get("temporal") or []) if isinstance(t, dict)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lang": self.lang,
            "events": [dict(e) for e in self.events],
            "spatial": [dict(s) for s in self.spatial],
            "temporal": [dict(t) for t in self.temporal],
        }


@dataclass(frozen=True)
class ModalityRoutingPlan:
    """Structured modality routing plan derived from QueryGraph for downstream retrieval stages."""

    original_query: str
    query_length_tier: str  # "short", "medium", "long"
    visual_clauses: tuple[str, ...]
    ocr_phrases: tuple[str, ...]
    asr_phrases: tuple[str, ...]
    object_requirements: tuple[dict[str, Any], ...]
    has_explicit_ocr_intent: bool
    has_explicit_asr_intent: bool
    has_spatial_constraints: bool
    has_temporal_constraints: bool
    query_graph: QueryGraph
    routing_mode: str = "fast_rule"  # "fast_rule", "tri_modal_broadcast", "llm_compact", "fallback"
    cache_hit: bool = False
    decomposer_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "query_length_tier": self.query_length_tier,
            "visual_clauses": list(self.visual_clauses),
            "ocr_phrases": list(self.ocr_phrases),
            "asr_phrases": list(self.asr_phrases),
            "object_requirements": [dict(o) for o in self.object_requirements],
            "has_explicit_ocr_intent": self.has_explicit_ocr_intent,
            "has_explicit_asr_intent": self.has_explicit_asr_intent,
            "has_spatial_constraints": self.has_spatial_constraints,
            "has_temporal_constraints": self.has_temporal_constraints,
            "query_graph": self.query_graph.to_dict(),
            "routing_mode": self.routing_mode,
            "cache_hit": self.cache_hit,
            "decomposer_latency_ms": self.decomposer_latency_ms,
        }
