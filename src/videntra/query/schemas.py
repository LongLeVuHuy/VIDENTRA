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
