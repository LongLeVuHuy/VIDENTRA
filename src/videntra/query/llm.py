"""Small LLM query parser with strict JSON schema validation and zero-crash rule fallback."""
import json
from typing import Any, Callable

from videntra.contracts import validate
from .schemas import (
    QueryGraph,
    Entity,
    EntityAttribute,
    Predicate,
    PredicateScope,
    LogicNode,
    ParserMetadata,
)
from .rules import RuleBasedQueryParser
from .cache import QueryGraphCache


QG_SYSTEM_PROMPT = """Bạn là trợ lý phân tích cú pháp truy vấn video VIDENTRA.
Nhiệm vụ: Chuyển câu hỏi ngôn ngữ tự nhiên tiếng Việt hoặc tiếng Anh thành QueryGraph JSON v1.
Quy tắc:
1. Trích xuất entities (person, object, text_region).
2. Trích xuất predicates (action, ocr, asr, spatial, temporal).
3. Nếu màu là 'xanh', ghi rõ ambiguity không chắc chắn giữa blue/green.
4. Chỉ trả về duy nhất chuỗi JSON hợp lệ, không giải thích ngoài JSON.
"""


class LLMQueryParser:
    """Parses natural queries via small LLM with guaranteed fallback to rules on schema deviation."""

    def __init__(
        self,
        llm_generate_fn: Callable[[str], str] | None = None,
        model_revision: str = "qwen2.5_3b_instruct",
        cache: QueryGraphCache | None = None,
    ):
        self.llm_generate_fn = llm_generate_fn
        self.model_revision = model_revision
        self.rule_parser = RuleBasedQueryParser()
        self.cache = cache or QueryGraphCache()

    def parse(self, query: str) -> QueryGraph:
        """Parse query, attempting LLM inference first, falling back to rule parser on failure."""
        cached = self.cache.get(query, self.model_revision)
        if cached is not None:
            return cached

        # If no LLM callable configured, directly use rules
        if self.llm_generate_fn is None:
            graph = self.rule_parser.parse(query)
            self.cache.put(query, graph, self.model_revision)
            return graph

        try:
            # 1. Execute LLM call
            raw_response = self.llm_generate_fn(query)

            # 2. Extract JSON block if wrapped in markdown fences
            json_str = raw_response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()

            data = json.loads(json_str)

            # 3. Validate against official JSON schema
            validate("query_graph", data)

            # Reconstruct typed QueryGraph
            entities = [
                Entity(
                    id=e["id"],
                    type=e["type"],
                    mention=e["mention"],
                    attributes=[EntityAttribute(**a) for a in e.get("attributes", [])],
                    coreference_of=e.get("coreference_of"),
                    uncertainty=e.get("uncertainty", []),
                )
                for e in data["entities"]
            ]

            predicates = [
                Predicate(
                    id=p["id"],
                    kind=p["kind"],
                    operator=p["operator"],
                    roles=p["roles"],
                    value=p.get("value"),
                    source_span=p.get("source_span", ""),
                    polarity=p.get("polarity", "positive"),
                    modality_options=p.get("modality_options", [["visual"]]),
                    scope=PredicateScope(**p["scope"]) if p.get("scope") else None,
                    confidence=p.get("confidence"),
                    uncertainty=p.get("uncertainty", []),
                )
                for p in data["predicates"]
            ]

            logic = [
                LogicNode(
                    id=l["id"],
                    op=l["op"],
                    children=l["children"],
                )
                for l in data["logic"]
            ]

            graph = QueryGraph(
                original_query=data["original_query"],
                language=data["language"],
                entities=entities,
                predicates=predicates,
                logic=logic,
                root_id=data["root_id"],
                ambiguities=data.get("ambiguities", []),
                residual_constraints=data.get("residual_constraints", []),
                parser=ParserMetadata(
                    status="validated",
                    model_revision=self.model_revision,
                    prompt_version="qg-v1",
                    confidence_kind="uncalibrated",
                ),
            )
            self.cache.put(query, graph, self.model_revision)
            return graph

        except Exception:
            # Zero-crash contract: Fallback seamlessly to rule-based parser
            graph = self.rule_parser.parse(query)
            # Mark status as fallback
            graph.parser = ParserMetadata(
                status="fallback",
                model_revision=self.model_revision,
                prompt_version="qg-v1",
                confidence_kind="uncalibrated",
            )
            self.cache.put(query, graph, self.model_revision)
            return graph
