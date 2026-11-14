"""Unit tests for Phase 7.2 Small-LLM Query Decomposer, Modality Router, and Semantic Validator."""
import json
import time
from pathlib import Path

import pytest

from videntra.contracts import validate
from videntra.query.llm import (
    AdaptiveQueryDecomposer,
    LocalSmallLLMBackend,
    QueryComplexityGate,
    SemanticDecomposerValidator,
    expand_compact_wire_to_query_graph,
)
from videntra.query.schemas import CompactDecomposedQuery, ModalityRoutingPlan


# ---------------------------------------------------------------------------
# 1. QueryComplexityGate (Short / Medium / Long + Tri-Modal Broadcast)
# ---------------------------------------------------------------------------


def test_complexity_gate_short_visual_vs_tri_modal():
    gate = QueryComplexityGate()

    # Simple visual short queries (1-4 words, no proper noun/number/acronym/modal cue)
    d1 = gate.classify("người đi xe đạp")
    assert d1.tier == "short"
    assert d1.routing_mode == "fast_rule"

    d2 = gate.classify("a dog catching frisbee")
    assert d2.tier == "short"
    assert d2.routing_mode == "fast_rule"

    # Short queries with proper noun, number/percentage, or acronym -> Tri-Modal Broadcast
    d3 = gate.classify("Gordon Ramsay")
    assert d3.tier == "short"
    assert d3.routing_mode == "tri_modal_broadcast"

    d4 = gate.classify("chữ SALE 50%")
    assert d4.tier == "short"
    assert d4.routing_mode == "tri_modal_broadcast"


def test_complexity_gate_medium_and_long():
    gate = QueryComplexityGate()

    medium = gate.classify("Người phụ nữ mặc áo đỏ đứng bên trái chiếc ô tô")
    assert medium.tier == "medium"
    assert medium.routing_mode == "llm_compact"

    long_words = gate.classify(
        "Đầu tiên một đầu bếp đang thái hành tây trên thớt gỗ rồi cho hành vào chảo xào "
        "sau đó anh ấy cầm đĩa thức ăn đặt lên bàn"
    )
    assert long_words.tier == "long"
    assert long_words.routing_mode == "llm_compact"

    # >= 2 temporal clauses also triggers long tier even if <= 18 words
    multi_temporal = gate.classify("Người đàn ông tắt đèn sau khi đóng cửa rồi ngồi xuống ghế")
    assert multi_temporal.tier == "long"
    assert multi_temporal.temporal_clause_count >= 2


# ---------------------------------------------------------------------------
# 2. Short Query Routing in AdaptiveQueryDecomposer (Vi & En)
# ---------------------------------------------------------------------------


def test_short_visual_query_uses_fast_rule_without_calling_llm(tmp_path: Path):
    llm_called = False

    def fail_if_called(_prompt: str) -> str:
        nonlocal llm_called
        llm_called = True
        return "{}"

    backend = LocalSmallLLMBackend(generate_fn=fail_if_called)
    decomposer = AdaptiveQueryDecomposer(
        llm_backend=backend,
        cache_path=tmp_path / "cache.jsonl",
        enable_disk_cache=False,
    )

    plan = decomposer.decompose("a dog catching frisbee")
    assert isinstance(plan, ModalityRoutingPlan)
    assert llm_called is False
    assert plan.query_length_tier == "short"
    assert plan.routing_mode == "fast_rule"
    assert plan.visual_clauses == ("a dog catching frisbee",)
    assert plan.ocr_phrases == ()
    assert plan.asr_phrases == ()
    assert plan.has_explicit_ocr_intent is False
    assert plan.has_explicit_asr_intent is False
    validate("query_graph", plan.query_graph.to_dict())


def test_short_proper_noun_triggers_tri_modal_broadcast(tmp_path: Path):
    decomposer = AdaptiveQueryDecomposer(
        llm_backend=LocalSmallLLMBackend(enabled=False),
        cache_path=tmp_path / "cache.jsonl",
        enable_disk_cache=False,
    )
    plan = decomposer.decompose("Gordon Ramsay")
    assert plan.query_length_tier == "short"
    assert plan.routing_mode == "tri_modal_broadcast"
    assert "Gordon Ramsay" in plan.visual_clauses
    assert "Gordon Ramsay" in plan.ocr_phrases
    assert "Gordon Ramsay" in plan.asr_phrases
    assert plan.has_explicit_ocr_intent is True
    assert plan.has_explicit_asr_intent is True


# ---------------------------------------------------------------------------
# 3. Medium Query Compact Wire Expansion & Modality Routing
# ---------------------------------------------------------------------------


def test_medium_query_compact_wire_decomposition_vi_en(tmp_path: Path):
    query = (
        "Người phụ nữ mặc áo đỏ bên trái ô tô nói thời tiết có chữ BREAKING NEWS"
    )
    compact_wire = {
        "lang": "vi",
        "events": [
            {
                "id": "e1",
                "vis": "woman in red shirt standing left of a car",
                "ocr": ["BREAKING NEWS"],
                "asr": ["thời tiết", "weather"],
                "objs": [
                    {"id": "p1", "cls": "phụ nữ", "cnt": 1, "color": "đỏ", "span": "Người phụ nữ mặc áo đỏ"},
                    {"id": "o1", "cls": "ô tô", "cnt": 1, "color": "", "span": "ô tô"},
                ],
                "pol": "pos",
            }
        ],
        "spatial": [{"subj": "p1", "op": "left_of", "obj": "o1", "pol": "pos"}],
        "temporal": [],
    }

    backend = LocalSmallLLMBackend(generate_fn=lambda _q: f"```json\n{json.dumps(compact_wire)}\n```")
    decomposer = AdaptiveQueryDecomposer(
        llm_backend=backend,
        cache_path=tmp_path / "cache.jsonl",
        enable_disk_cache=False,
    )

    plan = decomposer.decompose(query)
    assert plan.query_length_tier == "medium"
    assert plan.routing_mode == "llm_compact"
    assert plan.query_graph.parser.status == "validated"
    assert plan.visual_clauses == ("woman in red shirt standing left of a car",)
    assert plan.ocr_phrases == ("BREAKING NEWS",)
    assert set(plan.asr_phrases) == {"thời tiết", "weather"}
    assert plan.has_explicit_ocr_intent is True
    assert plan.has_explicit_asr_intent is True
    assert plan.has_spatial_constraints is True
    assert plan.has_temporal_constraints is False
    assert any(obj["class"] == "person" and obj["color"] == "red" for obj in plan.object_requirements)
    assert any(obj["class"] == "car" for obj in plan.object_requirements)
    validate("query_graph", plan.query_graph.to_dict())


# ---------------------------------------------------------------------------
# 4. Long Query & Temporal Inversion ("sau khi", "trước khi", "after")
# ---------------------------------------------------------------------------


def test_long_query_temporal_inversion_chronological_order(tmp_path: Path):
    query = (
        "Người đàn ông mở cửa phòng sau khi tắt đèn và trước khi bước ra ngoài "
        "thì anh ấy quay lại nói tạm biệt với mọi người"
    )
    # Events listed in surface order (open door -> turn off light -> step out),
    # while temporal edges specify actual chronological order: e2 -> e1 -> e3.
    compact_wire = {
        "lang": "vi",
        "events": [
            {"id": "e1", "vis": "man opening the door", "ocr": [], "asr": [], "objs": [], "pol": "pos"},
            {"id": "e2", "vis": "man turning off the light", "ocr": [], "asr": [], "objs": [], "pol": "pos"},
            {"id": "e3", "vis": "man stepping outside saying goodbye", "ocr": [], "asr": ["tạm biệt"], "objs": [], "pol": "pos"},
        ],
        "spatial": [],
        "temporal": [
            {"first": "e1", "op": "after", "second": "e2"},   # Inverted: e2 before e1
            {"first": "e1", "op": "before", "second": "e3"},
        ],
    }

    backend = LocalSmallLLMBackend(generate_fn=lambda _q: json.dumps(compact_wire))
    decomposer = AdaptiveQueryDecomposer(
        llm_backend=backend,
        cache_path=tmp_path / "cache.jsonl",
        enable_disk_cache=False,
    )

    plan = decomposer.decompose(query)
    assert plan.query_length_tier == "long"
    assert plan.has_temporal_constraints is True
    assert plan.visual_clauses == (
        "man turning off the light",
        "man opening the door",
        "man stepping outside saying goodbye",
    )
    temporal_preds = [p for p in plan.query_graph.predicates if p.kind == "temporal"]
    assert len(temporal_preds) == 2
    assert all(p.operator == "before" for p in temporal_preds)
    assert temporal_preds[0].roles == {"event_a": "e2_vis", "event_b": "e1_vis"}


# ---------------------------------------------------------------------------
# 5. SemanticDecomposerValidator: Anti-Hallucination, Negation, DAG, Color
# ---------------------------------------------------------------------------


def test_validator_filters_hallucinated_ocr_asr_spans():
    validator = SemanticDecomposerValidator()
    query = "Cô gái cầm cốc cà phê và trên biển có chữ OPEN"
    compact = CompactDecomposedQuery.from_dict(
        {
            "lang": "vi",
            "events": [
                {
                    "id": "e1",
                    "vis": "girl holding coffee cup",
                    "ocr": ["OPEN", "SUBSCRIBE NOW"],
                    "asr": ["buy bitcoin", "cà phê"],
                    "objs": [{"id": "p1", "cls": "girl", "cnt": 1}],
                    "pol": "pos",
                }
            ],
        }
    )

    graph, plan = validator.validate_and_expand(compact, original_query=query)
    assert plan.ocr_phrases == ("OPEN",)
    assert plan.asr_phrases == ("cà phê",)
    validate("query_graph", graph.to_dict())


def test_validator_rejects_temporal_dag_cycle_and_decomposer_falls_back(tmp_path: Path):
    validator = SemanticDecomposerValidator()
    cyclic_wire = {
        "lang": "en",
        "events": [
            {"id": "e1", "vis": "person enters room", "ocr": [], "asr": [], "objs": [], "pol": "pos"},
            {"id": "e2", "vis": "person sits down", "ocr": [], "asr": [], "objs": [], "pol": "pos"},
        ],
        "spatial": [],
        "temporal": [
            {"first": "e1", "op": "before", "second": "e2"},
            {"first": "e2", "op": "before", "second": "e1"},
        ],
    }

    with pytest.raises(ValueError, match="cycle"):
        validator.validate_and_expand(cyclic_wire, original_query="person enters room then sits down")

    backend = LocalSmallLLMBackend(generate_fn=lambda _q: json.dumps(cyclic_wire))
    decomposer = AdaptiveQueryDecomposer(
        llm_backend=backend,
        cache_path=tmp_path / "cache.jsonl",
        enable_disk_cache=False,
    )
    plan = decomposer.decompose("person enters room and then sits down on a chair")
    assert plan.routing_mode == "fallback"
    assert plan.query_graph.parser.status == "fallback"
    validate("query_graph", plan.query_graph.to_dict())


def test_validator_preserves_negation_and_xanh_color_ambiguity():
    validator = SemanticDecomposerValidator()
    query = "Người đàn ông mặc áo xanh không đội mũ bảo hiểm"
    wire = {
        "lang": "vi",
        "events": [
            {
                "id": "e1",
                "vis": "man in blue or green shirt without helmet",
                "ocr": [],
                "asr": [],
                "objs": [{"id": "p1", "cls": "person", "cnt": 1, "color": "xanh", "span": "Người đàn ông mặc áo xanh"}],
                "pol": "pos",  # LLM forgot to mark negative polarity
            }
        ],
    }

    graph, _plan = validator.validate_and_expand(wire, original_query=query)
    assert "negation_scope_unresolved" in graph.ambiguities
    assert "blue_or_green" in graph.ambiguities
    assert "blue_or_green" in graph.entities[0].uncertainty
    assert all("negation_scope_unresolved" in p.uncertainty for p in graph.predicates)

    # Negation inside quotes should NOT trigger negation_scope_unresolved
    quoted_query = 'Biển báo trên tường có ghi chữ "không hút thuốc"'
    quoted_wire = {
        "lang": "vi",
        "events": [{"id": "e1", "vis": "sign on wall", "ocr": ["không hút thuốc"], "asr": [], "objs": [], "pol": "pos"}],
    }
    q_graph, _ = validator.validate_and_expand(quoted_wire, original_query=quoted_query)
    assert "negation_scope_unresolved" not in q_graph.ambiguities


# ---------------------------------------------------------------------------
# 6. <1ms Fallback on Disabled / Timed-out LLM & RAM + Disk JSONL Cache
# ---------------------------------------------------------------------------


def test_disabled_llm_fast_fallback_under_5ms(tmp_path: Path):
    backend = LocalSmallLLMBackend(enabled=False)
    decomposer = AdaptiveQueryDecomposer(
        llm_backend=backend,
        cache_path=tmp_path / "cache.jsonl",
        enable_disk_cache=False,
    )

    query = "Người đàn ông mặc áo đỏ đang đi bộ bên cạnh chiếc xe hơi"
    t0 = time.perf_counter()
    plan = decomposer.decompose(query)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert plan.routing_mode == "fallback"
    assert plan.query_graph.parser.status == "fallback"
    assert elapsed_ms < 25.0  # Rule fallback itself is < 1 ms (generous CI margin)
    assert plan.decomposer_latency_ms < 25.0
    validate("query_graph", plan.query_graph.to_dict())


def test_timeout_llm_triggers_zero_crash_fallback(tmp_path: Path):
    def slow_llm(_prompt: str) -> str:
        time.sleep(0.15)
        return "{}"

    backend = LocalSmallLLMBackend(generate_fn=slow_llm, deadline_ms=25)
    decomposer = AdaptiveQueryDecomposer(
        llm_backend=backend,
        cache_path=tmp_path / "cache.jsonl",
        enable_disk_cache=False,
    )

    plan = decomposer.decompose("A woman in a red coat walking past a bakery shop")
    assert plan.routing_mode == "fallback"
    assert plan.query_graph.parser.status == "fallback"
    validate("query_graph", plan.query_graph.to_dict())


def test_ram_and_disk_jsonl_cache_hit(tmp_path: Path):
    cache_file = tmp_path / "query_decomposition_cache.jsonl"
    calls = 0

    def counting_llm(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return json.dumps(
            {
                "lang": "en",
                "events": [
                    {"id": "e1", "vis": "chef slicing onions on wooden board", "ocr": [], "asr": [], "objs": [], "pol": "pos"}
                ],
            }
        )

    d1 = AdaptiveQueryDecomposer(
        llm_backend=LocalSmallLLMBackend(generate_fn=counting_llm),
        cache_path=cache_file,
        enable_disk_cache=True,
    )
    query = "A chef slicing onions on a wooden cutting board"
    p1 = d1.decompose(query)
    assert calls == 1
    assert p1.cache_hit is False
    assert cache_file.exists()

    # Second call on same instance hits RAM cache
    p2 = d1.decompose(query)
    assert calls == 1
    assert p2.cache_hit is True

    # New instance with disabled LLM hits disk JSONL cache
    d2 = AdaptiveQueryDecomposer(
        llm_backend=LocalSmallLLMBackend(enabled=False),
        cache_path=cache_file,
        enable_disk_cache=True,
    )
    p3 = d2.decompose(query)
    assert p3.cache_hit is True
    assert p3.routing_mode == "llm_compact"
    assert p3.visual_clauses == ("chef slicing onions on wooden board",)


def test_fallback_sanitizes_dangling_temporal_only_rule_graph(tmp_path: Path):
    """Regression: queries yielding only a dangling temporal_order predicate in RuleBasedQueryParser must not crash."""
    decomposer = AdaptiveQueryDecomposer(
        llm_backend=LocalSmallLLMBackend(enabled=False),
        cache_path=tmp_path / "cache.jsonl",
        enable_disk_cache=False,
    )
    plan = decomposer.decompose("something happens and then another thing disappears quickly")
    assert plan.routing_mode == "fallback"
    assert plan.query_graph.parser.status == "fallback"
    assert len(plan.query_graph.predicates) >= 1
    validate("query_graph", plan.query_graph.to_dict())


def test_fallback_sanitizes_case_mismatched_entity_mentions(tmp_path: Path):
    """Regression: capitalized/case-mismatched entity mentions from RuleBasedQueryParser must pass QueryGraph v1."""
    decomposer = AdaptiveQueryDecomposer(
        llm_backend=LocalSmallLLMBackend(enabled=False),
        cache_path=tmp_path / "cache.jsonl",
        enable_disk_cache=False,
    )
    # Uppercase/Titlecase nouns can produce lowercase entity.mention in RuleBasedQueryParser
    plan = decomposer.decompose("A MAN IN A RED SHIRT IS TALKING ABOUT FOOTBALL ON STAGE")
    assert plan.routing_mode == "fallback"
    assert plan.query_graph.parser.status == "fallback"
    validate("query_graph", plan.query_graph.to_dict())
