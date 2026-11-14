"""Unit tests for Phase 7.2 Small-LLM Query Decomposer, Modality Router, and Semantic Validator."""
import json
import time
from pathlib import Path

import pytest

from videntra.contracts import validate
from videntra.query.llm import (
    AdaptiveQueryDecomposer,
    DeadlineStoppingCriteria,
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


# ---------------------------------------------------------------------------
# 7. VIDENTRA_LLM_DEADLINE_MS & 1.5B Model Discovery
# ---------------------------------------------------------------------------


def test_deadline_ms_env_var_and_cuda_cpu_defaults(monkeypatch):
    monkeypatch.delenv("VIDENTRA_LLM_DEADLINE_MS", raising=False)

    monkeypatch.setattr(LocalSmallLLMBackend, "_has_cuda", staticmethod(lambda: False))
    cpu_backend = LocalSmallLLMBackend(enabled=False)
    assert cpu_backend.deadline_ms == 5000

    monkeypatch.setattr(LocalSmallLLMBackend, "_has_cuda", staticmethod(lambda: True))
    cuda_backend = LocalSmallLLMBackend(enabled=False)
    assert cuda_backend.deadline_ms == 250

    monkeypatch.setenv("VIDENTRA_LLM_DEADLINE_MS", "1234")
    env_backend = LocalSmallLLMBackend(enabled=False)
    assert env_backend.deadline_ms == 1234

    # Explicit deadline_ms always overrides env var and defaults
    explicit_backend = LocalSmallLLMBackend(enabled=False, deadline_ms=42)
    assert explicit_backend.deadline_ms == 42


def test_discover_model_path_prefers_1_5b_and_reflects_model_revision(tmp_path: Path, monkeypatch):
    models_dir = tmp_path / "models" / "llm"
    models_dir.mkdir(parents=True)
    (models_dir / ".gitkeep").write_text("", encoding="utf-8")
    (models_dir / "qwen2.5-0.5b-instruct").mkdir()
    (models_dir / "qwen2.5-1.5b-instruct").mkdir()

    monkeypatch.delenv("VIDENTRA_LLM_MODEL_PATH", raising=False)
    monkeypatch.setattr(LocalSmallLLMBackend, "DEFAULT_MODELS_DIR", models_dir)

    backend = LocalSmallLLMBackend(enabled=False)
    discovered = backend._discover_model_path()
    assert discovered is not None
    assert discovered.name == "qwen2.5-1.5b-instruct"
    assert backend.model_revision == "qwen2.5-1.5b-instruct"

    # Explicit model_revision is preserved
    custom_backend = LocalSmallLLMBackend(enabled=False, model_revision="custom-rev")
    assert custom_backend.model_revision == "custom-rev"


# ---------------------------------------------------------------------------
# 8. DeadlineStoppingCriteria, Single Executor Lock, & CPU Dynamic Quantization
# ---------------------------------------------------------------------------


def test_deadline_stopping_criteria_stops_after_deadline(monkeypatch):
    import torch

    now = [100.0]
    monkeypatch.setattr("videntra.query.llm.time.monotonic", lambda: now[0])

    criteria = DeadlineStoppingCriteria(deadline_monotonic=105.0)
    input_ids = torch.zeros((2, 4), dtype=torch.long)

    before = criteria(input_ids, scores=None)
    assert isinstance(before, torch.BoolTensor)
    assert before.shape == (2,)
    assert not bool(before.any())

    now[0] = 105.0
    at_deadline = criteria(input_ids, scores=None)
    assert isinstance(at_deadline, torch.BoolTensor)
    assert bool(at_deadline.all())

    # Scalar fallback when input_ids has no tensor shape/device
    assert criteria([[1, 2, 3]], scores=None) is True


def test_hf_invoke_sync_passes_stopping_criteria_and_max_new_tokens():
    import torch
    from transformers import StoppingCriteriaList

    captured_kwargs: dict = {}

    class DummyInputs(dict):
        def to(self, _device):
            return self

    class DummyTokenizer:
        eos_token_id = 2

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "prompt"

        def __call__(self, texts, return_tensors="pt"):
            return DummyInputs({"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)})

        def decode(self, gen_ids, skip_special_tokens=True):
            return '{"lang":"en","events":[]}'

    class DummyModel:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            captured_kwargs.update(kwargs)
            return torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    backend = LocalSmallLLMBackend(enabled=True, max_new_tokens=96)
    backend._loaded_backend = "transformers"
    backend._hf_tokenizer = DummyTokenizer()
    backend._hf_model = DummyModel()

    deadline = time.monotonic() + 5.0
    out = backend._invoke_sync("test prompt", deadline_monotonic=deadline)
    assert out == '{"lang":"en","events":[]}'
    assert captured_kwargs["max_new_tokens"] == 96
    assert isinstance(captured_kwargs["stopping_criteria"], StoppingCriteriaList)
    assert len(captured_kwargs["stopping_criteria"]) == 1
    assert isinstance(captured_kwargs["stopping_criteria"][0], DeadlineStoppingCriteria)
    assert captured_kwargs["stopping_criteria"][0].deadline_monotonic == pytest.approx(deadline)
    backend.close()


def test_single_executor_and_busy_infer_lock_prevents_zombie_threads(tmp_path: Path):
    import threading

    started = threading.Event()
    release = threading.Event()
    seen_threads: list[int] = []

    def blocking_llm(_prompt: str) -> str:
        seen_threads.append(threading.get_ident())
        started.set()
        release.wait(timeout=2.0)
        return json.dumps(
            {
                "lang": "en",
                "events": [
                    {"id": "e1", "vis": "person walking", "ocr": [], "asr": [], "objs": [], "pol": "pos"}
                ],
            }
        )

    backend = LocalSmallLLMBackend(generate_fn=blocking_llm, deadline_ms=500)
    try:
        executor_ref = backend._executor
        results: list[str | None] = []

        def first_call():
            results.append(backend.generate("query 1", deadline_ms=500))

        t1 = threading.Thread(target=first_call)
        t1.start()
        assert started.wait(timeout=1.0)

        # While first call holds _infer_lock, second call must fail fast with TimeoutError
        t0 = time.perf_counter()
        with pytest.raises(TimeoutError, match="busy"):
            backend.generate("query 2", deadline_ms=500)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 100.0

        # AdaptiveQueryDecomposer also falls back cleanly when backend is busy
        decomposer = AdaptiveQueryDecomposer(
            llm_backend=backend,
            cache_path=tmp_path / "cache.jsonl",
            enable_disk_cache=False,
        )
        plan = decomposer.decompose("A woman in a red coat walking past a bakery shop")
        assert plan.routing_mode == "fallback"

        release.set()
        t1.join(timeout=2.0)
        assert len(results) == 1 and results[0] is not None

        # After release, subsequent call reuses the exact same executor and worker thread
        out2 = backend.generate("query 3", deadline_ms=500)
        assert out2 is not None
        assert backend._executor is executor_ref
        assert len(seen_threads) == 2
        assert seen_threads[0] == seen_threads[1]
    finally:
        release.set()
        backend.close()


def test_cpu_dynamic_int8_quantization_and_safe_fallback():
    import torch

    tiny = torch.nn.Sequential(torch.nn.Linear(8, 4))
    quantized = LocalSmallLLMBackend._quantize_cpu_model(tiny, torch)
    assert isinstance(quantized[0], torch.ao.nn.quantized.dynamic.Linear)

    # Safe fallback if quantize_dynamic raises
    class BrokenTorch:
        qint8 = torch.qint8
        nn = torch.nn

        class ao:
            class quantization:
                @staticmethod
                def quantize_dynamic(*_args, **_kwargs):
                    raise RuntimeError("quantization unsupported")

    fallback_model = LocalSmallLLMBackend._quantize_cpu_model(tiny, BrokenTorch)
    assert fallback_model is tiny
