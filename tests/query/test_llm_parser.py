"""Tests for LLM parser, schema validation, zero-crash fallback, and cache."""
import json
import pytest

from videntra.contracts import validate
from videntra.query.llm import LLMQueryParser
from videntra.query.cache import QueryGraphCache
from videntra.query.rules import RuleBasedQueryParser


def test_llm_parser_valid_json_success():
    # Valid synthetic LLM output
    sample_rule = RuleBasedQueryParser().parse("người áo đỏ đưa túi")
    valid_json = json.dumps(sample_rule.to_dict())

    def mock_llm_fn(prompt: str) -> str:
        return f"```json\n{valid_json}\n```"

    parser = LLMQueryParser(llm_generate_fn=mock_llm_fn)
    graph = parser.parse("người áo đỏ đưa túi")

    assert graph.parser.status == "validated"
    validate("query_graph", graph.to_dict())


def test_llm_parser_garbage_output_graceful_fallback():
    """Blueprint rule: Parser failure must fallback to rules without crashing."""
    call_count = 0

    def broken_llm_fn(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return "I am an AI and cannot produce JSON for this request: ###ERROR###"

    parser = LLMQueryParser(llm_generate_fn=broken_llm_fn)
    # Must NOT raise exception!
    graph = parser.parse("người áo đỏ đưa túi cho người áo xanh")

    assert call_count == 1
    # Fallback status recorded
    assert graph.parser.status == "fallback"
    # Schema validation must still pass 100%
    validate("query_graph", graph.to_dict())


def test_llm_parser_schema_violation_fallback():
    """If LLM outputs JSON missing required schema properties, fallback to rules."""
    invalid_schema_json = json.dumps({"schema_version": "1.0", "wrong_key": 123})

    def invalid_llm_fn(prompt: str) -> str:
        return invalid_schema_json

    parser = LLMQueryParser(llm_generate_fn=invalid_llm_fn)
    graph = parser.parse("tìm biển hiệu SALE")

    assert graph.parser.status == "fallback"
    validate("query_graph", graph.to_dict())


def test_query_graph_cache_prevents_reparsing():
    calls = 0

    def counting_llm_fn(prompt: str) -> str:
        nonlocal calls
        calls += 1
        sample = RuleBasedQueryParser().parse(prompt)
        return json.dumps(sample.to_dict())

    cache = QueryGraphCache()
    parser = LLMQueryParser(llm_generate_fn=counting_llm_fn, cache=cache)

    g1 = parser.parse("người áo đỏ")
    assert calls == 1

    # Second call with same query should hit cache
    g2 = parser.parse("  người   áo đỏ  ")
    assert calls == 1
    assert g1.root_id == g2.root_id
