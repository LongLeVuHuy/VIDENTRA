"""Tests validating synthetic reference query graphs against JSON Schema."""
import json
from pathlib import Path
import pytest

from videntra.contracts import validate
from videntra.contracts.validation import ContractError


def test_synthetic_reference_graphs_schema_adherence():
    schema_dir = Path("docs/schemas")
    vi_sample = schema_dir / "query_vi.synthetic.json"
    en_sample = schema_dir / "query_en.synthetic.json"

    assert vi_sample.is_file()
    assert en_sample.is_file()

    vi_doc = json.loads(vi_sample.read_text(encoding="utf-8"))
    en_doc = json.loads(en_sample.read_text(encoding="utf-8"))

    # Both must validate cleanly
    validate("query_graph", vi_doc)
    validate("query_graph", en_doc)


def test_invalid_query_graph_rejected():
    # Missing required 'predicates'
    invalid = {
        "schema_version": "1.0",
        "original_query": "test query",
        "language": "vi",
        "entities": [],
        "logic": [],
        "root_id": "root",
        "parser": {
            "status": "validated",
            "model_revision": None,
            "prompt_version": "qg-v1",
            "confidence_kind": "uncalibrated",
        },
    }

    with pytest.raises(ContractError):
        validate("query_graph", invalid)
