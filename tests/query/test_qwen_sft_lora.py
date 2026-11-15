"""Unit & integration tests for Qwen2.5-1.5B LoRA SFT dataset builder & Kaggle dry-run trainer."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.kaggle_train_qwen_lora import (
    DRY_RUN_SYNTHETIC_QUERIES,
    _DryRunTokenizer,
    format_and_tokenize_chatml,
    main as train_main,
)
from scripts.prepare_qwen_sft_dataset import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    build_sft_dataset,
    is_generic_placeholder_clause,
    main as prepare_main,
    validate_sft_example,
)
from videntra.contracts import validate
from videntra.query.llm import AdaptiveQueryDecomposer, SemanticDecomposerValidator


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_prepare_qwen_sft_dataset_validates_all_and_forbids_placeholders(tmp_path: Path) -> None:
    out_dir = tmp_path / "sft"
    rc = prepare_main(["--output-dir", str(out_dir), "--num-samples", "60", "--seed", "42"])
    assert rc == 0

    train_path = out_dir / "sft_train.jsonl"
    val_path = out_dir / "sft_val.jsonl"
    manifest_path = out_dir / "sft_manifest.json"
    assert train_path.is_file()
    assert val_path.is_file()
    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["base_model"] == BASE_MODEL_ID
    assert manifest["base_revision"] == BASE_MODEL_REVISION
    assert manifest["validator_pass_rate"] == 1.0
    assert manifest["placeholder_rejections"] == 0
    assert manifest["train_count"] > 0
    assert manifest["val_count"] > 0

    all_rows = _read_jsonl(train_path) + _read_jsonl(val_path)
    assert len(all_rows) == manifest["total_count"]

    validator = SemanticDecomposerValidator()
    categories_seen: set[str] = set()
    for row in all_rows:
        categories_seen.add(row["category"])
        msgs = row["messages"]
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        assert "visual action clause" not in msgs[0]["content"].lower()
        assert "visual action clause" not in msgs[2]["content"].lower()

        wire = json.loads(msgs[2]["content"])
        for ev in wire["events"]:
            assert not is_generic_placeholder_clause(ev["vis"])

        qg, _ = validator.validate_and_expand(
            wire,
            original_query=msgs[1]["content"],
            prompt_version=manifest["prompt_version"],
            model_revision=BASE_MODEL_REVISION,
        )
        validate("query_graph", qg.to_dict())

    for expected_cat in ("multi_clause", "ocr", "asr", "temporal", "spatial", "cardinality", "negation"):
        assert expected_cat in categories_seen


def test_prepare_qwen_sft_dataset_source_manifest_filters_strictly_to_tune_no_gt_leak(tmp_path: Path) -> None:
    manifest_in = tmp_path / "source_group_manifest.jsonl"
    rows = [
        {
            "query_id": "q_tune_1",
            "split": "tune",
            "source_group": "grp_tune",
            "query_text": "A chef wearing a white apron slices tomatoes on a wooden board",
            "vid": "SECRET_TUNE_VID_999",
            "relevant_windows": [[12.0, 25.0]],
        },
        {
            "query_id": "q_confirm_1",
            "split": "confirm",
            "source_group": "grp_confirm",
            "query_text": "FORBIDDEN_CONFIRM_QUERY_LEAK",
            "vid": "SECRET_CONFIRM_VID_888",
        },
        {
            "query_id": "q_heldout_1",
            "split": "heldout_test_201",
            "source_group": "grp_heldout",
            "query_text": "FORBIDDEN_HELDOUT_QUERY_LEAK",
            "vid": "SECRET_HELDOUT_VID_777",
        },
        {
            "query_id": "q_test_1",
            "split": "test",
            "source_group": "grp_test",
            "query_text": "FORBIDDEN_TEST_QUERY_LEAK",
        },
    ]
    manifest_in.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    out_dir = tmp_path / "sft_filtered"
    manifest = build_sft_dataset(
        output_dir=out_dir,
        num_samples=30,
        val_ratio=0.2,
        seed=7,
        source_group_manifest=manifest_in,
    )
    assert manifest["source_manifest_stats"]["tune_kept"] == 1
    assert manifest["source_manifest_stats"]["skipped_non_tune"] == 3

    combined_text = (out_dir / "sft_train.jsonl").read_text(encoding="utf-8") + (
        out_dir / "sft_val.jsonl"
    ).read_text(encoding="utf-8")

    assert "slices tomatoes on a wooden board" in combined_text
    for forbidden in (
        "FORBIDDEN_CONFIRM_QUERY_LEAK",
        "FORBIDDEN_HELDOUT_QUERY_LEAK",
        "FORBIDDEN_TEST_QUERY_LEAK",
        "SECRET_TUNE_VID_999",
        "SECRET_CONFIRM_VID_888",
        "SECRET_HELDOUT_VID_777",
        "relevant_windows",
    ):
        assert forbidden not in combined_text


def test_validate_sft_example_rejects_generic_placeholders() -> None:
    validator = SemanticDecomposerValidator()
    bad_wire = {
        "lang": "en",
        "events": [
            {
                "id": "e1",
                "vis": "visual action clause",
                "ocr": [],
                "asr": [],
                "objs": [{"id": "p1", "cls": "person", "cnt": 1, "color": ""}],
                "pol": "pos",
            }
        ],
        "spatial": [],
        "temporal": [],
    }
    with pytest.raises(ValueError, match="placeholder"):
        validate_sft_example("A person walking in the park", bad_wire, validator=validator)


def test_chatml_tokenization_masks_prompt_labels() -> None:
    tokenizer = _DryRunTokenizer()
    messages = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "A dog running on grass"},
        {"role": "assistant", "content": '{"lang":"en","events":[]}'},
    ]
    feat = format_and_tokenize_chatml(messages, tokenizer, max_seq_len=512)
    assert len(feat["input_ids"]) == len(feat["attention_mask"]) == len(feat["labels"])
    assert feat["labels"][0] == -100
    assert any(lbl != -100 for lbl in feat["labels"])


def test_kaggle_train_qwen_lora_dry_run_and_decomposer_cache_load(tmp_path: Path) -> None:
    out_dir = tmp_path / "lora_dry_run"
    rc = train_main(
        [
            "--dry-run",
            "--output-dir",
            str(out_dir),
            "--model-revision",
            "qwen2.5-1.5b-lora-test",
            "--prompt-version",
            "lora-sft-v1",
        ]
    )
    assert rc == 0

    lora_cfg = json.loads((out_dir / "lora_config.json").read_text(encoding="utf-8"))
    assert lora_cfg["base_model_name_or_path"] == BASE_MODEL_ID
    assert lora_cfg["base_model_revision"] == BASE_MODEL_REVISION
    assert lora_cfg["peft_type"] == "LORA"

    receipt = json.loads((out_dir / "qwen_receipt.json").read_text(encoding="utf-8"))
    assert receipt["dry_run"] is True
    assert receipt["cache_export"]["exported_records"] == len(DRY_RUN_SYNTHETIC_QUERIES)
    assert receipt["cache_export"]["rejected_count"] == 0

    cache_path = out_dir / "query_plan_cache.jsonl"
    assert cache_path.is_file()

    decomposer = AdaptiveQueryDecomposer(
        query_plan_cache=cache_path,
        query_plan_revision="qwen2.5-1.5b-lora-test",
        prompt_version="lora-sft-v1",
    )
    for query in DRY_RUN_SYNTHETIC_QUERIES:
        plan = decomposer.decompose(query)
        assert plan.cache_hit is True
        assert plan.routing_mode == "llm_compact"
        assert plan.query_graph.parser.status == "validated"
        assert plan.query_graph.parser.prompt_version == "lora-sft-v1"
        validate("query_graph", plan.query_graph.to_dict())


def test_kaggle_train_qwen_lora_dry_run_with_export_cache_queries_extractive_v2(tmp_path: Path) -> None:
    queries_file = tmp_path / "export_queries.jsonl"
    custom_queries = [
        {
            "query_id": "q1",
            "dataset": "custom",
            "query_text": 'A cyclist in a blue jacket stops beside a sign reading "STOP"',
        },
        {
            "query_id": "q2",
            "dataset": "custom",
            "query_text": "Người phụ nữ đặt chiếc cốc đỏ lên bàn trước khi mở laptop",
        },
    ]
    queries_file.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in custom_queries) + "\n", encoding="utf-8")

    out_dir = tmp_path / "lora_export_v2"
    rc = train_main(
        [
            "--dry-run",
            "--output-dir",
            str(out_dir),
            "--export-cache-queries",
            str(queries_file),
            "--model-revision",
            "qwen2.5-1.5b-lora-v2",
            "--prompt-version",
            "extractive-v2",
        ]
    )
    assert rc == 0

    decomposer = AdaptiveQueryDecomposer(
        query_plan_cache=out_dir / "query_plan_cache.jsonl",
        query_plan_revision="qwen2.5-1.5b-lora-v2",
        prompt_version="extractive-v2",
    )
    for item in custom_queries:
        plan = decomposer.decompose(item["query_text"])
        assert plan.cache_hit is True
        assert plan.routing_mode == "llm_compact"
        assert plan.query_graph.parser.prompt_version == "extractive-v2"
        assert all(not is_generic_placeholder_clause(c) for c in plan.visual_clauses)
