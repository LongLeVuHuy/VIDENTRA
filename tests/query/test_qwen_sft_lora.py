"""Unit & integration tests for Qwen2.5-1.5B LoRA SFT dataset builder & Kaggle dry-run trainer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from scripts.capture_msr_candidate_pools import main as capture_main
from scripts.kaggle_train_qwen_lora import (
    DRY_RUN_SYNTHETIC_QUERIES,
    _DryRunTokenizer,
    export_query_plan_cache,
    format_and_tokenize_chatml,
    main as train_main,
)
from scripts.prewarm_llm_query_cache import main as prewarm_main
from scripts.prepare_qwen_sft_dataset import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    build_sft_dataset,
    get_validator_commit,
    is_generic_placeholder_clause,
    main as prepare_main,
    normalize_query_for_dedup,
    validate_sft_example,
    verify_sft_dataset_hashes,
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


def _setup_minimal_capture_env(tmp_path: Path, cached_query: str, missed_query: str) -> dict[str, Path]:
    val_dir = tmp_path / "validation"
    test_dir = tmp_path / "test_queries"
    (val_dir / "qvhighlights").mkdir(parents=True, exist_ok=True)
    (test_dir / "qvhighlights").mkdir(parents=True, exist_ok=True)
    rows = [
        {"qid": 1, "vid": "v1", "query": cached_query, "relevant_windows": [[0.0, 2.0]]},
        {"qid": 2, "vid": "v1", "query": missed_query, "relevant_windows": [[1.0, 3.0]]},
    ]
    (val_dir / "qvhighlights" / "highlight_val_release.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "metadata.db"
    faiss_path = tmp_path / "faiss_index.bin"
    model_dir = tmp_path / "siglip2"
    model_dir.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"db")
    faiss_path.write_bytes(b"faiss")
    return {
        "val_dir": val_dir,
        "test_dir": test_dir,
        "db_path": db_path,
        "faiss_path": faiss_path,
        "model_dir": model_dir,
    }


def _make_fake_pipeline_cls():
    class _FakeCapturePipeline:
        def __init__(
            self,
            *,
            query_plan_cache=None,
            query_plan_revision=None,
            query_plan_prompt_version="compact-v1",
            rules_only=False,
            **kwargs,
        ) -> None:
            self.video_durations = {"v1": 5.0}
            self.keyframe_map = {}
            self.all_embeddings = None
            self.ltr_ranker = None
            self.last_routing_plan = None
            if query_plan_cache is not None and query_plan_revision is not None:
                self.query_decomposer = AdaptiveQueryDecomposer.from_frozen_llm_cache(
                    query_plan_cache,
                    query_plan_revision,
                    query_plan_prompt_version=query_plan_prompt_version,
                )
            elif rules_only:
                self.query_decomposer = AdaptiveQueryDecomposer(rules_only=True, enable_disk_cache=False)
            else:
                self.query_decomposer = AdaptiveQueryDecomposer(
                    query_plan_prompt_version=query_plan_prompt_version,
                    enable_disk_cache=False,
                )

        def search_with_metadata(self, *, query: str, **kwargs) -> dict:
            self.last_routing_plan = self.query_decomposer.decompose(query, mode="accurate")
            return {
                "ranked_videos": [
                    {
                        "video_id": "v1",
                        "ranking_score": 1.0,
                        "moment_id": "m1",
                        "moment_start_seconds": 0.0,
                        "moment_end_seconds": 2.0,
                        "representative_timestamp_seconds": 1.0,
                        "representative_original_frame_index": 0,
                        "match_status": "exact",
                    }
                ],
                "diagnostics": {"all_candidate_moments": []},
            }

        def close(self) -> None:
            return None

    return _FakeCapturePipeline


def test_prewarm_prompt_version_and_receipt_output(tmp_path: Path) -> None:
    queries_file = tmp_path / "queries.jsonl"
    queries_file.write_text(
        json.dumps({"query_id": "q1", "dataset": "msrvtt", "query_text": "A dog is swimming in a pool"}) + "\n",
        encoding="utf-8",
    )
    cache_path = tmp_path / "prewarm_cache.jsonl"
    receipt_path = tmp_path / "prewarm_receipt.json"
    rc = prewarm_main([
        "--queries-file", str(queries_file),
        "--cache-path", str(cache_path),
        "--model-revision", "rev-prewarm-v2",
        "--prompt-version", "extractive-v2",
        "--receipt-path", str(receipt_path),
        "--no-enable-llm",
    ])
    assert rc == 0
    assert cache_path.is_file()
    assert receipt_path.is_file()

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["dry_run"] is False
    assert receipt["prompt_version"] == "extractive-v2"
    assert receipt["model_revision"] == "rev-prewarm-v2"
    assert receipt["cache_sha256"] == hashlib.sha256(cache_path.read_bytes()).hexdigest()

    records = _read_jsonl(cache_path)
    assert len(records) == 1
    assert records[0]["prompt_version"] == "extractive-v2"
    assert records[0]["model_revision"] == "rev-prewarm-v2"


@pytest.mark.parametrize("prompt_version", ["compact-v1", "extractive-v2", "lora-sft-v1"])
def test_capture_cli_accepts_prompt_versions_and_falls_back_on_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_version: str,
) -> None:
    cached_query = "A cyclist in a blue jacket stops beside a sign"
    missed_query = "A chef in a white apron slices tomatoes on a wooden board"
    env = _setup_minimal_capture_env(tmp_path, cached_query, missed_query)

    cache_path = tmp_path / "query_plan_cache.jsonl"
    model_rev = f"qwen2.5-1.5b-{prompt_version}"

    def _gen_fn(q: str) -> str:
        return json.dumps(
            {
                "lang": "en",
                "events": [
                    {
                        "id": "e1",
                        "vis": "A cyclist in a blue jacket stops beside a sign",
                        "ocr": [],
                        "asr": [],
                        "objs": [{"id": "o1", "cls": "cyclist", "cnt": 1, "color": "blue"}],
                        "pol": "pos",
                    }
                ],
                "spatial": [],
                "temporal": [],
            }
        )

    export_query_plan_cache(
        queries=[{"query_id": "q1", "dataset": "qvhighlights", "query_text": cached_query}],
        cache_path=cache_path,
        generate_fn=_gen_fn,
        model_revision=model_rev,
        prompt_version=prompt_version,
    )
    cache_sha = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    receipt_in = tmp_path / "qwen_receipt.json"
    receipt_in.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dry_run": False,
                "model_revision": model_rev,
                "prompt_version": prompt_version,
                "lora_config": {"peft_type": "LORA"} if prompt_version == "lora-sft-v1" else None,
                "cache_export": {"cache_sha256": cache_sha},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("scripts.capture_msr_candidate_pools.UnifiedMultimodalSearchPipeline", _make_fake_pipeline_cls())
    out_dir = tmp_path / "pools"
    eval_dir = tmp_path / "eval"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture_msr_candidate_pools.py",
            "--split", "dev_full_859",
            "--max-k", "2",
            "--out-dir", str(out_dir),
            "--eval-dir", str(eval_dir),
            "--val-dir", str(env["val_dir"]),
            "--test-dir", str(env["test_dir"]),
            "--db-path", str(env["db_path"]),
            "--faiss-path", str(env["faiss_path"]),
            "--model-dir", str(env["model_dir"]),
            "--no-ltr",
            "--query-plan-cache", str(cache_path),
            "--query-plan-revision", model_rev,
            "--query-plan-prompt-version", prompt_version,
            "--query-plan-receipt", str(receipt_in),
        ],
    )
    capture_main()

    cap_receipt = json.loads((out_dir / "capture_receipt.json").read_text(encoding="utf-8"))
    assert cap_receipt["query_plan_prompt_version"] == prompt_version
    assert cap_receipt["query_plan_model_revision"] == model_rev
    assert cap_receipt["query_plan_cache_sha256"] == cache_sha
    stats = cap_receipt["query_decomposer_stats"]
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 1
    assert stats["llm_live_calls"] == 0

    split_lock = json.loads((eval_dir / "clip_vp_dev_full_859_split_lock.json").read_text(encoding="utf-8"))
    assert split_lock["metadata"]["labels_sha256"] == cap_receipt["labels_sha256"]

    a5_rows = _read_jsonl(out_dir / "a5_pool.jsonl")
    hits_by_qid = {r["query_id"]: r["query_plan_cache_hit"] for r in a5_rows}
    assert hits_by_qid["qvh_1"] is True
    assert hits_by_qid["qvh_2"] is False


def test_capture_rejects_mutated_prompt_revision_cache_byte_and_dry_run_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached_query = "A dog is swimming in a pool"
    env = _setup_minimal_capture_env(tmp_path, cached_query, "A cat is sleeping on a mat")
    monkeypatch.setattr("scripts.capture_msr_candidate_pools.UnifiedMultimodalSearchPipeline", _make_fake_pipeline_cls())

    queries_file = tmp_path / "export_queries.jsonl"
    queries_file.write_text(
        json.dumps({"query_id": "q1", "dataset": "qvhighlights", "query_text": cached_query}) + "\n",
        encoding="utf-8",
    )
    lora_dir = tmp_path / "lora_out"
    assert train_main([
        "--dry-run",
        "--output-dir", str(lora_dir),
        "--export-cache-queries", str(queries_file),
        "--model-revision", "rev-lora-1",
        "--prompt-version", "extractive-v2",
    ]) == 0

    cache_path = lora_dir / "query_plan_cache.jsonl"
    dry_receipt_path = lora_dir / "qwen_receipt.json"
    non_dry_receipt_path = tmp_path / "real_receipt.json"
    receipt_obj = json.loads(dry_receipt_path.read_text(encoding="utf-8"))
    receipt_obj["dry_run"] = False
    non_dry_receipt_path.write_text(json.dumps(receipt_obj), encoding="utf-8")

    def _run_capture(pv: str, rev: str, receipt_p: Path | None, allow_dry: bool = False, tag: str = "run") -> None:
        argv = [
            "capture_msr_candidate_pools.py",
            "--split", "dev_full_859",
            "--max-k", "2",
            "--out-dir", str(tmp_path / f"out_{tag}"),
            "--eval-dir", str(tmp_path / f"eval_{tag}"),
            "--val-dir", str(env["val_dir"]),
            "--test-dir", str(env["test_dir"]),
            "--db-path", str(env["db_path"]),
            "--faiss-path", str(env["faiss_path"]),
            "--model-dir", str(env["model_dir"]),
            "--no-ltr",
            "--query-plan-cache", str(cache_path),
            "--query-plan-revision", rev,
            "--query-plan-prompt-version", pv,
        ]
        if receipt_p is not None:
            argv.extend(["--query-plan-receipt", str(receipt_p)])
        if allow_dry:
            argv.append("--allow-dry-run-cache")
        monkeypatch.setattr(sys, "argv", argv)
        capture_main()

    # 1. dry_run=true receipt rejected unless --allow-dry-run-cache is passed
    with pytest.raises(ValueError, match="dry_run=true"):
        _run_capture("extractive-v2", "rev-lora-1", dry_receipt_path, allow_dry=False, tag="dry_fail")

    # Allowed when flagged smoke-only
    _run_capture("extractive-v2", "rev-lora-1", dry_receipt_path, allow_dry=True, tag="dry_smoke")
    smoke_receipt = json.loads((tmp_path / "out_dry_smoke" / "capture_receipt.json").read_text(encoding="utf-8"))
    assert smoke_receipt["query_plan_dry_run"] is True

    # 2. Mutated prompt_version rejected
    with pytest.raises(ValueError, match="prompt_version"):
        _run_capture("compact-v1", "rev-lora-1", non_dry_receipt_path, tag="bad_pv")

    # 3. Mutated revision rejected
    with pytest.raises(ValueError, match="model_revision"):
        _run_capture("extractive-v2", "wrong-rev", non_dry_receipt_path, tag="bad_rev")

    # 4. Mutate 1 byte in cache -> rejected by receipt SHA-256 check and by cache loader
    raw_bytes = cache_path.read_bytes()
    mutated_bytes = raw_bytes[:-2] + (b"X" if raw_bytes[-2:-1] != b"X" else b"Y") + raw_bytes[-1:]
    cache_path.write_bytes(mutated_bytes)
    with pytest.raises(ValueError, match="SHA-256"):
        _run_capture("extractive-v2", "rev-lora-1", non_dry_receipt_path, tag="bad_sha")
    with pytest.raises(ValueError):
        _run_capture("extractive-v2", "rev-lora-1", None, tag="bad_json")


def test_prepare_qwen_sft_dataset_group_split_no_norm_leak_and_frozen_hash(tmp_path: Path) -> None:
    manifest_in = tmp_path / "text_only_manifest.jsonl"
    rows = [
        {"query_id": "t1", "split": "tune", "source_group": "grp_a", "query_text": "A brown dog jumps over a fence"},
        {"query_id": "t2", "split": "tune", "source_group": "grp_a", "query_text": "  a   brown dog jumps over a fence  "},
        {"query_id": "t3", "split": "tune", "source_group": "grp_b", "query_text": "Two cyclists ride past a red bus"},
        {"query_id": "t4", "split": "tune", "source_group": "grp_c", "query_text": "A woman pours coffee into a white mug"},
        {"query_id": "t5", "split": "tune", "source_group": "grp_d", "query_text": "Children play soccer on a green field"},
        {"query_id": "c1", "split": "confirm", "source_group": "grp_c_only", "query_text": "CONFIRM_MUST_NOT_APPEAR"},
    ]
    manifest_in.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_in.read_bytes()).hexdigest()

    # Wrong frozen hash must fail
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_sft_dataset(
            output_dir=tmp_path / "bad_hash",
            num_samples=0,
            val_ratio=0.25,
            seed=11,
            source_group_manifest=manifest_in,
            expected_manifest_sha256="0" * 64,
        )

    out_dir = tmp_path / "sft_grouped"
    rc = prepare_main([
        "--output-dir", str(out_dir),
        "--num-samples", "0",
        "--val-ratio", "0.25",
        "--seed", "11",
        "--source-group-manifest", str(manifest_in),
        "--expected-manifest-sha256", manifest_sha,
    ])
    assert rc == 0

    sft_manifest = json.loads((out_dir / "sft_manifest.json").read_text(encoding="utf-8"))
    assert sft_manifest["source_group_manifest_sha256"] == manifest_sha
    assert sft_manifest["group_overlap_count"] == 0
    assert sft_manifest["normalized_query_overlap_count"] == 0
    assert sft_manifest["validator_commit"] == get_validator_commit()
    assert verify_sft_dataset_hashes(out_dir / "sft_manifest.json") is True

    train_rows = _read_jsonl(out_dir / "sft_train.jsonl")
    val_rows = _read_jsonl(out_dir / "sft_val.jsonl")
    assert train_rows and val_rows
    assert {r["split_group"] for r in train_rows}.isdisjoint({r["split_group"] for r in val_rows})
    train_norms = {normalize_query_for_dedup(r["messages"][1]["content"]) for r in train_rows}
    val_norms = {normalize_query_for_dedup(r["messages"][1]["content"]) for r in val_rows}
    assert train_norms.isdisjoint(val_norms)
    assert "CONFIRM_MUST_NOT_APPEAR" not in (out_dir / "sft_train.jsonl").read_text(encoding="utf-8")

    # Mutating train JSONL invalidates frozen hash check
    train_path = out_dir / "sft_train.jsonl"
    train_path.write_text(train_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert verify_sft_dataset_hashes(out_dir / "sft_manifest.json") is False


def test_validator_commit_propagated_across_prewarm_lora_and_capture_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_commit = get_validator_commit()
    assert expected_commit is not None and len(expected_commit) >= 7

    cached_query = "A dog is swimming in a pool"
    env = _setup_minimal_capture_env(tmp_path, cached_query, "A cat is sleeping on a mat")
    monkeypatch.setattr("scripts.capture_msr_candidate_pools.UnifiedMultimodalSearchPipeline", _make_fake_pipeline_cls())

    # 1. Prewarm receipt records validator_commit
    queries_file = tmp_path / "queries.jsonl"
    queries_file.write_text(
        json.dumps({"query_id": "q1", "dataset": "qvhighlights", "query_text": cached_query}) + "\n",
        encoding="utf-8",
    )
    prewarm_receipt = tmp_path / "prewarm_receipt.json"
    assert prewarm_main([
        "--queries-file", str(queries_file),
        "--cache-path", str(tmp_path / "prewarm_cache.jsonl"),
        "--model-revision", "rev-prewarm",
        "--prompt-version", "extractive-v2",
        "--receipt-path", str(prewarm_receipt),
        "--dry-run",
    ]) == 0
    assert json.loads(prewarm_receipt.read_text(encoding="utf-8"))["validator_commit"] == expected_commit

    # 2. LoRA dry-run receipt records validator_commit & propagates to capture_receipt.json
    lora_dir = tmp_path / "lora_out"
    assert train_main([
        "--dry-run",
        "--output-dir", str(lora_dir),
        "--export-cache-queries", str(queries_file),
        "--model-revision", "rev-lora-vc",
        "--prompt-version", "extractive-v2",
    ]) == 0
    lora_receipt = json.loads((lora_dir / "qwen_receipt.json").read_text(encoding="utf-8"))
    assert lora_receipt["validator_commit"] == expected_commit

    out_dir = tmp_path / "pools_vc"
    eval_dir = tmp_path / "eval_vc"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture_msr_candidate_pools.py",
            "--split", "dev_full_859",
            "--max-k", "2",
            "--out-dir", str(out_dir),
            "--eval-dir", str(eval_dir),
            "--val-dir", str(env["val_dir"]),
            "--test-dir", str(env["test_dir"]),
            "--db-path", str(env["db_path"]),
            "--faiss-path", str(env["faiss_path"]),
            "--model-dir", str(env["model_dir"]),
            "--no-ltr",
            "--query-plan-cache", str(lora_dir / "query_plan_cache.jsonl"),
            "--query-plan-revision", "rev-lora-vc",
            "--query-plan-prompt-version", "extractive-v2",
            "--query-plan-receipt", str(lora_dir / "qwen_receipt.json"),
            "--allow-dry-run-cache",
        ],
    )
    capture_main()
    cap_receipt = json.loads((out_dir / "capture_receipt.json").read_text(encoding="utf-8"))
    assert cap_receipt["validator_commit"] == expected_commit
    assert cap_receipt["query_plan_validator_commit"] == expected_commit


@pytest.mark.parametrize(
    ("query", "bad_vis", "pol"),
    [
        ("the dog is not really swimming in the pool", "swimming in the pool", "pos"),
        ("two brown dogs are swimming in the pool", "dogs are swimming in the pool", "pos"),
    ],
)
def test_sft_and_lora_export_enforce_extractive_v2_constraint_validator(
    tmp_path: Path,
    query: str,
    bad_vis: str,
    pol: str,
) -> None:
    bad_wire = {
        "lang": "en",
        "events": [
            {
                "id": "e1",
                "vis": bad_vis,
                "ocr": [],
                "asr": [],
                "objs": [{"id": "o1", "cls": "dog", "cnt": 1, "color": ""}],
                "pol": pol,
            }
        ],
        "spatial": [],
        "temporal": [],
    }
    validator = SemanticDecomposerValidator()
    with pytest.raises(ValueError):
        validate_sft_example(query, bad_wire, validator=validator, prompt_version="extractive-v2")

    cache_path = tmp_path / "rejected_cache.jsonl"
    stats = export_query_plan_cache(
        queries=[{"query_id": "q1", "dataset": "msrvtt", "query_text": query}],
        cache_path=cache_path,
        generate_fn=lambda _q: json.dumps(bad_wire),
        model_revision="rev-lora-reject",
        prompt_version="extractive-v2",
    )
    assert stats["valid_plan_count"] == 0
    assert stats["rejected_count"] == 1
    assert _read_jsonl(cache_path) == []
