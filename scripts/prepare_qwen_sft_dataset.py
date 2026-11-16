"""Build ChatML SFT dataset (train/val JSONL + manifest) for Qwen2.5-1.5B-Instruct LoRA.

Reuses VIDENTRA CompactDecomposedQuery, SemanticDecomposerValidator, and
RuleBasedQueryParser from videntra.query.

Strict zero-leakage guarantees:
- Template samples are synthetic and contain zero dataset/video IDs or timestamps.
- Optional --source-group-manifest is strictly filtered to the `tune` split;
  `confirm`, `heldout_test_201`, and any video IDs / GT timestamps are NEVER
  included or propagated.
- 100% of generated assistant Compact Wire JSON plans are validated through
  SemanticDecomposerValidator.validate_and_expand and checked to ensure no
  generic placeholder clauses (e.g. "visual action clause") are emitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import Any, Sequence
import unicodedata

# Ensure src/ is importable when invoked as `python scripts/prepare_qwen_sft_dataset.py`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from videntra.evidence.normalize import fold_diacritics, has_vietnamese_diacritics  # noqa: E402
import videntra.query.llm as llm_mod  # noqa: E402
from videntra.query.llm import SemanticDecomposerValidator  # noqa: E402
from videntra.query.rules import COCO_SYNONYM_MAP, RuleBasedQueryParser, decompose_query_clauses  # noqa: E402
from videntra.query.schemas import CompactDecomposedQuery  # noqa: E402

_VALID_COCO_CLASSES: frozenset[str] = frozenset(COCO_SYNONYM_MAP.values())

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

ALLOWED_SPLITS = frozenset({"tune"})
FORBIDDEN_SPLITS = frozenset({"confirm", "heldout_test_201", "heldout", "test", "test_queries"})
FORBIDDEN_GT_KEYS = frozenset(
    {
        "vid",
        "video_id",
        "target_videos",
        "relevant_windows",
        "target_windows",
        "timestamps",
        "start_ms",
        "end_ms",
        "known_target_window_id",
        "candidate_ids",
    }
)

GENERIC_PLACEHOLDER_CLAUSES = frozenset(
    {
        "visual action clause",
        "visual clause",
        "action clause",
        "scene description",
        "visual scene",
        "event 1",
        "event 2",
        "placeholder",
    }
)

DEFAULT_EXTRACTIVE_V2_SYSTEM_PROMPT = """You are VIDENTRA Query Decomposer.
Convert a Vietnamese/English video query into Compact Wire JSON (~60-100 tokens):
{
  "lang": "vi|en|mixed",
  "events": [{"id": "e1", "vis": "<exact grounded visual phrase from query>", "ocr": [], "asr": [], "objs": [{"id": "p1", "cls": "person", "cnt": 1, "color": ""}], "pol": "pos|neg"}],
  "spatial": [{"subj": "p1", "op": "left_of|right_of|above|below|inside|near|behind|in_front_of", "obj": "o1", "pol": "pos|neg"}],
  "temporal": [{"first": "e1", "op": "before", "second": "e2"}]
}
Rules:
1. Extract real descriptive words/phrases grounded in the user query into "vis" (never emit generic placeholders).
2. Order events chronologically (invert "after"/"sau khi" so first occurs before second).
3. Put on-screen text in "ocr", spoken words/topics in "asr", visual scenes in "vis".
4. Never invent OCR/ASR words not present in the user query.
5. Output ONLY valid JSON."""


def get_validator_commit() -> str | None:
    """Return current git HEAD commit hash of the validator/repository if available."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def normalize_query_for_dedup(query: str) -> str:
    """Normalize query text for leakage-free deduplication across train/val splits."""
    nfc = unicodedata.normalize("NFC", str(query or "")).strip().lower()
    return re.sub(r"\s+", " ", fold_diacritics(nfc)).strip()


def get_sft_system_prompt(prompt_version: str = "extractive-v2") -> str:
    """Return the extractive system prompt, reusing llm.py if available."""
    if prompt_version in ("extractive-v2", "lora-sft-v1") or prompt_version.startswith(("extractive-", "lora-")):
        prompt = getattr(llm_mod, "EXTRACTIVE_V2_SYSTEM_PROMPT", None)
        if isinstance(prompt, str) and prompt.strip():
            return prompt
    return DEFAULT_EXTRACTIVE_V2_SYSTEM_PROMPT


def is_generic_placeholder_clause(clause: str) -> bool:
    """Return True if a visual clause is empty or a generic non-extractive placeholder."""
    repo_fn = getattr(llm_mod, "is_generic_placeholder_clause", None)
    if callable(repo_fn) and repo_fn(clause):
        return True
    norm = re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(clause or "")).strip().lower())
    if not norm:
        return True
    if norm in GENERIC_PLACEHOLDER_CLAUSES:
        return True
    if re.fullmatch(r"(?:event|clause|scene)\s*\d*", norm):
        return True
    return False


def _has_lexical_overlap(vis: str, query: str) -> bool:
    vis_tokens = [t for t in re.findall(r"\w+", fold_diacritics(vis.lower())) if len(t) >= 2]
    q_folded = fold_diacritics(query.lower())
    return bool(vis_tokens) and any(tok in q_folded for tok in vis_tokens)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return len(rows)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _make_wire(
    lang: str,
    events: list[dict[str, Any]],
    spatial: list[dict[str, Any]] | None = None,
    temporal: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return CompactDecomposedQuery(
        lang=lang,
        events=tuple(events),
        spatial=tuple(spatial or ()),
        temporal=tuple(temporal or ()),
    ).to_dict()


def _build_template_specs() -> list[tuple[str, str, dict[str, Any]]]:
    """Return diverse bilingual (vi/en/mixed) SFT examples across all reasoning categories."""
    specs: list[tuple[str, str, dict[str, Any]]] = []

    # 1. Multi-clause visual scenes
    multi_clause_items = [
        (
            "en",
            "A chef chops onions on a wooden board and stirs soup in a pot",
            [
                {"id": "e1", "vis": "chef chops onions on a wooden board", "ocr": [], "asr": [], "objs": [{"id": "p1", "cls": "person", "cnt": 1, "color": ""}], "pol": "pos"},
                {"id": "e2", "vis": "stirs soup in a pot", "ocr": [], "asr": [], "objs": [{"id": "o1", "cls": "bowl", "cnt": 1, "color": ""}], "pol": "pos"},
            ],
        ),
        (
            "vi",
            "Người phụ nữ mở cửa sổ rồi tưới cây hoa trên ban công",
            [
                {"id": "e1", "vis": "Người phụ nữ mở cửa sổ", "ocr": [], "asr": [], "objs": [{"id": "p1", "cls": "person", "cnt": 1, "color": ""}], "pol": "pos"},
                {"id": "e2", "vis": "tưới cây hoa trên ban công", "ocr": [], "asr": [], "objs": [{"id": "o1", "cls": "potted plant", "cnt": 1, "color": ""}], "pol": "pos"},
            ],
        ),
        (
            "en",
            "A boy rides a bicycle along the park path while a dog runs beside him",
            [
                {"id": "e1", "vis": "boy rides a bicycle along the park path", "ocr": [], "asr": [], "objs": [{"id": "p1", "cls": "person", "cnt": 1, "color": ""}, {"id": "o1", "cls": "bicycle", "cnt": 1, "color": ""}], "pol": "pos"},
                {"id": "e2", "vis": "dog runs beside him", "ocr": [], "asr": [], "objs": [{"id": "o2", "cls": "dog", "cnt": 1, "color": ""}], "pol": "pos"},
            ],
        ),
        (
            "vi",
            "Một người đàn ông cầm chiếc cốc cà phê và đọc báo buổi sáng",
            [
                {"id": "e1", "vis": "người đàn ông cầm chiếc cốc cà phê", "ocr": [], "asr": [], "objs": [{"id": "p1", "cls": "person", "cnt": 1, "color": ""}, {"id": "o1", "cls": "cup", "cnt": 1, "color": ""}], "pol": "pos"},
                {"id": "e2", "vis": "đọc báo buổi sáng", "ocr": [], "asr": [], "objs": [{"id": "o2", "cls": "book", "cnt": 1, "color": ""}], "pol": "pos"},
            ],
        ),
    ]
    for lang, q, evs in multi_clause_items:
        specs.append(("multi_clause", q, _make_wire(lang, evs)))

    # 2. OCR queries (strictly grounded on-screen text)
    ocr_items = [
        ("en", 'A street corner showing a neon sign with the text "OPEN 24H"', "street corner showing a neon sign", ["OPEN 24H"], []),
        ("vi", 'Màn hình hiển thị chữ "GIẢM GIÁ 50%" trước cửa hàng quần áo', "cửa hàng quần áo", ["GIẢM GIÁ 50%"], [{"id": "o1", "cls": "tv", "cnt": 1, "color": ""}]),
        ("en", 'A news broadcast banner reading "BREAKING NEWS" at the bottom of the screen', "news broadcast banner at the bottom of the screen", ["BREAKING NEWS"], []),
        ("vi", 'Biển báo giao thông ghi chữ "CẤM ĐỖ XE" bên cạnh chiếc ô tô trắng', "Biển báo giao thông bên cạnh chiếc ô tô trắng", ["CẤM ĐỖ XE"], [{"id": "o1", "cls": "car", "cnt": 1, "color": "white"}]),
        ("mixed", 'Video hướng dẫn có tiêu đề "PYTHON TUTORIAL" trên màn hình laptop', "màn hình laptop", ["PYTHON TUTORIAL"], [{"id": "o1", "cls": "laptop", "cnt": 1, "color": ""}]),
    ]
    for lang, q, vis, ocr_list, objs in ocr_items:
        specs.append(("ocr", q, _make_wire(lang, [{"id": "e1", "vis": vis, "ocr": ocr_list, "asr": [], "objs": objs, "pol": "pos"}])))

    # 3. ASR queries (strictly grounded spoken phrases/topics)
    asr_items = [
        ("en", 'A speaker on stage saying "welcome to the annual summit" into a microphone', "speaker on stage", ["welcome to the annual summit"], [{"id": "p1", "cls": "person", "cnt": 1, "color": ""}]),
        ("vi", 'Phóng viên đang nói về "biến đổi khí hậu" trong phòng thu truyền hình', "trong phòng thu truyền hình", ["biến đổi khí hậu"], [{"id": "p1", "cls": "person", "cnt": 1, "color": ""}]),
        ("en", 'An instructor explaining "how to calibrate the camera" while pointing at a tripod', "pointing at a tripod", ["how to calibrate the camera"], [{"id": "p1", "cls": "person", "cnt": 1, "color": ""}]),
        ("vi", 'Người dẫn chương trình nhắc đến "dự báo thời tiết" trên bản tin tối', "Người dẫn chương trình", ["dự báo thời tiết"], [{"id": "p1", "cls": "person", "cnt": 1, "color": ""}]),
    ]
    for lang, q, vis, asr_list, objs in asr_items:
        specs.append(("asr", q, _make_wire(lang, [{"id": "e1", "vis": vis, "ocr": [], "asr": asr_list, "objs": objs, "pol": "pos"}])))

    # 4. Temporal queries (before & inverted after)
    temporal_items = [
        (
            "en",
            "A man turns off the lamp before closing the wooden door",
            "man turns off the lamp",
            "closing the wooden door",
            [{"id": "p1", "cls": "person", "cnt": 1, "color": ""}],
            [],
        ),
        (
            "en",
            "The runner drinks water after crossing the finish line",
            "crossing the finish line",
            "runner drinks water",
            [{"id": "p1", "cls": "person", "cnt": 1, "color": ""}],
            [{"id": "o1", "cls": "bottle", "cnt": 1, "color": ""}],
        ),
        (
            "vi",
            "Người đầu bếp bật bếp gas trước khi đặt chảo lên bếp",
            "Người đầu bếp bật bếp gas",
            "đặt chảo lên bếp",
            [{"id": "p1", "cls": "person", "cnt": 1, "color": ""}],
            [],
        ),
        (
            "vi",
            "Cô gái đeo balo sau khi tắt màn hình máy tính",
            "tắt màn hình máy tính",
            "Cô gái đeo balo",
            [{"id": "o1", "cls": "laptop", "cnt": 1, "color": ""}],
            [{"id": "p1", "cls": "person", "cnt": 1, "color": ""}, {"id": "o2", "cls": "backpack", "cnt": 1, "color": ""}],
        ),
    ]
    for lang, q, v1, v2, objs1, objs2 in temporal_items:
        specs.append(
            (
                "temporal",
                q,
                _make_wire(
                    lang,
                    [
                        {"id": "e1", "vis": v1, "ocr": [], "asr": [], "objs": objs1, "pol": "pos"},
                        {"id": "e2", "vis": v2, "ocr": [], "asr": [], "objs": objs2, "pol": "pos"},
                    ],
                    temporal=[{"first": "e1", "op": "before", "second": "e2"}],
                ),
            )
        )

    # 5. Spatial queries (all canonical spatial operators)
    spatial_cases = [
        ("en", "A woman in a red coat standing to the left of a parked car", "woman in a red coat standing", "p1", "person", "red", "left_of", "o1", "car", ""),
        ("vi", "Chú chó màu nâu ngồi bên phải chiếc ghế sofa trong phòng khách", "Chú chó màu nâu ngồi trong phòng khách", "o1", "dog", "brown", "right_of", "o2", "couch", ""),
        ("en", "A wall clock hanging above a wooden dining table", "wall clock hanging", "o1", "clock", "", "above", "o2", "dining table", ""),
        ("vi", "Đôi giày thể thao đặt phía dưới chiếc ghế gỗ", "Đôi giày thể thao đặt", "o1", "shoe", "", "below", "o2", "chair", ""),
        ("en", "A black cat sleeping inside an open cardboard box", "black cat sleeping", "o1", "cat", "black", "inside", "o2", "box", ""),
        ("vi", "Người đàn ông đứng phía sau chiếc xe đạp màu xanh dương", "Người đàn ông đứng", "p1", "person", "", "behind", "o1", "bicycle", "blue"),
        ("en", "A child standing in front of a large yellow school bus", "child standing", "p1", "person", "", "in_front_of", "o1", "bus", "yellow"),
        ("vi", "Chiếc điện thoại đặt gần cốc nước trên bàn làm việc", "Chiếc điện thoại đặt trên bàn làm việc", "o1", "cell phone", "", "near", "o2", "cup", ""),
    ]
    for lang, q, vis, s_id, s_cls, s_col, op, o_id, o_cls, o_col in spatial_cases:
        specs.append(
            (
                "spatial",
                q,
                _make_wire(
                    lang,
                    [
                        {
                            "id": "e1",
                            "vis": vis,
                            "ocr": [],
                            "asr": [],
                            "objs": [
                                {"id": s_id, "cls": s_cls, "cnt": 1, "color": s_col},
                                {"id": o_id, "cls": o_cls, "cnt": 1, "color": o_col},
                            ],
                            "pol": "pos",
                        }
                    ],
                    spatial=[{"subj": s_id, "op": op, "obj": o_id, "pol": "pos"}],
                ),
            )
        )

    # 6. Object cardinality & color (including Vietnamese 'xanh' ambiguity preservation)
    cardinality_cases = [
        ("en", "Three people wearing black jackets walking across a crosswalk", "Three people wearing black jackets walking across a crosswalk", [{"id": "p1", "cls": "person", "cnt": 3, "color": "black"}]),
        ("vi", "Hai chiếc xe máy màu đỏ dừng trước đèn giao thông", "Hai chiếc xe máy màu đỏ dừng trước đèn giao thông", [{"id": "o1", "cls": "motorcycle", "cnt": 2, "color": "red"}, {"id": "o2", "cls": "traffic light", "cnt": 1, "color": ""}]),
        ("en", "Four ceramic bowls arranged neatly on a kitchen counter", "Four ceramic bowls arranged neatly on a kitchen counter", [{"id": "o1", "cls": "bowl", "cnt": 4, "color": ""}]),
        ("vi", "Hai người đàn ông mặc áo xanh đang đẩy chiếc xe đẩy hàng", "Hai người đàn ông mặc áo xanh đang đẩy chiếc xe đẩy hàng", [{"id": "p1", "cls": "person", "cnt": 2, "color": "xanh"}]),
    ]
    for lang, q, vis, objs in cardinality_cases:
        specs.append(("cardinality", q, _make_wire(lang, [{"id": "e1", "vis": vis, "ocr": [], "asr": [], "objs": objs, "pol": "pos"}])))

    # 7. Negation queries (pol = "neg")
    negation_cases = [
        ("en", "A motorcyclist riding on the highway without wearing a helmet", "motorcyclist riding on the highway", [{"id": "p1", "cls": "person", "cnt": 1, "color": ""}, {"id": "o1", "cls": "motorcycle", "cnt": 1, "color": ""}], "neg"),
        ("vi", "Người công nhân bước vào công trường nhưng không đội mũ bảo hộ", "Người công nhân bước vào công trường", [{"id": "p1", "cls": "person", "cnt": 1, "color": ""}], "neg"),
        ("en", "A customer sitting at a cafe table not holding a phone", "customer sitting at a cafe table", [{"id": "p1", "cls": "person", "cnt": 1, "color": ""}, {"id": "o1", "cls": "cell phone", "cnt": 1, "color": ""}], "neg"),
        ("vi", "Nhân viên đứng trước quầy thu ngân không đeo khẩu trang", "Nhân viên đứng trước quầy thu ngân", [{"id": "p1", "cls": "person", "cnt": 1, "color": ""}], "neg"),
    ]
    for lang, q, vis, objs, pol in negation_cases:
        specs.append(("negation", q, _make_wire(lang, [{"id": "e1", "vis": vis, "ocr": [], "asr": [], "objs": objs, "pol": pol}])))

    # 8. Combined multimodal + spatial/temporal
    specs.append(
        (
            "multimodal",
            'Two cyclists stop beside a sign reading "SLOW" after crossing the bridge',
            _make_wire(
                "en",
                [
                    {"id": "e1", "vis": "Two cyclists crossing the bridge", "ocr": [], "asr": [], "objs": [{"id": "p1", "cls": "person", "cnt": 2, "color": ""}, {"id": "o1", "cls": "bicycle", "cnt": 2, "color": ""}], "pol": "pos"},
                    {"id": "e2", "vis": "cyclists stop beside a sign", "ocr": ["SLOW"], "asr": [], "objs": [], "pol": "pos"},
                ],
                spatial=[{"subj": "p1", "op": "near", "obj": "o1", "pol": "pos"}],
                temporal=[{"first": "e1", "op": "before", "second": "e2"}],
            ),
        )
    )
    specs.append(
        (
            "multimodal",
            'Người dẫn chương trình đứng bên trái màn hình có chữ "TIN NÓNG" và nói về "bão số 3"',
            _make_wire(
                "vi",
                [
                    {
                        "id": "e1",
                        "vis": "Người dẫn chương trình đứng bên trái màn hình",
                        "ocr": ["TIN NÓNG"],
                        "asr": ["bão số 3"],
                        "objs": [{"id": "p1", "cls": "person", "cnt": 1, "color": ""}, {"id": "o1", "cls": "tv", "cnt": 1, "color": ""}],
                        "pol": "pos",
                    }
                ],
                spatial=[{"subj": "p1", "op": "left_of", "obj": "o1", "pol": "pos"}],
            ),
        )
    )

    return specs


def _generate_parametric_variations(rng: random.Random, target_count: int) -> list[tuple[str, str, dict[str, Any]]]:
    """Generate additional deterministic compositional examples when target_count exceeds base templates."""
    colors = [("red", "đỏ"), ("blue", "xanh dương"), ("white", "trắng"), ("black", "đen"), ("yellow", "vàng")]
    actors_en = ["woman", "man", "chef", "cyclist", "student"]
    actors_vi = ["người phụ nữ", "người đàn ông", "đầu bếp", "sinh viên", "nhân viên"]
    objs = [("car", "ô tô"), ("bicycle", "xe đạp"), ("chair", "ghế"), ("laptop", "máy tính xách tay"), ("dog", "chú chó")]
    spatial_ops = [
        ("left_of", "to the left of", "bên trái"),
        ("right_of", "to the right of", "bên phải"),
        ("near", "near", "gần"),
        ("behind", "behind", "phía sau"),
        ("in_front_of", "in front of", "phía trước"),
    ]
    ocr_tokens = ["EXIT", "SALE 30%", "OPEN", "VIP", "STOP", "LIVE"]

    generated: list[tuple[str, str, dict[str, Any]]] = []
    seen_queries: set[str] = set()
    attempts = 0
    max_attempts = max(200, target_count * 10)

    while len(generated) < target_count and attempts < max_attempts:
        attempts += 1
        kind = attempts % 4
        c_en, c_vi = rng.choice(colors)
        o_en, o_vi = rng.choice(objs)
        cnt = rng.choice([1, 2, 3])
        if kind == 0:
            # EN spatial + color + count
            actor = rng.choice(actors_en)
            sp_op, sp_en, _ = rng.choice(spatial_ops)
            cnt_word = {1: "A", 2: "Two", 3: "Three"}[cnt]
            q = f"{cnt_word} {actor}{'s' if cnt > 1 else ''} standing {sp_en} a {c_en} {o_en}"
            wire = _make_wire(
                "en",
                [
                    {
                        "id": "e1",
                        "vis": f"{actor} standing {sp_en} a {c_en} {o_en}",
                        "ocr": [],
                        "asr": [],
                        "objs": [
                            {"id": "p1", "cls": "person", "cnt": cnt, "color": ""},
                            {"id": "o1", "cls": o_en, "cnt": 1, "color": c_en},
                        ],
                        "pol": "pos",
                    }
                ],
                spatial=[{"subj": "p1", "op": sp_op, "obj": "o1", "pol": "pos"}],
            )
            cat = "spatial"
        elif kind == 1:
            # VI spatial + OCR
            actor_vi = rng.choice(actors_vi)
            sp_op, _, sp_vi = rng.choice(spatial_ops)
            ocr = rng.choice(ocr_tokens)
            q = f'{actor_vi.capitalize()} mặc áo {c_vi} đứng {sp_vi} {o_vi} có chữ "{ocr}"'
            wire = _make_wire(
                "vi",
                [
                    {
                        "id": "e1",
                        "vis": f"{actor_vi} mặc áo {c_vi} đứng {sp_vi} {o_vi}",
                        "ocr": [ocr],
                        "asr": [],
                        "objs": [
                            {"id": "p1", "cls": "person", "cnt": 1, "color": c_vi},
                            {"id": "o1", "cls": o_vi, "cnt": 1, "color": ""},
                        ],
                        "pol": "pos",
                    }
                ],
                spatial=[{"subj": "p1", "op": sp_op, "obj": "o1", "pol": "pos"}],
            )
            cat = "ocr"
        elif kind == 2:
            # Temporal inverted after
            actor = rng.choice(actors_en)
            ocr = rng.choice(ocr_tokens)
            q = f'The {actor} points at a sign reading "{ocr}" after parking the {c_en} {o_en}'
            wire = _make_wire(
                "en",
                [
                    {"id": "e1", "vis": f"parking the {c_en} {o_en}", "ocr": [], "asr": [], "objs": [{"id": "o1", "cls": o_en, "cnt": 1, "color": c_en}], "pol": "pos"},
                    {"id": "e2", "vis": f"{actor} points at a sign", "ocr": [ocr], "asr": [], "objs": [{"id": "p1", "cls": "person", "cnt": 1, "color": ""}], "pol": "pos"},
                ],
                temporal=[{"first": "e1", "op": "before", "second": "e2"}],
            )
            cat = "temporal"
        else:
            # Negation
            actor_vi = rng.choice(actors_vi)
            q = f"{actor_vi.capitalize()} đi cạnh {o_vi} màu {c_vi} nhưng không mang theo túi xách"
            wire = _make_wire(
                "vi",
                [
                    {
                        "id": "e1",
                        "vis": f"{actor_vi} đi cạnh {o_vi} màu {c_vi}",
                        "ocr": [],
                        "asr": [],
                        "objs": [
                            {"id": "p1", "cls": "person", "cnt": 1, "color": ""},
                            {"id": "o1", "cls": o_vi, "cnt": 1, "color": c_vi},
                        ],
                        "pol": "neg",
                    }
                ],
            )
            cat = "negation"

        if q not in seen_queries:
            seen_queries.add(q)
            generated.append((cat, q, wire))

    return generated


def _verify_frozen_manifest_integrity(
    manifest_path: Path,
    loaded_obj: Any,
    expected_manifest_sha256: str | None = None,
) -> str:
    """Verify SHA-256 and optional split-lock canonical hashes of a source-group manifest."""
    actual_sha = _sha256_file(manifest_path)
    if expected_manifest_sha256 is not None:
        if actual_sha.lower() != expected_manifest_sha256.strip().lower():
            raise ValueError(
                f"Source-group manifest SHA-256 mismatch for {manifest_path}: "
                f"expected {expected_manifest_sha256}, got {actual_sha}"
            )
    if isinstance(loaded_obj, dict) and "records_sha256" in loaded_obj:
        records = loaded_obj.get("records")
        if not isinstance(records, list):
            raise ValueError(f"Invalid split-lock manifest {manifest_path}: missing records list")
        payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        computed = hashlib.sha256(payload).hexdigest()
        if computed != loaded_obj.get("records_sha256"):
            raise ValueError(
                f"Source-group manifest records_sha256 mismatch for {manifest_path}: "
                f"expected {loaded_obj.get('records_sha256')}, got {computed}"
            )
    if isinstance(loaded_obj, dict) and "metadata_sha256" in loaded_obj:
        meta_payload = json.dumps(
            loaded_obj.get("metadata", {}), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        computed_meta = hashlib.sha256(meta_payload).hexdigest()
        if computed_meta != loaded_obj.get("metadata_sha256"):
            raise ValueError(
                f"Source-group manifest metadata_sha256 mismatch for {manifest_path}: "
                f"expected {loaded_obj.get('metadata_sha256')}, got {computed_meta}"
            )
    return actual_sha


def _load_tune_queries_from_manifest(
    manifest_path: Path,
    expected_manifest_sha256: str | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Load query texts & source_group strictly from `split == 'tune'` in a source-group manifest (zero GT)."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Source-group manifest not found: {manifest_path}")

    raw_text = manifest_path.read_text(encoding="utf-8").strip()
    rows: list[dict[str, Any]] = []
    loaded_obj: Any = None
    if manifest_path.suffix.lower() == ".json":
        loaded_obj = json.loads(raw_text)
        if isinstance(loaded_obj, list):
            rows = [r for r in loaded_obj if isinstance(r, dict)]
        elif isinstance(loaded_obj, dict):
            for key in ("queries", "rows", "items", "records"):
                if isinstance(loaded_obj.get(key), list):
                    rows = [r for r in loaded_obj[key] if isinstance(r, dict)]
                    break
    else:
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)

    manifest_sha256 = _verify_frozen_manifest_integrity(
        manifest_path,
        loaded_obj,
        expected_manifest_sha256=expected_manifest_sha256,
    )

    stats: dict[str, Any] = {
        "total_rows": len(rows),
        "tune_kept": 0,
        "skipped_non_tune": 0,
        "deduped_normalized": 0,
        "manifest_sha256": manifest_sha256,
    }
    entries: list[dict[str, str]] = []
    seen_norm: set[str] = set()
    for row in rows:
        split_val = str(row.get("split") or "").strip().lower()
        if split_val in FORBIDDEN_SPLITS or split_val not in ALLOWED_SPLITS:
            stats["skipped_non_tune"] += 1
            continue
        # Never read or propagate GT fields even if present on the source manifest row
        clean_text = unicodedata.normalize(
            "NFC", str(row.get("query_text") or row.get("query") or "")
        ).strip()
        if not clean_text:
            continue
        norm_text = normalize_query_for_dedup(clean_text)
        if norm_text in seen_norm:
            stats["deduped_normalized"] += 1
            continue
        seen_norm.add(norm_text)
        source_group = str(row.get("source_group") or "").strip()
        if not source_group:
            source_group = str(row.get("query_id") or "").strip() or f"tune_row_{len(entries)}"
        entries.append({"query_text": clean_text, "source_group": source_group})
        stats["tune_kept"] += 1

    return entries, stats


def _extract_wire_from_rule_parser(query: str, parser: RuleBasedQueryParser) -> dict[str, Any]:
    """Convert a raw tune query into an extractive Compact Wire JSON using RuleBasedQueryParser."""
    graph = parser.parse(query)
    lang = "vi" if has_vietnamese_diacritics(query) else "en"

    objs: list[dict[str, Any]] = []
    for ent in graph.entities:
        raw_cls = ent.mention.strip().lower() if ent.mention else ""
        cls_name = COCO_SYNONYM_MAP.get(raw_cls, "")
        if not cls_name and ent.type == "person":
            cls_name = "person"
        if cls_name not in _VALID_COCO_CLASSES:
            continue
        cnt = 1
        color = ""
        for attr in ent.attributes:
            if attr.name == "cardinality" and str(attr.value).isdigit():
                cnt = max(1, int(attr.value))
            elif attr.name == "color" and attr.value:
                color = str(attr.value)
        objs.append({"id": ent.id, "cls": cls_name, "cnt": cnt, "color": color})

    ocr_list = list(dict.fromkeys(p.value for p in graph.predicates if p.kind == "ocr" and p.value))
    asr_list = list(dict.fromkeys(p.value for p in graph.predicates if p.kind == "asr" and p.value))
    has_neg = any(p.polarity == "negative" for p in graph.predicates)
    spatial_list = [
        {
            "subj": p.roles.get("subject", ""),
            "op": p.operator,
            "obj": p.roles.get("object", ""),
            "pol": "neg" if p.polarity == "negative" else "pos",
        }
        for p in graph.predicates
        if p.kind == "spatial" and p.roles.get("subject") and p.roles.get("object")
    ]

    clauses = decompose_query_clauses(query)
    if len(clauses) > 1:
        sub_clauses = [c for c in clauses[1:] if c.strip()]
        if len(sub_clauses) >= 2:
            multi_events: list[dict[str, Any]] = []
            for idx, clause_vis in enumerate(sub_clauses, start=1):
                c_low = clause_vis.lower()
                ev_ocr = [p for p in ocr_list if p.lower() in c_low]
                ev_asr = [p for p in asr_list if p.lower() in c_low]
                multi_events.append(
                    {
                        "id": f"e{idx}",
                        "vis": clause_vis,
                        "ocr": ev_ocr if idx > 1 else (ev_ocr or ocr_list),
                        "asr": ev_asr if idx > 1 else (ev_asr or asr_list),
                        "objs": objs if idx == 1 else [],
                        "pol": "neg" if has_neg and idx == 1 else "pos",
                    }
                )
            temporal_edges = [
                {"first": f"e{i}", "op": "before", "second": f"e{i + 1}"}
                for i in range(1, len(multi_events))
            ]
            multi_wire = _make_wire(
                lang=lang,
                events=multi_events,
                spatial=spatial_list,
                temporal=temporal_edges,
            )
            try:
                SemanticDecomposerValidator().validate_extractive_events(
                    CompactDecomposedQuery.from_dict(multi_wire),
                    query,
                )
                return multi_wire
            except Exception:
                pass

    return _make_wire(
        lang=lang,
        events=[
            {
                "id": "e1",
                "vis": query,
                "ocr": ocr_list,
                "asr": asr_list,
                "objs": objs,
                "pol": "neg" if has_neg else "pos",
            }
        ],
        spatial=spatial_list,
        temporal=[],
    )


def validate_sft_example(
    query: str,
    wire_dict: dict[str, Any],
    validator: SemanticDecomposerValidator,
    prompt_version: str = "extractive-v2",
) -> None:
    """Validate that a Compact Wire JSON has no generic placeholders and passes 5-step validator."""
    events = wire_dict.get("events") or []
    if not events:
        raise ValueError(f"SFT example has no events for query: {query!r}")
    for ev in events:
        vis = str(ev.get("vis") or "").strip()
        if is_generic_placeholder_clause(vis):
            raise ValueError(f"Rejected generic placeholder visual clause {vis!r} for query {query!r}")
        if not _has_lexical_overlap(vis, query):
            raise ValueError(f"Visual clause {vis!r} has zero lexical overlap with query {query!r}")

    validator.validate_and_expand(
        wire_dict,
        original_query=query,
        query_length_tier="medium",
        routing_mode="llm_compact",
        model_revision=BASE_MODEL_REVISION,
        prompt_version=prompt_version,
        reject_generic_clauses=True,
    )


def verify_sft_dataset_hashes(manifest_path: Path) -> bool:
    """Verify that sft_train.jsonl and sft_val.jsonl match the SHA-256 hashes recorded in sft_manifest.json."""
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_path = Path(manifest.get("train_path", ""))
    val_path = Path(manifest.get("val_path", ""))
    if not train_path.is_file() or not val_path.is_file():
        return False
    if _sha256_file(train_path) != manifest.get("train_sha256"):
        return False
    if _sha256_file(val_path) != manifest.get("val_sha256"):
        return False
    src_manifest = manifest.get("source_group_manifest")
    expected_src_sha = manifest.get("source_group_manifest_sha256")
    if src_manifest and expected_src_sha:
        src_path = Path(src_manifest)
        if not src_path.is_file() or _sha256_file(src_path) != expected_src_sha:
            return False
    return True


def _split_records_by_group(
    records: list[dict[str, Any]],
    record_groups: list[str],
    val_ratio: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into (train_rows, val_rows) by source/template group with zero normalized query leakage."""
    if len(records) <= 1 or val_ratio <= 0.0:
        return list(records), []

    grouped: dict[str, list[dict[str, Any]]] = {}
    for rec, grp in zip(records, record_groups):
        grouped.setdefault(grp, []).append(rec)

    group_keys = list(grouped.keys())
    rng.shuffle(group_keys)
    target_val_count = max(1, int(round(len(records) * float(val_ratio))))

    if len(group_keys) >= 2:
        val_rows: list[dict[str, Any]] = []
        train_rows: list[dict[str, Any]] = []
        val_groups: set[str] = set()
        for idx, gkey in enumerate(group_keys):
            # Always keep at least 1 group for training
            remaining_groups = len(group_keys) - idx
            if len(val_rows) < target_val_count and (remaining_groups > 1 or len(train_rows) > 0):
                val_rows.extend(grouped[gkey])
                val_groups.add(gkey)
            else:
                train_rows.extend(grouped[gkey])
        if not train_rows:
            # Move last val group to train if needed
            last_g = group_keys[-1]
            val_groups.discard(last_g)
            val_rows = [r for g in group_keys if g in val_groups for r in grouped[g]]
            train_rows = list(grouped[last_g])
    else:
        shuffled = list(records)
        rng.shuffle(shuffled)
        val_count = min(len(shuffled) - 1, target_val_count)
        val_rows = shuffled[:val_count]
        train_rows = shuffled[val_count:]

    train_norms = {normalize_query_for_dedup(r["messages"][1]["content"]) for r in train_rows}
    val_norms = {normalize_query_for_dedup(r["messages"][1]["content"]) for r in val_rows}
    overlap = train_norms & val_norms
    if overlap:
        raise ValueError(f"Normalized query leakage detected between train and val splits ({len(overlap)} queries)")

    return train_rows, val_rows


def build_sft_dataset(
    output_dir: Path,
    num_samples: int = 120,
    val_ratio: float = 0.15,
    seed: int = 42,
    prompt_version: str = "extractive-v2",
    source_group_manifest: Path | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Generate and validate sft_train.jsonl, sft_val.jsonl, and sft_manifest.json."""
    rng = random.Random(seed)
    system_prompt = get_sft_system_prompt(prompt_version)
    validator = SemanticDecomposerValidator()

    candidates: list[tuple[str, str, str, dict[str, Any], str]] = []
    for cat, q, wire in _build_template_specs():
        candidates.append(("template", cat, q, wire, f"template_family:{cat}"))

    remaining = max(0, int(num_samples) - len(candidates))
    if remaining > 0:
        for cat, q, wire in _generate_parametric_variations(rng, remaining):
            candidates.append(("template_var", cat, q, wire, f"template_family:{cat}"))

    manifest_stats: dict[str, Any] = {
        "total_rows": 0,
        "tune_kept": 0,
        "skipped_non_tune": 0,
        "deduped_normalized": 0,
        "manifest_sha256": None,
    }
    if source_group_manifest is not None:
        tune_entries, manifest_stats = _load_tune_queries_from_manifest(
            source_group_manifest,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        rule_parser = RuleBasedQueryParser()
        for entry in tune_entries:
            q = entry["query_text"]
            wire = _extract_wire_from_rule_parser(q, rule_parser)
            candidates.append(("source_manifest_tune", "extractive_tune", q, wire, f"source_group:{entry['source_group']}"))

    validated_records: list[dict[str, Any]] = []
    validated_groups: list[str] = []
    seen_norm_queries: set[str] = set()
    category_counts: dict[str, int] = {}
    placeholder_rejections = 0

    for idx, (source_kind, cat, query, wire, split_group) in enumerate(candidates):
        norm_q = normalize_query_for_dedup(query)
        if not norm_q or norm_q in seen_norm_queries:
            continue
        try:
            validate_sft_example(query, wire, validator=validator, prompt_version=prompt_version)
        except ValueError as exc:
            if "placeholder" in str(exc).lower():
                placeholder_rejections += 1
            continue

        seen_norm_queries.add(norm_q)
        assistant_json = json.dumps(wire, ensure_ascii=False, separators=(",", ":"))
        validated_records.append(
            {
                "id": f"sft_{idx:05d}",
                "source": source_kind,
                "category": cat,
                "split_group": split_group,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": assistant_json},
                ],
            }
        )
        validated_groups.append(split_group)
        category_counts[cat] = category_counts.get(cat, 0) + 1

    if not validated_records:
        raise ValueError("No valid SFT records remained after SemanticDecomposerValidator checks")

    train_rows, val_rows = _split_records_by_group(validated_records, validated_groups, val_ratio, rng)
    train_groups = sorted({str(r.get("split_group", "")) for r in train_rows})
    val_groups = sorted({str(r.get("split_group", "")) for r in val_rows})
    group_overlap = set(train_groups) & set(val_groups)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "sft_train.jsonl"
    val_path = output_dir / "sft_val.jsonl"
    manifest_path = output_dir / "sft_manifest.json"

    _atomic_write_jsonl(train_path, train_rows)
    _atomic_write_jsonl(val_path, val_rows)

    train_sha256 = _sha256_file(train_path)
    val_sha256 = _sha256_file(val_path)

    manifest = {
        "schema_version": "1.0",
        "base_model": BASE_MODEL_ID,
        "base_revision": BASE_MODEL_REVISION,
        "prompt_version": prompt_version,
        "validator_commit": get_validator_commit(),
        "seed": seed,
        "val_ratio": val_ratio,
        "split_strategy": "group_by_source_and_template_family",
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "total_count": len(validated_records),
        "train_groups_count": len(train_groups),
        "val_groups_count": len(val_groups),
        "group_overlap_count": len(group_overlap),
        "normalized_query_overlap_count": 0,
        "validator_pass_rate": 1.0,
        "placeholder_rejections": placeholder_rejections,
        "category_counts": category_counts,
        "source_group_manifest": str(source_group_manifest) if source_group_manifest else None,
        "source_group_manifest_sha256": manifest_stats.get("manifest_sha256"),
        "source_manifest_stats": manifest_stats,
        "allowed_splits": sorted(ALLOWED_SPLITS),
        "excluded_splits": sorted(FORBIDDEN_SPLITS),
        "forbidden_gt_keys_stripped": sorted(FORBIDDEN_GT_KEYS),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_sha256": train_sha256,
        "val_sha256": val_sha256,
    }
    _atomic_write_json(manifest_path, manifest)
    if not verify_sft_dataset_hashes(manifest_path):
        raise RuntimeError(f"Frozen SFT dataset hash verification failed for {manifest_path}")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build ChatML SFT dataset (train/val JSONL + manifest) for Qwen2.5-1.5B-Instruct LoRA."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft/qwen2.5_1.5b_lora"))
    parser.add_argument("--num-samples", type=int, default=120)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-version", type=str, default="extractive-v2")
    parser.add_argument(
        "--source-group-manifest",
        type=Path,
        default=None,
        help="Optional source-group manifest (JSON/JSONL); strictly filtered to split=='tune' only.",
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        type=str,
        default=None,
        help="Optional frozen SHA-256 hash to verify --source-group-manifest integrity.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = build_sft_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
        prompt_version=args.prompt_version,
        source_group_manifest=args.source_group_manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    print(
        f"Built Qwen SFT dataset at {args.output_dir}: "
        f"train={manifest['train_count']}, val={manifest['val_count']}, "
        f"validator_pass_rate={manifest['validator_pass_rate']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
