"""Small LLM query parser, adaptive complexity gate, compact wire expander, and semantic validator."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable
import unicodedata

from videntra.contracts import validate
from videntra.evidence.normalize import fold_diacritics, has_vietnamese_diacritics, normalize_text
from .schemas import (
    CompactDecomposedQuery,
    Entity,
    EntityAttribute,
    LogicNode,
    ModalityRoutingPlan,
    ParserMetadata,
    Predicate,
    PredicateScope,
    QueryGraph,
)
from .rules import COCO_SYNONYM_MAP, RuleBasedQueryParser
from .canonicalize import canonicalize_color, canonicalize_spatial
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


# ---------------------------------------------------------------------------
# Phase 7.2: Small-LLM Compact Wire Decomposer & Adaptive Modality Router
# ---------------------------------------------------------------------------

COMPACT_WIRE_SYSTEM_PROMPT = """You are VIDENTRA Query Decomposer.
Convert a Vietnamese/English video query into Compact Wire JSON (~60-100 tokens):
{
  "lang": "vi|en|mixed",
  "events": [{"id": "e1", "vis": "visual action clause", "ocr": [], "asr": [], "objs": [{"id": "p1", "cls": "person", "cnt": 1, "color": ""}], "pol": "pos|neg"}],
  "spatial": [{"subj": "p1", "op": "left_of|right_of|above|below|inside|near|behind|in_front_of", "obj": "o1", "pol": "pos|neg"}],
  "temporal": [{"first": "e1", "op": "before", "second": "e2"}]
}
Rules:
1. Order events chronologically (invert "after"/"sau khi" so first occurs before second).
2. Put on-screen text in "ocr", spoken words/topics in "asr", visual scenes in "vis".
3. Never invent OCR/ASR words not present in the user query.
4. Output ONLY valid JSON."""

_VALID_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_QUOTED_RE = re.compile(r'["\']([^"\']+)["\']')
_NEGATION_RE = re.compile(
    r"\b(không|chưa|chẳng|đừng|without|not|no|never|didn't|doesn't|don't|cannot|can't)\b",
    re.IGNORECASE,
)
_TEMPORAL_CLAUSE_RE = re.compile(
    r"\b(sau khi|trước khi|sau đó|trong lúc|khi đang|kế tiếp|tiếp theo|rồi|after|before|and then|then|while|meanwhile|followed by)\b",
    re.IGNORECASE,
)
_EXPLICIT_MODAL_CUE_RE = re.compile(
    r"\b(chữ|biển|tiêu đề|logo|text|sign|caption|subtitle|title|nói|hét|kể|phỏng vấn|giảng|say|says|said|speak|speaking|talk|talking|"
    r"bên trái|bên phải|phía trên|phía dưới|bên trong|phía sau|đằng sau|phía trước|left of|right of|above|below|inside|behind|in front of)\b",
    re.IGNORECASE,
)
_ACRONYM_OR_NUMBER_RE = re.compile(r"(?:\b[A-Z]{2,}[0-9]*\b|\d+%?)")


def _is_proper_noun_token(word: str) -> bool:
    clean = word.strip(".,!?;:\"'()[]{}")
    return len(clean) >= 2 and clean[0].isupper() and any(ch.islower() for ch in clean[1:])

# Minimal bilingual bridge for lexical grounding check of translated OCR/ASR phrases
_BILINGUAL_GROUNDING_SYNONYMS: dict[str, tuple[str, ...]] = {
    "weather": ("thời tiết", "thoi tiet"),
    "thời tiết": ("weather",),
    "hello": ("xin chào", "xin chao"),
    "xin chào": ("hello", "hi"),
    "sale": ("giảm giá", "giam gia", "khuyến mãi"),
    "breaking news": ("tin nóng", "tin mới"),
    "news": ("tin tức", "tin tuc", "bản tin"),
}


def _sanitize_id(raw_id: Any, prefix: str = "e") -> str:
    text = str(raw_id or "").strip().lower()
    text = re.sub(r"[^a-z0-9_]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text or not text[0].isalpha():
        text = f"{prefix}_{text}" if text else f"{prefix}_1"
    if not _VALID_ID_RE.match(text):
        return f"{prefix}_1"
    return text


def _extract_json_block(raw_response: str) -> str:
    text = raw_response.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _strip_quoted_segments(query: str) -> str:
    return _QUOTED_RE.sub(" ", query)


def _normalize_for_grounding(text: str) -> str:
    nfc = unicodedata.normalize("NFC", text).lower().strip()
    cleaned = re.sub(r"[^\w\s%]", " ", nfc)
    return re.sub(r"\s+", " ", cleaned).strip()


def _is_xanh_ambiguous(query: str, color_val: str | None = None) -> bool:
    q_low = unicodedata.normalize("NFC", query).lower()
    disambiguated = any(
        term in q_low
        for term in ("xanh dương", "xanh nước biển", "xanh da trời", "xanh lá", "xanh lục")
    )
    if disambiguated:
        return False
    if color_val and color_val.strip().lower() == "xanh":
        return True
    return bool(re.search(r"\bxanh\b", q_low))


def _ground_span(candidate: str, original_query: str) -> str:
    """Return an exact substring of original_query matching candidate, or '' if paraphrased/translated."""
    if not candidate:
        return ""
    if candidate in original_query:
        return candidate
    idx = original_query.lower().find(candidate.lower())
    if idx != -1:
        return original_query[idx : idx + len(candidate)]
    return ""


@dataclass(frozen=True)
class ComplexityDecision:
    """Output of QueryComplexityGate classifying a query into short/medium/long tiers."""

    tier: str  # "short", "medium", "long"
    routing_mode: str  # "fast_rule", "tri_modal_broadcast", "llm_compact"
    word_count: int
    temporal_clause_count: int
    has_tri_modal_signals: bool


class QueryComplexityGate:
    """Classifies queries into short (1-4w), medium (5-18w), or long (>18w or >=2 temporal clauses)."""

    def classify(self, query: str) -> ComplexityDecision:
        norm = normalize_text(query)
        words = norm.split()
        word_count = len(words)
        temporal_matches = _TEMPORAL_CLAUSE_RE.findall(norm)
        temporal_count = len(temporal_matches)

        if word_count > 18 or temporal_count >= 2:
            return ComplexityDecision(
                tier="long",
                routing_mode="llm_compact",
                word_count=word_count,
                temporal_clause_count=temporal_count,
                has_tri_modal_signals=False,
            )

        if word_count <= 4 and temporal_count == 0:
            has_quotes = bool(_QUOTED_RE.search(norm))
            has_acronym_or_num = bool(_ACRONYM_OR_NUMBER_RE.search(norm))
            has_modal_cue = bool(_EXPLICIT_MODAL_CUE_RE.search(norm))
            # Check proper nouns: any mid-sentence capitalized word OR >=2 capitalized words
            cap_words = [w for w in words if _is_proper_noun_token(w)]
            mid_cap = any(_is_proper_noun_token(w) for w in words[1:]) if len(words) > 1 else False
            has_proper_noun = mid_cap or len(cap_words) >= 2

            has_tri_modal = has_quotes or has_acronym_or_num or has_modal_cue or has_proper_noun
            routing_mode = "tri_modal_broadcast" if has_tri_modal else "fast_rule"
            return ComplexityDecision(
                tier="short",
                routing_mode=routing_mode,
                word_count=word_count,
                temporal_clause_count=temporal_count,
                has_tri_modal_signals=has_tri_modal,
            )

        return ComplexityDecision(
            tier="medium",
            routing_mode="llm_compact",
            word_count=word_count,
            temporal_clause_count=temporal_count,
            has_tri_modal_signals=False,
        )


class LocalSmallLLMBackend:
    """Lazy-loading local Small-LLM adapter (llama_cpp GGUF or transformers) with hard deadline_ms guard."""

    DEFAULT_MODELS_DIR = Path("models/llm")

    def __init__(
        self,
        model_path: str | Path | None = None,
        backend: str = "auto",
        model_revision: str = "qwen2.5-0.5b-instruct",
        deadline_ms: int = 250,
        enabled: bool = True,
        generate_fn: Callable[[str], str] | None = None,
        n_ctx: int = 1024,
        n_threads: int = 4,
    ):
        env_path = os.environ.get("VIDENTRA_LLM_MODEL_PATH")
        resolved_path = model_path if model_path is not None else env_path
        self.model_path: Path | None = Path(resolved_path) if resolved_path else None
        self.backend = backend
        self.model_revision = model_revision
        self.deadline_ms = max(1, int(deadline_ms))
        self.enabled = enabled
        self.generate_fn = generate_fn
        self.n_ctx = n_ctx
        self.n_threads = n_threads

        self._loaded_backend: str | None = None
        self._llama_model: Any = None
        self._hf_tokenizer: Any = None
        self._hf_model: Any = None
        self._load_attempted: bool = False
        self._available: bool = False

    def _discover_model_path(self) -> Path | None:
        if self.model_path is not None:
            return self.model_path if self.model_path.exists() else None
        if not self.DEFAULT_MODELS_DIR.exists():
            return None
        for pattern in ("*.gguf", "*qwen2.5*0.5b*", "*qwen2.5*1.5b*", "*"):
            matches = sorted(self.DEFAULT_MODELS_DIR.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _ensure_loaded(self) -> bool:
        if self.generate_fn is not None:
            self._available = True
            self._loaded_backend = "custom"
            return True
        if self._load_attempted:
            return self._available

        self._load_attempted = True
        discovered = self._discover_model_path()
        if discovered is None:
            self._available = False
            return False

        # Try llama_cpp first for GGUF files
        if self.backend in ("auto", "llama_cpp") and discovered.is_file() and discovered.suffix.lower() == ".gguf":
            try:
                from llama_cpp import Llama  # type: ignore

                self._llama_model = Llama(
                    model_path=str(discovered),
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    verbose=False,
                )
                self._loaded_backend = "llama_cpp"
                self._available = True
                return True
            except Exception:
                pass

        # Try transformers for HuggingFace directory
        if self.backend in ("auto", "transformers") and discovered.is_dir():
            try:
                import torch  # type: ignore
                from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

                self._hf_tokenizer = AutoTokenizer.from_pretrained(str(discovered), local_files_only=True)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                self._hf_model = AutoModelForCausalLM.from_pretrained(
                    str(discovered),
                    local_files_only=True,
                    torch_dtype=dtype,
                ).to(device)
                self._hf_model.eval()
                self._loaded_backend = "transformers"
                self._available = True
                return True
            except Exception:
                pass

        self._available = False
        return False

    def _invoke_sync(self, prompt: str) -> str:
        if self.generate_fn is not None:
            return self.generate_fn(prompt)

        if self._loaded_backend == "llama_cpp" and self._llama_model is not None:
            res = self._llama_model.create_chat_completion(
                messages=[
                    {"role": "system", "content": COMPACT_WIRE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=160,
                response_format={"type": "json_object"},
            )
            return res["choices"][0]["message"]["content"]

        if self._loaded_backend == "transformers" and self._hf_model is not None and self._hf_tokenizer is not None:
            import torch  # type: ignore

            messages = [
                {"role": "system", "content": COMPACT_WIRE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            text = self._hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._hf_tokenizer([text], return_tensors="pt").to(self._hf_model.device)
            with torch.no_grad():
                outputs = self._hf_model.generate(**inputs, max_new_tokens=160, do_sample=False)
            gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
            return self._hf_tokenizer.decode(gen_ids, skip_special_tokens=True)

        raise RuntimeError("No LLM backend loaded")

    def generate(self, prompt: str, deadline_ms: int | None = None) -> str | None:
        """Run inference with hard deadline_ms timeout guard. Returns None if disabled/unavailable."""
        if not self.enabled:
            return None
        if not self._ensure_loaded():
            return None

        eff_deadline_ms = self.deadline_ms if deadline_ms is None else max(1, int(deadline_ms))
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._invoke_sync, prompt)
        try:
            return future.result(timeout=eff_deadline_ms / 1000.0)
        except FuturesTimeoutError as exc:
            executor.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError(f"LocalSmallLLMBackend exceeded deadline_ms={eff_deadline_ms}") from exc
        finally:
            executor.shutdown(wait=False)


def _topological_event_order(
    event_ids: list[str],
    temporal_edges: list[tuple[str, str]],
) -> list[str]:
    """Return event_ids sorted topologically by `before` edges; raises ValueError on cycle."""
    if not event_ids:
        return []
    id_set = set(event_ids)
    adj: dict[str, list[str]] = {eid: [] for eid in event_ids}
    indeg: dict[str, int] = {eid: 0 for eid in event_ids}

    for first, second in temporal_edges:
        if first in id_set and second in id_set and first != second:
            if second not in adj[first]:
                adj[first].append(second)
                indeg[second] += 1
        elif first == second and first in id_set:
            raise ValueError(f"Self-loop temporal cycle on event {first}")

    # Preserve original relative order for tie-breaking
    order_idx = {eid: idx for idx, eid in enumerate(event_ids)}
    ready = sorted((eid for eid, deg in indeg.items() if deg == 0), key=lambda x: order_idx[x])
    queue = deque(ready)
    sorted_ids: list[str] = []

    while queue:
        curr = queue.popleft()
        sorted_ids.append(curr)
        newly_ready: list[str] = []
        for nxt in adj[curr]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                newly_ready.append(nxt)
        for nxt in sorted(newly_ready, key=lambda x: order_idx[x]):
            queue.append(nxt)

    if len(sorted_ids) != len(event_ids):
        raise ValueError("Temporal DAG cycle detected")
    return sorted_ids


def expand_compact_wire_to_query_graph(
    compact: CompactDecomposedQuery | dict[str, Any],
    original_query: str,
    model_revision: str | None = None,
    prompt_version: str = "compact-v1",
    query_length_tier: str = "medium",
    routing_mode: str = "llm_compact",
) -> tuple[QueryGraph, ModalityRoutingPlan]:
    """Deterministically expand ~60-100 token Compact Wire JSON into QueryGraph v1 + ModalityRoutingPlan."""
    wire = compact if isinstance(compact, CompactDecomposedQuery) else CompactDecomposedQuery.from_dict(compact)

    lang = wire.lang if wire.lang in ("vi", "en", "mixed", "unknown") else (
        "vi" if has_vietnamese_diacritics(original_query) else "en"
    )

    entities: list[Entity] = []
    seen_entity_ids: set[str] = set()
    object_requirements: list[dict[str, Any]] = []
    ambiguities: list[str] = []

    predicates: list[Predicate] = []
    event_primary_pred: dict[str, str] = {}
    event_visual_clause: dict[str, str] = {}
    raw_event_ids: list[str] = []
    ocr_phrases: list[str] = []
    asr_phrases: list[str] = []

    for ev_idx, ev in enumerate(wire.events, start=1):
        ev_id = _sanitize_id(ev.get("id") or f"e{ev_idx}", prefix="e")
        while ev_id in event_primary_pred:
            ev_id = f"{ev_id}_{ev_idx}"
        raw_event_ids.append(ev_id)

        pol_raw = str(ev.get("pol", "pos")).lower()
        pol = "negative" if pol_raw in ("neg", "negative") else "positive"

        ev_entity_ids: list[str] = []
        for obj_idx, obj in enumerate(ev.get("objs") or [], start=1):
            if not isinstance(obj, dict):
                continue
            ent_id = _sanitize_id(obj.get("id") or f"obj_{ev_idx}_{obj_idx}", prefix="o")
            raw_cls = str(obj.get("cls", "object")).strip().lower()
            coco_cls = COCO_SYNONYM_MAP.get(raw_cls, raw_cls)
            ent_type = "person" if coco_cls == "person" else "object"
            span = _ground_span(str(obj.get("span") or coco_cls), original_query)
            cnt = int(obj.get("cnt", 1) or 1)
            color_raw = str(obj.get("color") or "").strip()

            attrs: list[EntityAttribute] = []
            uncertainty: list[str] = []
            if cnt > 1:
                attrs.append(EntityAttribute(name="cardinality", value=str(cnt), source_span=span))
            if color_raw:
                can_color = canonicalize_color(color_raw)
                attrs.append(EntityAttribute(name="color", value=can_color, source_span=span))
                if _is_xanh_ambiguous(original_query, color_raw):
                    uncertainty.append("blue_or_green")
                    if "blue_or_green" not in ambiguities:
                        ambiguities.append("blue_or_green")

            if ent_id not in seen_entity_ids:
                seen_entity_ids.add(ent_id)
                entities.append(
                    Entity(
                        id=ent_id,
                        type=ent_type,
                        mention=span,
                        attributes=attrs,
                        coreference_of=None,
                        uncertainty=uncertainty,
                    )
                )
                object_requirements.append(
                    {
                        "id": ent_id,
                        "class": coco_cls,
                        "count": cnt,
                        "color": canonicalize_color(color_raw) if color_raw else None,
                    }
                )
            ev_entity_ids.append(ent_id)

        vis_text = str(ev.get("vis") or "").strip()
        if vis_text:
            vis_pred_id = _sanitize_id(f"{ev_id}_vis", prefix="p")
            roles: dict[str, str] = {}
            if ev_entity_ids:
                roles["subject"] = ev_entity_ids[0]
            if len(ev_entity_ids) > 1:
                roles["object"] = ev_entity_ids[1]
            predicates.append(
                Predicate(
                    id=vis_pred_id,
                    kind="action",
                    operator="visual_event",
                    roles=roles,
                    value=vis_text,
                    source_span=_ground_span(str(ev.get("span") or vis_text), original_query),
                    polarity=pol,
                    modality_options=[["visual"]],
                )
            )
            event_primary_pred[ev_id] = vis_pred_id
            event_visual_clause[ev_id] = vis_text

        for ocr_idx, ocr_val in enumerate(ev.get("ocr") or [], start=1):
            ocr_str = str(ocr_val).strip()
            if not ocr_str:
                continue
            ocr_phrases.append(ocr_str)
            ocr_pred_id = _sanitize_id(f"{ev_id}_ocr_{ocr_idx}", prefix="p")
            predicates.append(
                Predicate(
                    id=ocr_pred_id,
                    kind="ocr",
                    operator="exact_text",
                    roles={},
                    value=ocr_str,
                    source_span=_ground_span(ocr_str, original_query),
                    polarity=pol,
                    modality_options=[["ocr"]],
                )
            )
            event_primary_pred.setdefault(ev_id, ocr_pred_id)

        for asr_idx, asr_val in enumerate(ev.get("asr") or [], start=1):
            asr_str = str(asr_val).strip()
            if not asr_str:
                continue
            asr_phrases.append(asr_str)
            asr_pred_id = _sanitize_id(f"{ev_id}_asr_{asr_idx}", prefix="p")
            predicates.append(
                Predicate(
                    id=asr_pred_id,
                    kind="asr",
                    operator="semantic_phrase",
                    roles={},
                    value=asr_str,
                    source_span=_ground_span(asr_str, original_query),
                    polarity=pol,
                    modality_options=[["asr", "audio_visual"]],
                )
            )
            event_primary_pred.setdefault(ev_id, asr_pred_id)

    has_spatial = False
    for sp_idx, sp in enumerate(wire.spatial, start=1):
        op_raw = str(sp.get("op") or "").strip()
        if not op_raw:
            continue
        has_spatial = True
        can_op = canonicalize_spatial(op_raw)
        subj = _sanitize_id(sp.get("subj") or "", prefix="o") if sp.get("subj") else ""
        obj = _sanitize_id(sp.get("obj") or "", prefix="o") if sp.get("obj") else ""
        roles = {}
        if subj and subj in seen_entity_ids:
            roles["subject"] = subj
        if obj and obj in seen_entity_ids:
            roles["object"] = obj
        pol = "negative" if str(sp.get("pol", "pos")).lower() in ("neg", "negative") else "positive"
        predicates.append(
            Predicate(
                id=_sanitize_id(f"spatial_{sp_idx}", prefix="s"),
                kind="spatial",
                operator=can_op,
                roles=roles,
                value=None,
                source_span=_ground_span(str(sp.get("span") or ""), original_query),
                polarity=pol,
                modality_options=[["visual"]],
            )
        )

    temporal_edges: list[tuple[str, str]] = []
    for t_idx, tm in enumerate(wire.temporal, start=1):
        first_raw = _sanitize_id(tm.get("first") or "", prefix="e")
        second_raw = _sanitize_id(tm.get("second") or "", prefix="e")
        op_raw = str(tm.get("op") or "before").strip().lower()
        if op_raw == "after":
            first_raw, second_raw = second_raw, first_raw
        temporal_edges.append((first_raw, second_raw))

        ev_a = event_primary_pred.get(first_raw)
        ev_b = event_primary_pred.get(second_raw)
        if ev_a and ev_b:
            predicates.append(
                Predicate(
                    id=_sanitize_id(f"temporal_{t_idx}", prefix="t"),
                    kind="temporal",
                    operator="before",
                    roles={"event_a": ev_a, "event_b": ev_b},
                    value=None,
                    source_span=_ground_span(str(tm.get("span") or ""), original_query),
                    polarity="positive",
                    modality_options=[["visual"]],
                )
            )

    # Order visual clauses chronologically if temporal edges form a valid DAG
    try:
        ordered_ev_ids = _topological_event_order(raw_event_ids, temporal_edges)
    except ValueError:
        ordered_ev_ids = raw_event_ids

    visual_clauses = tuple(
        event_visual_clause[eid] for eid in ordered_ev_ids if eid in event_visual_clause
    )
    if not visual_clauses:
        visual_clauses = (original_query,)

    if not predicates:
        predicates.append(
            Predicate(
                id="action_main",
                kind="action",
                operator="generic_action",
                roles={},
                value=original_query,
                source_span=original_query,
                polarity="positive",
                modality_options=[["visual"]],
            )
        )

    root_node = LogicNode(
        id="node_root",
        op="AND",
        children=[p.id for p in predicates],
    )
    meta = ParserMetadata(
        status="validated",
        model_revision=model_revision,
        prompt_version=prompt_version,
        confidence_kind="uncalibrated",
    )
    qg = QueryGraph(
        original_query=original_query,
        language=lang,
        entities=entities,
        predicates=predicates,
        logic=[root_node],
        root_id="node_root",
        ambiguities=ambiguities,
        residual_constraints=[],
        parser=meta,
        schema_version="1.0",
    )

    plan = ModalityRoutingPlan(
        original_query=original_query,
        query_length_tier=query_length_tier,
        visual_clauses=visual_clauses,
        ocr_phrases=tuple(dict.fromkeys(ocr_phrases)),
        asr_phrases=tuple(dict.fromkeys(asr_phrases)),
        object_requirements=tuple(object_requirements),
        has_explicit_ocr_intent=bool(ocr_phrases),
        has_explicit_asr_intent=bool(asr_phrases),
        has_spatial_constraints=has_spatial,
        has_temporal_constraints=bool(temporal_edges),
        query_graph=qg,
        routing_mode=routing_mode,
    )
    return qg, plan


class SemanticDecomposerValidator:
    """5-step anti-hallucination & guardrail validator (<0.5 ms) for Compact Wire & QueryGraph."""

    @staticmethod
    def is_span_grounded(span: str, original_query: str) -> bool:
        """Step 1: Verify OCR/ASR span is lexically grounded in original_query (or configured synonym)."""
        norm_span = _normalize_for_grounding(span)
        if not norm_span:
            return False
        norm_q = _normalize_for_grounding(original_query)
        folded_span = fold_diacritics(norm_span)
        folded_q = fold_diacritics(norm_q)

        if norm_span in norm_q or folded_span in folded_q:
            return True

        for syn in _BILINGUAL_GROUNDING_SYNONYMS.get(norm_span, ()):
            if syn in norm_q or fold_diacritics(syn) in folded_q:
                return True

        # Token-level check for multi-word spans (all non-trivial tokens must appear in query)
        tokens = [t for t in folded_span.split() if len(t) >= 2]
        if tokens and all(tok in folded_q for tok in tokens):
            return True
        return False

    def filter_hallucinated_compact(
        self, compact: CompactDecomposedQuery, original_query: str
    ) -> CompactDecomposedQuery:
        """Strip ungrounded OCR/ASR spans from CompactDecomposedQuery before expansion."""
        cleaned_events: list[dict[str, Any]] = []
        for ev in compact.events:
            ev_copy = dict(ev)
            ev_copy["ocr"] = [
                s for s in (ev.get("ocr") or []) if self.is_span_grounded(str(s), original_query)
            ]
            ev_copy["asr"] = [
                s for s in (ev.get("asr") or []) if self.is_span_grounded(str(s), original_query)
            ]
            cleaned_events.append(ev_copy)
        return CompactDecomposedQuery(
            lang=compact.lang,
            events=tuple(cleaned_events),
            spatial=compact.spatial,
            temporal=compact.temporal,
        )

    @staticmethod
    def check_temporal_dag(compact: CompactDecomposedQuery) -> None:
        """Step 3: Reject temporal cycles in compact wire."""
        event_ids = [_sanitize_id(e.get("id") or f"e{i}", prefix="e") for i, e in enumerate(compact.events, start=1)]
        edges: list[tuple[str, str]] = []
        for tm in compact.temporal:
            first_raw = _sanitize_id(tm.get("first") or "", prefix="e")
            second_raw = _sanitize_id(tm.get("second") or "", prefix="e")
            op_raw = str(tm.get("op") or "before").strip().lower()
            if op_raw == "after":
                first_raw, second_raw = second_raw, first_raw
            edges.append((first_raw, second_raw))
        if edges:
            _topological_event_order(event_ids, edges)

    @staticmethod
    def preserve_negation(graph: QueryGraph, original_query: str) -> None:
        """Step 2: Preserve negation_scope_unresolved if query negates outside quotes without negative polarity."""
        unquoted = _strip_quoted_segments(original_query)
        if _NEGATION_RE.search(unquoted):
            has_negative_pred = any(p.polarity == "negative" for p in graph.predicates)
            if not has_negative_pred:
                if "negation_scope_unresolved" not in graph.ambiguities:
                    graph.ambiguities.append("negation_scope_unresolved")
                for pred in graph.predicates:
                    if "negation_scope_unresolved" not in pred.uncertainty:
                        pred.uncertainty.append("negation_scope_unresolved")

    @staticmethod
    def enforce_coco_and_color_ambiguity(graph: QueryGraph, original_query: str) -> None:
        """Step 4: Normalize COCO classes and annotate 'xanh' -> blue_or_green ambiguity."""
        xanh_ambiguous = _is_xanh_ambiguous(original_query)
        for ent in graph.entities:
            if ent.type == "object":
                low_mention = ent.mention.lower().strip()
                if low_mention in COCO_SYNONYM_MAP:
                    ent.type = "person" if COCO_SYNONYM_MAP[low_mention] == "person" else "object"
            for attr in ent.attributes:
                if attr.name == "color" and (attr.value == "xanh" or xanh_ambiguous):
                    attr.value = canonicalize_color(attr.value)
                    if xanh_ambiguous and "blue_or_green" not in ent.uncertainty:
                        ent.uncertainty.append("blue_or_green")
                        if "blue_or_green" not in graph.ambiguities:
                            graph.ambiguities.append("blue_or_green")

    def validate_and_expand(
        self,
        compact: CompactDecomposedQuery | dict[str, Any],
        original_query: str,
        query_length_tier: str = "medium",
        routing_mode: str = "llm_compact",
        model_revision: str | None = None,
        prompt_version: str = "compact-v1",
    ) -> tuple[QueryGraph, ModalityRoutingPlan]:
        """Run all 5 guardrail steps and return validated (QueryGraph, ModalityRoutingPlan)."""
        wire = compact if isinstance(compact, CompactDecomposedQuery) else CompactDecomposedQuery.from_dict(compact)

        # Step 3 (pre-expansion): Temporal DAG cycle check
        self.check_temporal_dag(wire)

        # Step 1: Anti-hallucination lexical grounding check for OCR/ASR
        grounded_wire = self.filter_hallucinated_compact(wire, original_query)

        # Expand to QueryGraph v1 + ModalityRoutingPlan
        graph, plan = expand_compact_wire_to_query_graph(
            grounded_wire,
            original_query=original_query,
            model_revision=model_revision,
            prompt_version=prompt_version,
            query_length_tier=query_length_tier,
            routing_mode=routing_mode,
        )

        # Step 2: Negation preservation
        self.preserve_negation(graph, original_query)

        # Step 4: COCO & color ambiguity
        self.enforce_coco_and_color_ambiguity(graph, original_query)

        # Step 5: Official JSON Schema validation
        validate("query_graph", graph.to_dict())
        return graph, plan


class AdaptiveQueryDecomposer:
    """Adaptive query decomposer with RAM + disk JSONL cache, ComplexityGate, Small-LLM, and <1ms fallback."""

    DEFAULT_CACHE_PATH = Path("data/cache/query_decomposition_cache.jsonl")

    def __init__(
        self,
        llm_backend: LocalSmallLLMBackend | None = None,
        complexity_gate: QueryComplexityGate | None = None,
        validator: SemanticDecomposerValidator | None = None,
        rule_parser: RuleBasedQueryParser | None = None,
        cache_path: str | Path | None = None,
        model_revision: str = "qwen2.5-0.5b-instruct",
        prompt_version: str = "compact-v1",
        enable_disk_cache: bool = True,
    ):
        self.llm_backend = llm_backend or LocalSmallLLMBackend(model_revision=model_revision)
        self.complexity_gate = complexity_gate or QueryComplexityGate()
        self.validator = validator or SemanticDecomposerValidator()
        self.rule_parser = rule_parser or RuleBasedQueryParser()
        self.model_revision = model_revision
        self.prompt_version = prompt_version
        self.enable_disk_cache = enable_disk_cache
        self.cache_path = Path(cache_path) if cache_path is not None else self.DEFAULT_CACHE_PATH
        self._ram_cache: dict[str, ModalityRoutingPlan] = {}
        self._disk_loaded = False

    def _make_cache_key(self, query: str) -> str:
        nfc = unicodedata.normalize("NFC", query.strip())
        raw = f"{nfc}|{self.model_revision}|{self.prompt_version}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _plan_from_dict(data: dict[str, Any]) -> ModalityRoutingPlan:
        qg_data = data["query_graph"]
        entities = [
            Entity(
                id=e["id"],
                type=e["type"],
                mention=e["mention"],
                attributes=[EntityAttribute(**a) for a in e.get("attributes", [])],
                coreference_of=e.get("coreference_of"),
                uncertainty=e.get("uncertainty", []),
            )
            for e in qg_data["entities"]
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
            for p in qg_data["predicates"]
        ]
        logic = [LogicNode(id=l["id"], op=l["op"], children=l["children"]) for l in qg_data["logic"]]
        parser_meta = ParserMetadata(**qg_data["parser"])
        qg = QueryGraph(
            original_query=qg_data["original_query"],
            language=qg_data["language"],
            entities=entities,
            predicates=predicates,
            logic=logic,
            root_id=qg_data["root_id"],
            ambiguities=qg_data.get("ambiguities", []),
            residual_constraints=qg_data.get("residual_constraints", []),
            parser=parser_meta,
            schema_version=qg_data.get("schema_version", "1.0"),
        )
        return ModalityRoutingPlan(
            original_query=data["original_query"],
            query_length_tier=data["query_length_tier"],
            visual_clauses=tuple(data["visual_clauses"]),
            ocr_phrases=tuple(data["ocr_phrases"]),
            asr_phrases=tuple(data["asr_phrases"]),
            object_requirements=tuple(data.get("object_requirements", [])),
            has_explicit_ocr_intent=bool(data["has_explicit_ocr_intent"]),
            has_explicit_asr_intent=bool(data["has_explicit_asr_intent"]),
            has_spatial_constraints=bool(data["has_spatial_constraints"]),
            has_temporal_constraints=bool(data["has_temporal_constraints"]),
            query_graph=qg,
            routing_mode=data.get("routing_mode", "llm_compact"),
            cache_hit=True,
            decomposer_latency_ms=0.0,
        )

    def _load_disk_cache(self) -> None:
        if self._disk_loaded or not self.enable_disk_cache:
            self._disk_loaded = True
            return
        self._disk_loaded = True
        if not self.cache_path.exists():
            return
        try:
            with self.cache_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    key = rec.get("key")
                    plan_dict = rec.get("plan")
                    if key and isinstance(plan_dict, dict):
                        self._ram_cache[key] = self._plan_from_dict(plan_dict)
        except Exception:
            pass

    def _save_to_cache(self, key: str, plan: ModalityRoutingPlan) -> None:
        self._ram_cache[key] = plan
        if not self.enable_disk_cache:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "plan": plan.to_dict()}, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _build_plan_from_rule_graph(
        self,
        graph: QueryGraph,
        original_query: str,
        tier: str,
        routing_mode: str,
        latency_ms: float,
    ) -> ModalityRoutingPlan:
        ocr_phrases = tuple(
            dict.fromkeys(p.value for p in graph.predicates if p.kind == "ocr" and p.value)
        )
        asr_phrases = tuple(
            dict.fromkeys(p.value for p in graph.predicates if p.kind == "asr" and p.value)
        )
        if routing_mode == "tri_modal_broadcast":
            visual_clauses = (original_query,)
            ocr_phrases = tuple(dict.fromkeys((original_query, *ocr_phrases)))
            asr_phrases = tuple(dict.fromkeys((original_query, *asr_phrases)))
            has_ocr = True
            has_asr = True
        elif routing_mode == "fast_rule":
            visual_clauses = (original_query,)
            ocr_phrases = ()
            asr_phrases = ()
            has_ocr = False
            has_asr = False
        else:
            vis = [
                p.value or p.source_span
                for p in graph.predicates
                if p.kind in ("action", "attribute", "spatial") and (p.value or p.source_span)
            ]
            visual_clauses = tuple(dict.fromkeys(vis)) if vis else (original_query,)
            has_ocr = bool(ocr_phrases)
            has_asr = bool(asr_phrases)

        obj_reqs = tuple(
            {
                "id": e.id,
                "class": COCO_SYNONYM_MAP.get(e.mention.lower().strip(), e.type),
                "count": next((int(a.value) for a in e.attributes if a.name == "cardinality" and a.value.isdigit()), 1),
                "color": next((a.value for a in e.attributes if a.name == "color"), None),
            }
            for e in graph.entities
        )
        has_spatial = any(p.kind == "spatial" for p in graph.predicates)
        has_temporal = any(p.kind == "temporal" for p in graph.predicates)

        return ModalityRoutingPlan(
            original_query=original_query,
            query_length_tier=tier,
            visual_clauses=visual_clauses,
            ocr_phrases=ocr_phrases,
            asr_phrases=asr_phrases,
            object_requirements=obj_reqs,
            has_explicit_ocr_intent=has_ocr,
            has_explicit_asr_intent=has_asr,
            has_spatial_constraints=has_spatial,
            has_temporal_constraints=has_temporal,
            query_graph=graph,
            routing_mode=routing_mode,
            cache_hit=False,
            decomposer_latency_ms=latency_ms,
        )

    @staticmethod
    def _sanitize_rule_graph(graph: QueryGraph, norm_query: str) -> QueryGraph:
        """Sanitize RuleBasedQueryParser outputs so entity/predicate spans and roles pass QueryGraph v1 contracts."""
        oq = graph.original_query or norm_query
        for ent in graph.entities:
            ent.mention = _ground_span(ent.mention, oq)
            for attr in ent.attributes:
                attr.source_span = _ground_span(attr.source_span, oq)

        valid_ent_ids = {e.id for e in graph.entities}
        valid_pred_ids = {p.id for p in graph.predicates}
        cleaned_preds: list[Predicate] = []
        for p in graph.predicates:
            p.source_span = _ground_span(p.source_span, oq)
            p.roles = {
                r: t
                for r, t in p.roles.items()
                if t in (valid_pred_ids if r in ("event_a", "event_b") else valid_ent_ids)
            }
            if p.kind == "temporal" and ("event_a" not in p.roles or "event_b" not in p.roles):
                continue
            cleaned_preds.append(p)

        if not cleaned_preds:
            cleaned_preds = [
                Predicate(
                    id="action_main",
                    kind="action",
                    operator="generic_action",
                    roles={},
                    value=norm_query,
                    source_span=oq,
                    polarity="positive",
                    modality_options=[["visual"]],
                )
            ]

        graph.predicates = cleaned_preds
        kept_ids = {p.id for p in cleaned_preds}
        for node in graph.logic:
            node.children = [c for c in node.children if c in kept_ids or any(l.id == c for l in graph.logic)]
            if not node.children:
                node.children = [cleaned_preds[0].id]
        return graph

    def decompose(self, query: str, mode: str | None = None) -> ModalityRoutingPlan:
        """Decompose query into validated QueryGraph v1 + ModalityRoutingPlan with <1ms fallback."""
        t0 = time.perf_counter()
        norm_query = unicodedata.normalize("NFC", query.strip())
        cache_key = self._make_cache_key(norm_query)

        self._load_disk_cache()
        if cache_key in self._ram_cache:
            cached = self._ram_cache[cache_key]
            return ModalityRoutingPlan(
                original_query=cached.original_query,
                query_length_tier=cached.query_length_tier,
                visual_clauses=cached.visual_clauses,
                ocr_phrases=cached.ocr_phrases,
                asr_phrases=cached.asr_phrases,
                object_requirements=cached.object_requirements,
                has_explicit_ocr_intent=cached.has_explicit_ocr_intent,
                has_explicit_asr_intent=cached.has_explicit_asr_intent,
                has_spatial_constraints=cached.has_spatial_constraints,
                has_temporal_constraints=cached.has_temporal_constraints,
                query_graph=cached.query_graph,
                routing_mode=cached.routing_mode,
                cache_hit=True,
                decomposer_latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        decision = self.complexity_gate.classify(norm_query)

        # Short query fast paths (< 1 ms)
        if decision.tier == "short" and mode != "force_llm":
            rule_graph = self._sanitize_rule_graph(self.rule_parser.parse(norm_query), norm_query)
            validate("query_graph", rule_graph.to_dict())
            latency_ms = (time.perf_counter() - t0) * 1000.0
            plan = self._build_plan_from_rule_graph(
                rule_graph,
                original_query=norm_query,
                tier="short",
                routing_mode=decision.routing_mode,
                latency_ms=latency_ms,
            )
            self._save_to_cache(cache_key, plan)
            return plan

        # Medium / Long query -> Small-LLM Compact Wire JSON + 5-step Validator
        try:
            raw_resp = self.llm_backend.generate(norm_query)
            if raw_resp is not None:
                json_str = _extract_json_block(raw_resp)
                data = json.loads(json_str)
                graph, plan = self.validator.validate_and_expand(
                    data,
                    original_query=norm_query,
                    query_length_tier=decision.tier,
                    routing_mode="llm_compact",
                    model_revision=self.model_revision,
                    prompt_version=self.prompt_version,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                final_plan = ModalityRoutingPlan(
                    original_query=plan.original_query,
                    query_length_tier=plan.query_length_tier,
                    visual_clauses=plan.visual_clauses,
                    ocr_phrases=plan.ocr_phrases,
                    asr_phrases=plan.asr_phrases,
                    object_requirements=plan.object_requirements,
                    has_explicit_ocr_intent=plan.has_explicit_ocr_intent,
                    has_explicit_asr_intent=plan.has_explicit_asr_intent,
                    has_spatial_constraints=plan.has_spatial_constraints,
                    has_temporal_constraints=plan.has_temporal_constraints,
                    query_graph=graph,
                    routing_mode="llm_compact",
                    cache_hit=False,
                    decomposer_latency_ms=latency_ms,
                )
                self._save_to_cache(cache_key, final_plan)
                return final_plan
        except Exception:
            pass

        # Deterministic <1ms RuleBasedQueryParser fallback
        fb_graph = self._sanitize_rule_graph(self.rule_parser.parse(norm_query), norm_query)

        fb_graph.parser = ParserMetadata(
            status="fallback",
            model_revision=self.model_revision,
            prompt_version=self.prompt_version,
            confidence_kind="uncalibrated",
        )
        validate("query_graph", fb_graph.to_dict())
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return self._build_plan_from_rule_graph(
            fb_graph,
            original_query=norm_query,
            tier=decision.tier,
            routing_mode="fallback",
            latency_ms=latency_ms,
        )

    def parse(self, query: str) -> QueryGraph:
        """Convenience wrapper returning QueryGraph directly."""
        return self.decompose(query).query_graph
