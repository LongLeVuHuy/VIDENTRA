"""Deterministic rule-based query parser mapping natural language to QueryGraph with multi-clause decomposition and 80 COCO synonyms."""
from __future__ import annotations

import re
from typing import Any

from videntra.evidence.normalize import has_vietnamese_diacritics
from .schemas import (
    QueryGraph,
    Entity,
    EntityAttribute,
    Predicate,
    PredicateScope,
    LogicNode,
    ParserMetadata,
)
from .canonicalize import canonicalize_action, canonicalize_spatial, canonicalize_color

# Full bilingual COCO 80 Class Synonyms Map
COCO_SYNONYM_MAP: dict[str, str] = {
    # Person
    "person": "person", "man": "person", "woman": "person", "boy": "person", "girl": "person",
    "guy": "person", "lady": "person", "people": "person", "child": "person", "kid": "person",
    "baby": "person", "player": "person", "chef": "person", "speaker": "person", "someone": "person",
    "children": "person", "men": "person", "women": "person",
    "người": "person", "đàn ông": "person", "phụ nữ": "person", "cô gái": "person", "chàng trai": "person",
    "em bé": "person", "đứa trẻ": "person", "cầu thủ": "person", "đầu bếp": "person",
    # Vehicles
    "bicycle": "bicycle", "bike": "bicycle", "cyclist": "bicycle", "xe đạp": "bicycle",
    "car": "car", "automobile": "car", "sedan": "car", "suv": "car", "van": "car", "ô tô": "car", "xe hơi": "car",
    "motorcycle": "motorcycle", "motorbike": "motorcycle", "motorcyclist": "motorcycle", "xe máy": "motorcycle",
    "airplane": "airplane", "plane": "airplane", "máy bay": "airplane",
    "bus": "bus", "xe buýt": "bus",
    "train": "train", "tàu hỏa": "train", "xe lửa": "train",
    "truck": "truck", "xe tải": "truck",
    "boat": "boat", "ship": "boat", "thuyền": "boat", "tàu thủy": "boat",
    # Outdoor & Traffic
    "traffic light": "traffic light", "đèn giao thông": "traffic light",
    "fire hydrant": "fire hydrant", "trụ cứu hỏa": "fire hydrant",
    "stop sign": "stop sign", "biển dừng": "stop sign",
    "parking meter": "parking meter", "đồng hồ đỗ xe": "parking meter",
    "bench": "bench", "ghế dài": "bench", "ghế công viên": "bench",
    # Animals
    "bird": "bird", "chim": "bird",
    "cat": "cat", "kitten": "cat", "mèo": "cat",
    "dog": "dog", "puppy": "dog", "hound": "dog", "chó": "dog",
    "horse": "horse", "ngựa": "horse",
    "sheep": "sheep", "cừu": "sheep",
    "cow": "cow", "bò": "cow",
    "elephant": "elephant", "voi": "elephant",
    "bear": "bear", "gấu": "bear",
    "zebra": "zebra", "ngựa vằn": "zebra",
    "giraffe": "giraffe", "hươu cao cổ": "giraffe",
    # Accessories
    "backpack": "backpack", "rucksack": "backpack", "ba lô": "backpack",
    "umbrella": "umbrella", "ô": "umbrella", "dù": "umbrella",
    "handbag": "handbag", "purse": "handbag", "bag": "handbag", "túi xách": "handbag", "túi": "handbag",
    "tie": "tie", "necktie": "tie", "cà vạt": "tie",
    "suitcase": "suitcase", "luggage": "suitcase", "vali": "suitcase",
    # Sports
    "frisbee": "frisbee", "đĩa bay": "frisbee",
    "skis": "skis", "skier": "skis", "ván trượt tuyết": "skis",
    "snowboard": "snowboard", "ván trượt tuyết đơn": "snowboard",
    "sports ball": "sports ball", "ball": "sports ball", "ping pong": "sports ball",
    "soccer ball": "sports ball", "basketball": "sports ball", "football": "sports ball", "quả bóng": "sports ball", "bóng": "sports ball",
    "kite": "kite", "diều": "kite",
    "baseball bat": "baseball bat", "gậy bóng chày": "baseball bat",
    "baseball glove": "baseball glove", "găng tay bóng chày": "baseball glove",
    "skateboard": "skateboard", "skater": "skateboard", "ván trượt": "skateboard",
    "surfboard": "surfboard", "surfer": "surfboard", "ván lướt sóng": "surfboard",
    "tennis racket": "tennis racket", "vợt tennis": "tennis racket",
    # Kitchen & Food
    "bottle": "bottle", "water bottle": "bottle", "chai": "bottle", "bình nước": "bottle",
    "wine glass": "wine glass", "ly rượu": "wine glass",
    "cup": "cup", "mug": "cup", "glass": "cup", "cốc": "cup", "ly": "cup", "tách": "cup",
    "fork": "fork", "dĩa": "fork", "nĩa": "fork",
    "knife": "knife", "dao": "knife",
    "spoon": "spoon", "thìa": "spoon", "muỗng": "spoon",
    "bowl": "bowl", "bát": "bowl", "tô": "bowl", "chén": "bowl",
    "banana": "banana", "chuối": "banana",
    "apple": "apple", "táo": "apple",
    "sandwich": "sandwich", "bánh mì kẹp": "sandwich",
    "orange": "orange", "cam": "orange",
    "broccoli": "broccoli", "súp lơ": "broccoli",
    "carrot": "carrot", "cà rốt": "carrot",
    "hot dog": "hot dog", "bánh mì xúc xích": "hot dog",
    "pizza": "pizza", "bánh pizza": "pizza",
    "donut": "donut", "bánh donut": "donut",
    "cake": "cake", "bánh ngọt": "cake", "bánh kem": "cake",
    # Furniture
    "chair": "chair", "seat": "chair", "ghế": "chair",
    "couch": "couch", "sofa": "couch", "ghế sofa": "couch",
    "potted plant": "potted plant", "plant": "potted plant", "chậu cây": "potted plant", "cây cảnh": "potted plant",
    "bed": "bed", "giường": "bed",
    "dining table": "dining table", "table": "dining table", "desk": "dining table", "bàn": "dining table",
    "toilet": "toilet", "bồn cầu": "toilet",
    # Electronics
    "tv": "tv", "television": "tv", "monitor": "tv", "screen": "tv", "tivi": "tv", "màn hình": "tv",
    "laptop": "laptop", "computer": "laptop", "pc": "laptop", "máy tính": "laptop",
    "mouse": "mouse", "mice": "mouse", "chuột máy tính": "mouse",
    "remote": "remote", "điều khiển": "remote",
    "keyboard": "keyboard", "bàn phím": "keyboard",
    "cell phone": "cell phone", "phone": "cell phone", "smartphone": "cell phone", "điện thoại": "cell phone",
    # Appliances & Indoor
    "microwave": "microwave", "lò vi sóng": "microwave",
    "oven": "oven", "lò nướng": "oven",
    "toaster": "toaster", "máy nướng bánh mì": "toaster",
    "sink": "sink", "bồn rửa": "sink",
    "refrigerator": "refrigerator", "fridge": "refrigerator", "tủ lạnh": "refrigerator",
    "book": "book", "sách": "book", "cuốn sách": "book",
    "clock": "clock", "watch": "clock", "đồng hồ": "clock",
    "vase": "vase", "lọ hoa": "vase", "bình hoa": "vase",
    "scissors": "scissors", "kéo": "scissors",
    "teddy bear": "teddy bear", "gấu bông": "teddy bear",
    "hair drier": "hair drier", "máy sấy tóc": "hair drier",
    "toothbrush": "toothbrush", "bàn chải đánh răng": "toothbrush",
}

NUMBER_WORDS: dict[str, int] = {
    "a": 1, "an": 1, "one": 1, "single": 1, "một": 1,
    "two": 2, "pair of": 2, "couple of": 2, "hai": 2,
    "three": 3, "ba": 3,
    "four": 4, "bốn": 4,
    "five": 5, "năm": 5,
    "six": 6, "sáu": 6,
    "seven": 7, "bảy": 7,
    "eight": 8, "tám": 8,
    "nine": 9, "chín": 9,
    "ten": 10, "mười": 10,
    "multiple": 2, "several": 2, "many": 2, "group of": 2, "nhiều": 2, "nhóm": 2,
}


INHERENT_PLURAL_NOUNS: set[str] = {
    "people",
    "children",
    "men",
    "women",
    "mice",
}

IRREGULAR_PLURAL_MAP: dict[str, str] = {
    "people": "person",
    "peoples": "person",
    "children": "child",
    "childrens": "child",
    "men": "man",
    "mans": "man",
    "women": "woman",
    "womens": "woman",
    "mice": "mouse",
}


def _singularize_coco_token(word: str) -> str:
    """Normalize English plural nouns (-s, -es, -ies, -ves, irregulars) to singular form."""
    w = word.lower().strip()
    if w in IRREGULAR_PLURAL_MAP:
        return IRREGULAR_PLURAL_MAP[w]
    if w in COCO_SYNONYM_MAP:
        return w
    # 1. -ies -> -y (e.g. puppies -> puppy, parties -> party)
    if w.endswith("ies") and len(w) > 4:
        cand = w[:-3] + "y"
        if cand in COCO_SYNONYM_MAP:
            return cand
    # 2. -ves -> -fe or -f (e.g. knives -> knife, wolves -> wolf)
    if w.endswith("ves") and len(w) > 4:
        cand_fe = w[:-3] + "fe"
        if cand_fe in COCO_SYNONYM_MAP:
            return cand_fe
        cand_f = w[:-3] + "f"
        if cand_f in COCO_SYNONYM_MAP:
            return cand_f
    # 3. -es -> "" or -e (e.g. couches -> couch, buses -> bus, glasses -> glass, horses -> horse)
    if w.endswith("es") and len(w) > 3:
        cand_base = w[:-2]
        if cand_base in COCO_SYNONYM_MAP:
            return cand_base
        cand_e = w[:-1]
        if cand_e in COCO_SYNONYM_MAP:
            return cand_e
    # 4. -s -> "" (e.g. laptops -> laptop, cups -> cup, dogs -> dog)
    if w.endswith("s") and not w.endswith("ss") and len(w) > 2:
        cand = w[:-1]
        if cand in COCO_SYNONYM_MAP:
            return cand
    return w


def _get_noun_plural_regex(noun: str) -> str:
    """Generate regex pattern matching singular and inflected plural forms of a noun."""
    words = noun.split()
    if not words:
        return re.escape(noun)
    lead = " ".join(words[:-1]) + " " if len(words) > 1 else ""
    last = words[-1]

    cand_forms = [last]
    if last.endswith("y") and len(last) > 2 and last[-2] not in "aeiou":
        cand_forms.append(last[:-1] + "ies")
    elif last.endswith("fe"):
        cand_forms.append(last[:-2] + "ves")
        cand_forms.append(last + "s")
    elif last.endswith("f") and not last.endswith("ff"):
        cand_forms.append(last[:-1] + "ves")
        cand_forms.append(last + "s")
    elif last.endswith(("s", "sh", "ch", "x", "z")):
        cand_forms.append(last + "es")
        cand_forms.append(last + "s")
    else:
        cand_forms.append(last + "s")
        cand_forms.append(last + "es")

    unique_forms = sorted(set(cand_forms), key=len, reverse=True)
    inner = "|".join(re.escape(f) for f in unique_forms)
    return re.escape(lead) + rf'(?:{inner})'


_PRECOMPILED_COCO_PATTERNS: list[tuple[str, str, re.Pattern[str], set[str]]] = []
for _noun, _coco_class in sorted(COCO_SYNONYM_MAP.items(), key=lambda x: len(x[0]), reverse=True):
    _pat_str = rf'\b(?:(a|an|one|two|three|four|five|six|seven|eight|nine|ten|multiple|several|many|group of|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|nhiều)\s+)?(?:(?!and\b|or\b)[\w-]+\s+)?{_get_noun_plural_regex(_noun)}\b'
    _last_w = _noun.split()[-1]
    _forms = {_last_w, _last_w + "s", _last_w + "es"}
    if _last_w.endswith("y") and len(_last_w) > 2:
        _forms.add(_last_w[:-1] + "ies")
    elif _last_w.endswith("fe"):
        _forms.add(_last_w[:-2] + "ves")
    elif _last_w.endswith("f"):
        _forms.add(_last_w[:-1] + "ves")
    _PRECOMPILED_COCO_PATTERNS.append((_noun, _coco_class, re.compile(_pat_str), _forms))

_BRAND_APPLE_RE = re.compile(r'\bapple\s+(?:store|watch|tv|mac|iphone|ipad|inc)\b')
_TWO_PERSONS_RE = re.compile(
    r'\b(?:a|an|one|the)\s+(?:person|man|woman|boy|girl|guy|lady|child|kid|baby|player|chef|speaker)\s+(?:and|with)\s+(?:a|an|one|the|another)\s+(?:person|man|woman|boy|girl|guy|lady|child|kid|baby|player|chef|speaker)\b'
)
_EN_TOKEN_RE = re.compile(r'\b[a-zA-Z]+\b')
_VI_WORD_RE = re.compile(r'\b[\wà-ỹ]+\b')


def decompose_query_clauses(query: str) -> list[str]:
    raw = query.strip()
    if not raw:
        return []

    # B11: Check for reverse temporal conjunctions: "after" (not "after that") or "sau khi"
    after_pattern = r'\b(?:after\s+(?!that\b|which\b)|sau\s+khi\b)'
    is_after = bool(re.search(after_pattern, raw, flags=re.IGNORECASE))

    if is_after:
        parts = re.split(after_pattern, raw, maxsplit=1, flags=re.IGNORECASE)
    else:
        split_pattern = (
            r'\b(?:and leads into|leading into|followed by|and then|then|after that|'
            r'before|trước khi|sau đó|rồi|tiếp theo|and(?=\s+\w+ing))\b|[;]'
        )
        parts = re.split(split_pattern, raw, flags=re.IGNORECASE)

    cleaned = []
    first_subject = ""
    for idx, p in enumerate(parts):
        p_clean = p.strip()
        words = p_clean.split()
        if len(words) < 2:
            continue
        if idx == 0 and len(words) >= 2:
            if words[0].lower() in ("a", "an", "the", "two", "three", "four", "five", "multiple"):
                first_subject = f"{words[0]} {words[1]}"
        elif idx > 0 and first_subject and words[0].lower().endswith("ing"):
            p_clean = f"{first_subject} {p_clean}"
        if p_clean.lower() != raw.lower():
            cleaned.append(p_clean)

    # B11: When conjunction is "after" / "sau khi", Clause 2 occurs before Clause 1 in time (t1 <= t2)
    if is_after and len(cleaned) == 2:
        cleaned = [cleaned[1], cleaned[0]]

    result = [raw]
    if len(cleaned) >= 2:
        for c in cleaned[:3]:
            if c not in result:
                result.append(c)
    return result


def extract_object_requirements(query: str) -> list[dict[str, Any]]:
    low = query.lower()
    # Exclude brand names before matching objects
    if "apple" in low:
        low = _BRAND_APPLE_RE.sub(' ', low)
    class_to_req: dict[str, dict[str, Any]] = {}
    query_words = set(_VI_WORD_RE.findall(low))

    # Pass 1: Fast pre-compiled regex matching for nouns in COCO_SYNONYM_MAP
    for noun, coco_class, compiled_pat, trig_forms in _PRECOMPILED_COCO_PATTERNS:
        if not (query_words & trig_forms):
            continue
        match = compiled_pat.search(low)
        if match:
            num_word = match.group(1)
            matched_text = match.group(0).strip()
            if num_word:
                cardinality = NUMBER_WORDS.get(num_word, 1)
                has_explicit_num = True
            else:
                last_word = matched_text.split()[-1]
                is_plural = (
                    last_word in INHERENT_PLURAL_NOUNS
                    or noun in INHERENT_PLURAL_NOUNS
                    or last_word.endswith(("ies", "ves"))
                    or (last_word.endswith("es") and not noun.endswith("es"))
                    or (last_word.endswith("s") and not noun.endswith("s"))
                )
                cardinality = 2 if is_plural else 1
                has_explicit_num = False

            new_req = {
                "class_name": coco_class,
                "cardinality": cardinality,
                "count": cardinality,
                "mention": matched_text,
                "_has_num": has_explicit_num,
            }

            if coco_class not in class_to_req:
                class_to_req[coco_class] = new_req
            else:
                old_req = class_to_req[coco_class]
                if cardinality > old_req["cardinality"]:
                    class_to_req[coco_class] = new_req
                elif cardinality == old_req["cardinality"] and has_explicit_num and not old_req["_has_num"]:
                    class_to_req[coco_class] = new_req

    # Pass 2: Fallback single-token singularization for any plural nouns missed
    for tok in _EN_TOKEN_RE.findall(low):
        sing = _singularize_coco_token(tok)
        if sing in COCO_SYNONYM_MAP:
            coco_class = COCO_SYNONYM_MAP[sing]
            num_pattern = rf'\b(a|an|one|two|three|four|five|six|seven|eight|nine|ten|multiple|several|many|group of|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|nhiều)\s+(?:(?!and\b|or\b)[\w-]+\s+)?{re.escape(tok)}\b'
            num_match = re.search(num_pattern, low)
            num_word = num_match.group(1) if num_match else None
            is_plural_tok = (tok != sing) or (tok in INHERENT_PLURAL_NOUNS)
            cardinality = NUMBER_WORDS.get(num_word, 2 if is_plural_tok else 1) if num_word else (2 if is_plural_tok else 1)
            new_req = {
                "class_name": coco_class,
                "cardinality": cardinality,
                "count": cardinality,
                "mention": num_match.group(0).strip() if num_match else tok,
                "_has_num": bool(num_word),
            }
            if coco_class not in class_to_req:
                class_to_req[coco_class] = new_req
            else:
                old_req = class_to_req[coco_class]
                if cardinality > old_req["cardinality"]:
                    class_to_req[coco_class] = new_req
                elif cardinality == old_req["cardinality"] and bool(num_word) and not old_req["_has_num"]:
                    class_to_req[coco_class] = new_req

    results = []
    # Check if two distinct singular person mentions are joined by "and" or "with" (e.g. "a lady and a man", "a man and a woman")
    if "person" in class_to_req and class_to_req["person"]["cardinality"] == 1:
        if _TWO_PERSONS_RE.search(low):
            class_to_req["person"]["cardinality"] = 2
            class_to_req["person"]["count"] = 2

    for req in class_to_req.values():
        req.pop("_has_num", None)
        results.append(req)
    return results


class RuleBasedQueryParser:
    """Parses Vietnamese and English natural language queries into valid QueryGraph schemas."""

    def decompose(self, query: str) -> list[str]:
        return decompose_query_clauses(query)

    def extract_objects(self, query: str) -> list[dict[str, Any]]:
        return extract_object_requirements(query)

    def parse(self, query: str) -> QueryGraph:
        raw_query = query.strip()
        low = raw_query.lower()

        is_vi = has_vietnamese_diacritics(raw_query) or any(
            w in low for w in ("người", "tìm", "sau đó", "áo", "chữ", "nói", "không")
        )
        lang = "vi" if is_vi else "en"

        entities: list[Entity] = []
        predicates: list[Predicate] = []
        ambiguities: list[str] = []
        entity_ids = set()

        def add_entity(e_id: str, e_type: str, mention_span: str, attrs=None, unc=None) -> str:
            if e_id in entity_ids:
                return e_id
            entities.append(
                Entity(
                    id=e_id,
                    type=e_type,
                    mention=mention_span,
                    attributes=attrs or [],
                    uncertainty=unc or [],
                )
            )
            entity_ids.add(e_id)
            return e_id

        # Polarity detection
        # Quoted content is evidence text, not a negation operator in the query.
        query_without_quotes = re.sub(r'(["\'])(.*?)\1', lambda m: " " * len(m.group(0)), raw_query)
        neg_m = re.search(r'\b(không|chưa|không có|not|no|without|never)\b', query_without_quotes, re.IGNORECASE)
        default_polarity = "negative" if neg_m else "positive"

        # 1. OCR text literal extraction
        quote_m = re.search(r'["\']([^"\']+)["\']', raw_query)
        ocr_entity_id = None
        if quote_m:
            full_span = quote_m.group(0)
            val = quote_m.group(1)
            ocr_entity_id = add_entity("sign", "text_region", full_span)
            predicates.append(
                Predicate(
                    id="sale_ocr",
                    kind="ocr",
                    operator="exact_text",
                    roles={"text_entity": ocr_entity_id},
                    value=val,
                    source_span=full_span,
                    polarity=default_polarity,
                    modality_options=[["ocr"]],
                )
            )
        else:
            kw_m = re.search(r'\b(chữ|biển|sign|text)\s+([A-Za-z0-9_%]+)\b', raw_query, re.IGNORECASE)
            if kw_m:
                full_span = kw_m.group(0)
                val = kw_m.group(2)
                ocr_entity_id = add_entity("sign", "text_region", full_span)
                predicates.append(
                    Predicate(
                        id="sale_ocr",
                        kind="ocr",
                        operator="exact_text",
                        roles={"text_entity": ocr_entity_id},
                        value=val,
                        source_span=full_span,
                        polarity=default_polarity,
                        modality_options=[["ocr"]],
                    )
                )

        # 2. Persons and clothing colors
        person_spans = []
        for m in re.finditer(r'(?:người\s+áo\s+(\w+)|person\s+in\s+(\w+))', raw_query, re.IGNORECASE):
            full_span = m.group(0)
            color_word = (m.group(1) or m.group(2)).lower()
            st_idx = m.start(1) if m.group(1) else m.start(2)
            end_idx = m.end(1) if m.group(1) else m.end(2)
            exact_color_span = raw_query[st_idx:end_idx]

            p_id = f"person_{len(person_spans) + 1}"
            person_spans.append(p_id)

            can_color = canonicalize_color(color_word)
            unc = []
            if color_word == "xanh":
                unc.append("áo xanh có thể blue hoặc green; cần giữ alternative nếu người dùng chưa xác định")
                ambiguities.append("áo xanh chưa phân biệt blue/green")

            attr = EntityAttribute(
                name="clothing_color",
                value=can_color,
                source_span=exact_color_span,
            )
            add_entity(p_id, "person", full_span, attrs=[attr], unc=unc)

        if not person_spans:
            generic_p = re.search(r'\b(người|someone|person|man|woman|girl|boy|lady|guy|people|children|men|women)\b', raw_query, re.IGNORECASE)
            if generic_p:
                p_id = "person_main"
                person_spans.append(p_id)
                add_entity(p_id, "person", generic_p.group(0))

        # 3. Object detection via COCO synonym map
        obj_entity_id = None
        extracted_objs = extract_object_requirements(raw_query)
        for idx, obj_req in enumerate(extracted_objs):
            if obj_req["class_name"] != "person":
                o_id = f"obj_{idx + 1}"
                if obj_entity_id is None:
                    obj_entity_id = o_id
                attrs = [
                    EntityAttribute(name="cardinality", value=str(obj_req["cardinality"]), source_span=obj_req["mention"]),
                    EntityAttribute(name="object_class", value=obj_req["class_name"], source_span=obj_req["mention"]),
                ]
                add_entity(o_id, "object", obj_req["mention"], attrs=attrs)

        # 4. Action detection
        action_m = re.search(r'\b(đưa|trao|tặng|gives?|hands?|chạy|runs?|đi bộ|walks?|nấu|cooks?|ăn|eats?|ngồi|sits?|cắt|cuts?)\b', raw_query, re.IGNORECASE)
        action_pred_id = None
        if action_m:
            act_word = action_m.group(1).lower()
            can_act = canonicalize_action(act_word)
            action_pred_id = "action_main"
            roles = {}
            if person_spans:
                roles["subject"] = person_spans[0]
            if obj_entity_id:
                roles["object"] = obj_entity_id
            if len(person_spans) > 1:
                roles["recipient"] = person_spans[1]

            predicates.append(
                Predicate(
                    id=action_pred_id,
                    kind="action",
                    operator=can_act,
                    roles=roles,
                    source_span=action_m.group(0),
                    polarity=default_polarity,
                    modality_options=[["visual"]],
                )
            )

        # 5. ASR speech detection
        asr_m = re.search(r'\b(nói|kêu|hét|says?|speaking|discussing)\s+([^,.;]+)', raw_query, re.IGNORECASE)
        asr_pred_id = None
        if asr_m:
            full_asr_span = asr_m.group(0)
            speech_val = asr_m.group(2).strip()
            asr_pred_id = "speak_asr"
            roles = {}
            if len(person_spans) > 1:
                roles["speaker"] = person_spans[1]
            elif person_spans:
                roles["speaker"] = person_spans[0]

            predicates.append(
                Predicate(
                    id=asr_pred_id,
                    kind="asr",
                    operator="semantic_phrase",
                    roles=roles,
                    value=speech_val,
                    source_span=full_asr_span,
                    polarity=default_polarity,
                    modality_options=[["asr", "audio_visual"]],
                )
            )

        # 6. Spatial relation detection
        sp_m = re.search(r'\b(phía sau|đằng sau|behind|phía trước|in front of)\b', raw_query, re.IGNORECASE)
        if sp_m:
            sp_phrase = sp_m.group(1).lower()
            can_sp = canonicalize_spatial(sp_phrase)
            if can_sp == "behind":
                ambiguities.append("phía sau chưa chỉ rõ sau người nào hay nền cảnh; không chốt 3D behind")

            roles = {}
            if ocr_entity_id:
                roles["subject"] = ocr_entity_id
            elif obj_entity_id:
                roles["subject"] = obj_entity_id

            if person_spans:
                roles["reference"] = person_spans[0]

            predicates.append(
                Predicate(
                    id="spatial_rel",
                    kind="spatial",
                    operator=can_sp,
                    roles=roles,
                    source_span=sp_m.group(0),
                    polarity=default_polarity,
                    modality_options=[["visual"]],
                )
            )

        # 7. Temporal ordering
        seq_m = re.search(r'\b(sau đó|rồi|and then|then)\b', raw_query, re.IGNORECASE)
        if seq_m:
            predicates.append(
                Predicate(
                    id="temporal_order",
                    kind="temporal",
                    operator="before",
                    roles={"event_a": action_pred_id or "action_main", "event_b": asr_pred_id or "speak_asr"},
                    source_span=seq_m.group(0),
                    polarity="positive",
                    modality_options=[["visual"]],
                )
            )

        # Build LogicNode tree compliant with QueryGraph schema
        p_ids = [p.id for p in predicates]
        if not p_ids:
            # Fallback action
            p_fallback = Predicate(
                id="action_main",
                kind="action",
                operator="generic_action",
                roles={"subject": "person_main"} if person_spans else {},
                source_span=raw_query,
                polarity=default_polarity,
            )
            predicates.append(p_fallback)
            p_ids.append("action_main")

        if neg_m and len(predicates) > 1:
            # The rule parser cannot safely bind a negative operator to one of
            # several predicates. Keep the graph but abstain on their polarity.
            ambiguities.append("negation_scope_unresolved")
            for predicate in predicates:
                predicate.uncertainty.append("negation_scope_unresolved")

        root_node = LogicNode(
            id="node_root",
            op="AND",
            children=p_ids,
        )

        meta = ParserMetadata(
            status="validated",
            model_revision="rule-based-2.0",
            prompt_version="qg-v1",
            confidence_kind="uncalibrated",
        )

        qg = QueryGraph(
            original_query=raw_query,
            language=lang,
            entities=entities,
            predicates=predicates,
            logic=[root_node],
            root_id="node_root",
            ambiguities=ambiguities,
            parser=meta,
            schema_version="1.0",
        )
        return qg
