"""Canonicalization mappings for predicates, spatial roles, and colors."""
from typing import Any

ACTION_MAP = {
    "đưa": "give",
    "trao": "give",
    "tặng": "give",
    "chuyển": "give",
    "give": "give",
    "gives": "give",
    "hand": "give",
    "hands": "give",
    "chạy": "run",
    "run": "run",
    "runs": "run",
    "đi bộ": "walk",
    "walk": "walk",
    "walks": "walk",
    "nói": "speak",
    "speak": "speak",
    "speaks": "speak",
    "say": "speak",
    "says": "speak",
}

SPATIAL_MAP = {
    "phía sau": "behind",
    "đằng sau": "behind",
    "ở sau": "behind",
    "behind": "behind",
    "phía trước": "in_front_of",
    "đằng trước": "in_front_of",
    "in front of": "in_front_of",
    "bên cạnh": "beside",
    "kế bên": "beside",
    "beside": "beside",
    "next to": "beside",
}

COLOR_MAP = {
    "đỏ": "red",
    "red": "red",
    "vàng": "yellow",
    "yellow": "yellow",
    "trắng": "white",
    "white": "white",
    "đen": "black",
    "black": "black",
    "xanh": "blue",  # Annotated with semantic ambiguity (blue/green)
    "blue": "blue",
    "green": "green",
}


def canonicalize_action(word: str) -> str:
    return ACTION_MAP.get(word.lower().strip(), word.lower().strip())


def canonicalize_spatial(phrase: str) -> str:
    return SPATIAL_MAP.get(phrase.lower().strip(), phrase.lower().strip())


def canonicalize_color(word: str) -> str:
    return COLOR_MAP.get(word.lower().strip(), word.lower().strip())


def canonicalize_predicates(predicates: list[Any]) -> list[Any]:
    """In-place or shallow normalization of predicate operators."""
    for p in predicates:
        if getattr(p, "kind", None) == "action":
            p.operator = canonicalize_action(p.operator)
        elif getattr(p, "kind", None) == "spatial":
            p.operator = canonicalize_spatial(p.operator)
    return predicates
