"""In-memory LRU cache for parsed QueryGraph instances."""
from collections import OrderedDict
import hashlib
from typing import Any

from .schemas import QueryGraph
from videntra.evidence.normalize import normalize_text


class QueryGraphCache:
    """Thread-safe-like LRU cache avoiding redundant parsing of repeated queries."""

    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self._cache: OrderedDict[str, QueryGraph] = OrderedDict()

    def _make_key(self, query: str, model_revision: str | None = None) -> str:
        norm = normalize_text(query).lower()
        token = f"{norm}:{model_revision or 'rules'}"
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def get(self, query: str, model_revision: str | None = None) -> QueryGraph | None:
        key = self._make_key(query, model_revision)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, query: str, graph: QueryGraph, model_revision: str | None = None) -> None:
        key = self._make_key(query, model_revision)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = graph
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
