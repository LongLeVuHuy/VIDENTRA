"""Lexical text index store using SQLite FTS5 for OCR and ASR transcript retrieval."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Sequence

from videntra.evidence.normalize import normalize_text

EN_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "at", "on", "in", "to", "of",
    "for", "with", "by", "and", "or", "but", "showing", "seen", "view", "clip",
    "video", "shows", "shown", "appears", "this", "that", "these", "those", "from",
    "there", "here", "it", "its", "as", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "about", "into", "through", "during", "before", "after",
    "above", "below", "up", "down", "out", "off", "over", "under", "again", "further",
    "then", "once", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "just", "now", "while", "also", "many", "their",
    "who", "whom", "which", "her", "his", "him", "she", "he",
    "man", "men", "woman", "women", "person", "people", "guy", "guys", "boy", "boys",
    "girl", "girls", "someone", "somebody", "something", "anyone", "anybody", "anything",
    "everyone", "everybody", "everything", "one", "two", "three", "four", "five",
    "other", "others", "another", "any", "all", "every", "each", "both", "few", "more", "most",
})

VI_STOPWORDS: frozenset[str] = frozenset({
    "của", "và", "các", "những", "là", "ở", "trong", "trên", "tại", "với", "cho",
    "để", "có", "video", "đoạn", "cảnh", "thấy", "này", "đó", "kia", "được", "bị",
    "bởi", "ra", "vào", "lại", "nhiều", "ít", "rất", "quá", "đang", "đã", "sẽ",
    "một", "hai", "ba", "bốn", "năm", "người", "khi", "lúc", "như", "thì", "mà",
    "gì", "nào", "đâu", "ai", "sao", "thế",
})

ALL_STOPWORDS: frozenset[str] = EN_STOPWORDS | VI_STOPWORDS

GENERIC_VISUAL_WORDS: frozenset[str] = frozenset({
    "make", "makes", "making", "made",
    "cut", "cuts", "cutting",
    "hold", "holds", "holding",
    "sit", "sits", "sitting",
    "stand", "stands", "standing",
    "walk", "walks", "walking",
    "run", "runs", "running",
    "play", "plays", "playing",
    "eat", "eats", "eating",
    "talk", "talks", "talking",
    "speak", "speaks", "speaking",
    "show", "shows", "showing",
    "look", "looks", "looking",
    "use", "uses", "using",
    "wear", "wears", "wearing",
    "drive", "drives", "driving",
    "ride", "rides", "riding",
    "put", "puts", "putting",
    "give", "gives", "giving",
    "take", "takes", "taking",
    "get", "gets", "getting",
    "go", "goes", "going",
    "come", "comes", "coming",
    "do", "does", "doing",
    "room", "background", "front", "camera", "table", "street", "road", "water", "house", "home",
    "day", "night", "time", "way", "thing", "things",
    "different", "young", "old", "little", "small", "big", "large",
    "white", "black", "blue", "red", "green",
    "cup", "cups", "glass", "dish", "food", "foods", "car", "dog", "cat", "bird",
    "baby", "child", "children", "childrens", "kids", "movie", "film", "trailer",
    "trailor", "commercial", "advertisement", "channel", "website", "show", "tv",
    "music", "song", "dance", "sing", "singing", "live", "band", "stage",
    "performance", "performing", "game", "phone", "screen", "computer", "laptop",
    "desk", "chair", "chairs", "couch", "bed", "hand", "hands", "face", "hair",
    "eye", "eyes", "dress", "shirt", "suit", "top", "uniform", "product", "products",
    "new", "good", "important", "yellow", "brown", "purple", "dark",
})


VI_LOWER: str = "a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ"
VI_UPPER: str = "A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ"

_RE_HAS_DIGIT = re.compile(r"\d")
_RE_HAS_UPPER = re.compile(rf"[{VI_UPPER}]")
_RE_THOUSAND_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_RE_COLON_DIGIT = re.compile(r":(?=\d)")
_RE_DIGIT_WORD = re.compile(rf"(\b\d+)([{VI_LOWER}{VI_UPPER}]+)")
_RE_WORD_DIGIT = re.compile(rf"([{VI_LOWER}{VI_UPPER}]{{3,}})(\d+\b)")
_RE_CAMEL_CASE = re.compile(rf"([{VI_LOWER}])([{VI_UPPER}])")
_RE_ALLCAPS_LOWER = re.compile(rf"([{VI_UPPER}]{{2,}})([{VI_LOWER}]{{2,}})")
_RE_MULTI_SPACE = re.compile(r"[ \t]+")
_RE_NON_ALNUM_SPACE = re.compile(r"[^\w\s]")
_RE_QUOTED = re.compile(r'["\']([^"\']+)["\']')
_RE_PROPER_NOUN = re.compile(rf"\b[{VI_UPPER}][{VI_LOWER}0-9]+(?:\s+[{VI_UPPER}][{VI_LOWER}0-9]+)*\b")
_RE_WORD_ALL = re.compile(r"\w+")


def split_compound_tokens(text: str) -> str:
    """Split compound tokens from OCR/ASR:
    - CamelCase (e.g. forBEAUTY -> for BEAUTY, B2cuteCupcakes -> B2cute Cupcakes)
    - ALLCAPS to lowercase/TitleCase (e.g. PRODUCTSfor -> PRODUCTS for)
    - Numbers attached to words (e.g. 2019-20:1,558Positions -> 2019-20: 1558 Positions)
    - Thousand-separator commas in numbers (e.g. 1,558 -> 1558)
    - Preserves scientific and short notations (e.g. H2O2, IO3, B2, F1).
    """
    if not text:
        return ""

    result = str(text)

    if _RE_HAS_DIGIT.search(result):
        # 1. Remove thousand-separator commas in numbers (e.g. 1,558 -> 1558)
        result = _RE_THOUSAND_COMMA.sub("", result)
        # 2. Add space after colon before digits (e.g. 2019-20:1558 -> 2019-20: 1558)
        result = _RE_COLON_DIGIT.sub(": ", result)
        # 3. Numbers attached to words at word boundaries (e.g. 1558Positions -> 1558 Positions)
        result = _RE_DIGIT_WORD.sub(r"\1 \2", result)
        # Words of length >= 3 followed by numbers at end of word (e.g. Positions1558 -> Positions 1558)
        result = _RE_WORD_DIGIT.sub(r"\1 \2", result)

    if _RE_HAS_UPPER.search(result):
        # 4. Split CamelCase: lowercase followed by uppercase (e.g. cuteCupcakes -> cute Cupcakes, forBEAUTY -> for BEAUTY)
        result = _RE_CAMEL_CASE.sub(r"\1 \2", result)
        # 5. Split ALLCAPS followed by lowercase (e.g. PRODUCTSfor -> PRODUCTS for)
        result = _RE_ALLCAPS_LOWER.sub(r"\1 \2", result)

    # Normalize multiple whitespace
    return _RE_MULTI_SPACE.sub(" ", result).strip()


def sanitize_fts5_token(token: str) -> str:
    """Strip FTS5 special characters and punctuation."""
    return _RE_NON_ALNUM_SPACE.sub("", token).strip()


def extract_fts5_keywords(raw_query: str, max_keywords: int = 12) -> list[str]:
    """Extract prioritized keywords for SQLite FTS5 search.

    Priority:
    1. Quoted phrases ("TOXIC INVADERS" or 'TOXIC INVADERS')
    2. Proper nouns (Capitalized / TitleCase words inside the sentence):
       - Exactly 2 words -> retain as quoted phrase
       - Chains >= 3 words -> split into individual tokens and trailing 2-word phrase
    3. Content words (nouns/verbs) after bilingual EN-VI stopword filtering
    """
    clean_q = split_compound_tokens(normalize_text(raw_query))
    if not clean_q:
        return []

    results: list[str] = []
    seen: set[str] = set()

    # Priority 1: Quoted phrases
    quoted_matches = _RE_QUOTED.findall(clean_q)
    for qm in quoted_matches:
        sanitized = sanitize_fts5_token(qm)
        if sanitized and sanitized.lower() not in seen:
            results.append(sanitized)
            seen.add(sanitized.lower())
            for sub_tok in sanitized.split():
                seen.add(sub_tok.lower())

    # Remove quoted parts from remaining text to avoid duplicate extraction
    unquoted = _RE_QUOTED.sub(" ", clean_q) if quoted_matches else clean_q

    # Priority 2: Consecutive Capitalized Words / Proper Nouns (e.g. Toxic Invaders, Ha Noi)
    proper_nouns = _RE_PROPER_NOUN.findall(unquoted)
    for pn in proper_nouns:
        sanitized = sanitize_fts5_token(pn)
        if not sanitized:
            continue
        words = sanitized.split()
        if len(words) == 1:
            w = words[0]
            if w.lower() not in seen and w.lower() not in ALL_STOPWORDS:
                results.append(w)
                seen.add(w.lower())
        elif len(words) == 2:
            if sanitized.lower() not in seen and sanitized.lower() not in ALL_STOPWORDS:
                results.append(sanitized)
                seen.add(sanitized.lower())
            for w in words:
                if w.lower() not in seen and w.lower() not in ALL_STOPWORDS:
                    results.append(w)
                    seen.add(w.lower())
        else:
            # len(words) >= 3: split into individual tokens and trailing 2-word phrase
            trailing_phrase = f"{words[-2]} {words[-1]}"
            for w in words:
                if w.lower() not in seen and w.lower() not in ALL_STOPWORDS:
                    results.append(w)
                    seen.add(w.lower())
            if trailing_phrase.lower() not in seen:
                results.append(trailing_phrase)
                seen.add(trailing_phrase.lower())

    # Priority 3: Remaining individual tokens filtered by stopwords
    tokens = _RE_WORD_ALL.findall(unquoted)
    for tok in tokens:
        sanitized = sanitize_fts5_token(tok)
        tok_lower = sanitized.lower()
        if (
            sanitized
            and len(sanitized) > 1
            and tok_lower not in ALL_STOPWORDS
            and tok_lower not in seen
        ):
            results.append(sanitized)
            seen.add(tok_lower)

    return results[:max_keywords]


def build_fts5_query(raw_query: str, max_keywords: int = 12, enable_prefix: bool = False) -> str:
    """Build a robust, injection-safe FTS5 query string with optional prefix matching."""
    keywords = extract_fts5_keywords(raw_query, max_keywords=max_keywords)
    if not keywords:
        return ""

    escaped_tokens = []
    for kw in keywords:
        # If kw is multi-word phrase, quote it as a phrase: "toxic invaders"
        clean = re.sub(r'["\*\^\:\-\(\)]', '', kw).strip()
        if clean:
            if " " in clean:
                escaped_tokens.append(f'"{clean}"')
            else:
                escaped_tokens.append(f'"{clean}"')
                if enable_prefix and len(clean) >= 5:
                    escaped_tokens.append(f'{clean}*')

    return " OR ".join(escaped_tokens)


@dataclass(frozen=True)
class TextHit:
    hit_id: str
    video_id: str
    source_type: str  # "ocr" or "asr"
    text: str
    start_seconds: float
    end_seconds: float
    start_ordinal: int | None
    end_ordinal: int | None
    confidence: float
    score: float  # BM25 ranking score


def soundex(token: str) -> str:
    """Computes American Soundex code for a phonetic word."""
    token = re.sub(r'[^a-zA-Z]', '', token).upper()
    if not token:
        return ""
    mapping = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }
    first_letter = token[0]
    codes = [first_letter]
    prev = mapping.get(first_letter, "0")
    for ch in token[1:]:
        code = mapping.get(ch, "0")
        if code != "0" and code != prev:
            codes.append(code)
            if len(codes) == 4:
                break
        prev = code
    while len(codes) < 4:
        codes.append("0")
    return "".join(codes)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1] + [0] * len(s2)
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row[j + 1] = min(insertions, deletions, substitutions)
        prev_row = curr_row
    return prev_row[-1]


class TextIndexStore:
    """SQLite FTS5 full-text search index for OCR tracks and ASR segments."""

    def __init__(self, db_path: Path | str = ":memory:"):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS text_entries USING fts5(
                    hit_id UNINDEXED,
                    video_id UNINDEXED,
                    source_type,
                    text,
                    start_seconds UNINDEXED,
                    end_seconds UNINDEXED,
                    start_ordinal UNINDEXED,
                    end_ordinal UNINDEXED,
                    confidence UNINDEXED,
                    tokenize='unicode61 remove_diacritics 0'
                );
            """)

    def add_entries(
        self,
        video_id: str,
        source_type: str,
        entries: Sequence[dict],
    ) -> int:
        """Insert text entries into FTS5 index.
        
        entries: list of dicts with keys: id, text, start_seconds, end_seconds, start_ordinal, end_ordinal, confidence
        """
        rows = []
        for e in entries:
            clean = normalize_text(e.get("text", ""))
            if not clean:
                continue
            rows.append((
                str(e.get("id")),
                str(video_id),
                str(source_type),
                clean,
                float(e.get("start_seconds", 0.0)),
                float(e.get("end_seconds", 0.0)),
                int(e.get("start_ordinal", -1)),
                int(e.get("end_ordinal", -1)),
                float(e.get("confidence", 1.0)),
            ))

        if not rows:
            return 0

        with self._conn:
            self._conn.executemany("""
                INSERT INTO text_entries (
                    hit_id, video_id, source_type, text,
                    start_seconds, end_seconds, start_ordinal, end_ordinal, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, rows)

        return len(rows)

    def fuzzy_search(
        self,
        query: str,
        source_type: str | None = None,
        top_k: int = 50,
        min_similarity: float = 0.65,
    ) -> list[TextHit]:
        """Subword, trigram, and phonetic fallback search when exact FTS5 returns 0 results.
        
        Uses substring LIKE patterns and phonetic soundex / edit-distance matching
        to rescue slightly misspelled, phonetic-shifted, or subword-variant queries.
        """
        keywords = extract_fts5_keywords(query, max_keywords=8)
        if not keywords:
            keywords = [w for w in re.findall(r'\w+', query.lower()) if len(w) >= 3 and w not in ALL_STOPWORDS][:6]
        if not keywords:
            return []

        # Build candidate retrieval using SQLite LIKE OR conditions on text_entries
        like_clauses = []
        like_params = []
        for kw in keywords:
            clean = sanitize_fts5_token(kw)
            if len(clean) >= 3:
                like_clauses.append("text LIKE ?")
                like_params.append(f"%{clean}%")

        candidates = []
        if like_clauses:
            sql = f"""
                SELECT hit_id, video_id, source_type, text,
                       start_seconds, end_seconds, start_ordinal, end_ordinal, confidence
                FROM text_entries
                WHERE ({" OR ".join(like_clauses)})
            """
            params: list[Any] = list(like_params)
            if source_type:
                sql += " AND source_type = ?"
                params.append(source_type)
            sql += " LIMIT 200"
            cursor = self._conn.execute(sql, params)
            candidates = cursor.fetchall()

        # If no substring LIKE matches, expand search across distinct text entries in table (limit 300)
        if not candidates:
            fallback_sql = """
                SELECT hit_id, video_id, source_type, text,
                       start_seconds, end_seconds, start_ordinal, end_ordinal, confidence
                FROM text_entries
            """
            fb_params = []
            if source_type:
                fallback_sql += " WHERE source_type = ?"
                fb_params.append(source_type)
            fallback_sql += " LIMIT 300"
            cursor = self._conn.execute(fallback_sql, fb_params)
            candidates = cursor.fetchall()

        # Score candidates with subword / soundex / edit distance
        kw_clean = [sanitize_fts5_token(k).lower() for k in keywords if len(k) >= 3]
        kw_soundex = {k: soundex(k) for k in kw_clean}

        scored_hits = []
        for row in candidates:
            text = str(row[3]).lower()
            text_tokens = re.findall(r'\w+', text)
            if not text_tokens:
                continue

            text_soundex_set = {soundex(t) for t in text_tokens if len(t) >= 3}
            best_sim = 0.0

            for q_tok in kw_clean:
                # 1. Exact substring match
                if q_tok in text:
                    best_sim = max(best_sim, 1.0)
                    continue
                # 2. Soundex phonetic match
                q_sx = kw_soundex.get(q_tok)
                if q_sx and q_sx in text_soundex_set:
                    best_sim = max(best_sim, 0.85)
                # 3. Levenshtein edit distance with closest token
                q_len = len(q_tok)
                for t_tok in text_tokens:
                    t_len = len(t_tok)
                    if abs(q_len - t_len) <= 2:
                        dist = levenshtein_distance(q_tok, t_tok)
                        sim = 1.0 - dist / max(q_len, t_len)
                        if sim > best_sim:
                            best_sim = sim

            if best_sim >= min_similarity:
                conf = float(row[8])
                score = round(best_sim * (0.60 + 0.40 * conf), 4)
                st_ord = int(row[6]) if int(row[6]) >= 0 else None
                end_ord = int(row[7]) if int(row[7]) >= 0 else None
                scored_hits.append(
                    TextHit(
                        hit_id=row[0],
                        video_id=row[1],
                        source_type=row[2],
                        text=row[3],
                        start_seconds=float(row[4]),
                        end_seconds=float(row[5]),
                        start_ordinal=st_ord,
                        end_ordinal=end_ord,
                        confidence=conf,
                        score=score,
                    )
                )

        scored_hits.sort(key=lambda x: x.score, reverse=True)
        return scored_hits[:top_k]

    def search(
        self,
        query: str,
        source_type: str | None = None,
        top_k: int = 50,
        enable_prefix: bool = False,
        enable_fuzzy_fallback: bool = True,
    ) -> list[TextHit]:
        """Search text index using FTS5 BM25 relevance scoring with optional fuzzy fallback."""
        fts_query = build_fts5_query(query, enable_prefix=enable_prefix)
        hits = []
        if fts_query:
            sql = """
                SELECT hit_id, video_id, source_type, text,
                       start_seconds, end_seconds, start_ordinal, end_ordinal, confidence,
                       bm25(text_entries) as bm25_rank
                FROM text_entries
                WHERE text_entries MATCH ?
            """
            params: list[Any] = [fts_query]

            if source_type:
                sql += " AND source_type = ?"
                params.append(source_type)

            # FTS5 bm25 produces negative numbers where more negative is more relevant
            sql += " ORDER BY bm25_rank ASC LIMIT ?"
            params.append(top_k)

            cursor = self._conn.execute(sql, params)
            for row in cursor.fetchall():
                # Invert BM25 negative score to positive ranking score
                raw_bm25 = abs(float(row[9]))
                # Fallback to base score when document count N is very small in SQLite IDF
                score = round(raw_bm25 if raw_bm25 > 1e-4 else (0.5 + float(row[8]) * 0.5), 4)
                st_ord = int(row[6]) if int(row[6]) >= 0 else None
                end_ord = int(row[7]) if int(row[7]) >= 0 else None
                hits.append(
                    TextHit(
                        hit_id=row[0],
                        video_id=row[1],
                        source_type=row[2],
                        text=row[3],
                        start_seconds=float(row[4]),
                        end_seconds=float(row[5]),
                        start_ordinal=st_ord,
                        end_ordinal=end_ord,
                        confidence=float(row[8]),
                        score=score,
                    )
                )

        # P3 Fallback: When exact FTS5 returns 0 hits, trigger subword/fuzzy fallback
        if not hits and enable_fuzzy_fallback:
            hits = self.fuzzy_search(query=query, source_type=source_type, top_k=top_k)

        return hits

    def close(self) -> None:
        self._conn.close()
