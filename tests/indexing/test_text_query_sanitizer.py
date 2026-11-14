"""Unit tests for FTS5 lexical text query sanitizer and keyword extraction."""
import pytest
from videntra.indexing.text_store import (
    TextIndexStore,
    extract_fts5_keywords,
    build_fts5_query,
    ALL_STOPWORDS,
)


def test_extract_quoted_phrase():
    query = 'A racing car with text "TOXIC INVADERS" on the door'
    kws = extract_fts5_keywords(query)
    assert "TOXIC INVADERS" in kws
    # Ensure stopwords like 'with', 'on', 'the' are removed
    assert "with" not in [k.lower() for k in kws]
    assert "on" not in [k.lower() for k in kws]
    assert "the" not in [k.lower() for k in kws]


def test_stopword_removal_en_vi():
    query = "đoạn video này có một chiếc xe Toyota đang chạy rất nhanh"
    kws = extract_fts5_keywords(query)
    kws_lower = [k.lower() for k in kws]

    # Vietnamese stopwords filtered out
    assert "đoạn" not in kws_lower
    assert "video" not in kws_lower
    assert "này" not in kws_lower
    assert "có" not in kws_lower
    assert "một" not in kws_lower
    assert "đang" not in kws_lower
    assert "rất" not in kws_lower

    # Key nouns/proper nouns kept
    assert any("toyota" in k for k in kws_lower)


def test_build_fts5_query_syntax():
    query = 'Special query with symbols: (hello * world ^ 123 - "phrase test")'
    fts_q = build_fts5_query(query)

    # Should not contain raw unquoted symbols that crash SQLite FTS5
    assert "*" not in fts_q
    assert "^" not in fts_q
    assert "phrase test" in fts_q or '"phrase test"' in fts_q


def test_fts5_search_in_database():
    store = TextIndexStore(":memory:")
    store.add_entries(
        video_id="vid_001",
        source_type="ocr",
        entries=[
            {
                "id": "ocr_1",
                "text": "Monster Truck TOXIC INVADERS in the arena",
                "start_seconds": 10.0,
                "end_seconds": 15.0,
                "confidence": 0.95,
            },
            {
                "id": "ocr_2",
                "text": "Regular truck parking outside",
                "start_seconds": 20.0,
                "end_seconds": 25.0,
                "confidence": 0.85,
            },
        ],
    )

    # Search with special characters and stopwords
    hits = store.search('Find the "TOXIC INVADERS" (arena * ?)')
    assert len(hits) > 0
    assert hits[0].video_id == "vid_001"
    assert "TOXIC INVADERS" in hits[0].text
    store.close()


def test_person_and_quantity_stopwords_filtered():
    query = "A man and a woman are cooking two meals"
    kws = extract_fts5_keywords(query)
    kws_lower = [k.lower() for k in kws]
    assert "man" not in kws_lower
    assert "woman" not in kws_lower
    assert "two" not in kws_lower
    assert "cooking" in kws_lower or "meals" in kws_lower


def test_all_stopwords_returns_empty():
    query = "is there any in it and"
    kws = extract_fts5_keywords(query)
    assert kws == []
    fts_q = build_fts5_query(query)
    assert fts_q == ""


def test_prefix_query_expansion():
    query = "cooking dinner"
    fts_q_no_prefix = build_fts5_query(query, enable_prefix=False)
    assert "*" not in fts_q_no_prefix

    fts_q_prefix = build_fts5_query(query, enable_prefix=True)
    assert "cooking*" in fts_q_prefix
    assert "dinner*" in fts_q_prefix


def test_text_index_store_search_enable_prefix():
    store = TextIndexStore(":memory:")
    store.add_entries(
        video_id="vid_prefix",
        source_type="asr",
        entries=[
            {
                "id": "asr_p1",
                "text": "The chef is baking delicious cakes today",
                "start_seconds": 1.0,
                "end_seconds": 5.0,
                "confidence": 0.9,
            }
        ],
    )
    # Search with prefix on "delicio"
    hits = store.search("delicio", enable_prefix=True)
    assert len(hits) == 1
    assert hits[0].video_id == "vid_prefix"
    store.close()


def test_split_compound_tokens():
    from videntra.indexing.text_store import split_compound_tokens

    # CamelCase and ALLCAPS-to-Title/Lower
    assert split_compound_tokens("forBEAUTY") == "for BEAUTY"
    assert split_compound_tokens("B2cuteCupcakes") == "B2cute Cupcakes"
    assert split_compound_tokens("PRODUCTSfor") == "PRODUCTS for"

    # Numbers attached to words & comma separators
    assert split_compound_tokens("2019-20:1,558Positions") == "2019-20: 1558 Positions"
    assert split_compound_tokens("1,558") == "1558"
    assert split_compound_tokens("12,345,678") == "12345678"

    # Preserved scientific and short notations
    assert split_compound_tokens("H2O2") == "H2O2"
    assert split_compound_tokens("IO3") == "IO3"
    assert split_compound_tokens("B2") == "B2"
    assert split_compound_tokens("F1") == "F1"

    # Empty and edge cases
    assert split_compound_tokens("") == ""
    assert split_compound_tokens(None) == ""
    assert split_compound_tokens("   ") == ""


def test_generic_visual_words_filtering():
    from videntra.indexing.text_store import GENERIC_VISUAL_WORDS

    # Ensure standard low-specificity action and object words are present
    common_actions = {"make", "makes", "making", "cut", "cuts", "cutting", "hold", "sit", "stand", "walk", "run", "play"}
    common_objects = {"room", "camera", "table", "street", "road", "water", "background", "front"}
    assert common_actions.issubset(GENERIC_VISUAL_WORDS)
    assert common_objects.issubset(GENERIC_VISUAL_WORDS)

    # Ensure specific entity and domain words are NOT in GENERIC_VISUAL_WORDS
    assert "pizza" not in GENERIC_VISUAL_WORDS
    assert "sandwich" not in GENERIC_VISUAL_WORDS
    assert "president" not in GENERIC_VISUAL_WORDS
    assert "huawei" not in GENERIC_VISUAL_WORDS
    assert "toyota" not in GENERIC_VISUAL_WORDS
    assert "azerbaijan" not in GENERIC_VISUAL_WORDS


def test_proper_noun_phrase_length_gating():
    from videntra.indexing.text_store import extract_fts5_keywords, build_fts5_query

    # 1. Exactly 2 words -> retained as 2-word phrase
    kws_2w = extract_fts5_keywords("A live report from Fox News today")
    assert "Fox News" in kws_2w

    kws_2w_person = extract_fts5_keywords("An interview with Joaquin Phoenix on cinema")
    assert "Joaquin Phoenix" in kws_2w_person

    # 2. Chains >= 3 words -> split into individual tokens and trailing 2-word phrase
    kws_long = extract_fts5_keywords("Coverage from Virginia Democratic Party Fox News regarding elections")
    # Must NOT contain the 5-word rigid phrase
    assert "Virginia Democratic Party Fox News" not in kws_long
    # Must contain individual proper noun tokens and trailing 2-word phrase
    assert "Fox News" in kws_long
    assert "Virginia" in kws_long
    assert "Democratic" in kws_long
    assert "Party" in kws_long

    # 3. Ensure FTS5 query does not produce rigid 5-word quote
    fts_q = build_fts5_query("Virginia Democratic Party Fox News")
    assert '"Virginia Democratic Party Fox News"' not in fts_q
    assert '"Fox News"' in fts_q
    assert '"Virginia"' in fts_q


def test_soundex_and_levenshtein_distance():
    from videntra.indexing.text_store import soundex, levenshtein_distance

    # Soundex phonetic equivalence
    assert soundex("Pence") == soundex("Pense")
    assert soundex("Smith") == soundex("Smyth")
    assert soundex("Robert") == soundex("Rupert")
    assert soundex("") == ""

    # Levenshtein distance
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("pence", "pense") == 1
    assert levenshtein_distance("espn", "espn") == 0
    assert levenshtein_distance("", "test") == 4


def test_sqlite_fuzzy_and_subword_fallback_when_exact_fails():
    from videntra.indexing.text_store import TextIndexStore

    store = TextIndexStore(":memory:")
    store.add_entries(
        video_id="vid_pence",
        source_type="asr",
        entries=[
            {
                "id": "asr_01",
                "text": "Vice President Mike Pence addresses the press conference",
                "start_seconds": 12.0,
                "end_seconds": 24.0,
                "confidence": 0.95,
            }
        ],
    )
    store.add_entries(
        video_id="vid_espn",
        source_type="ocr",
        entries=[
            {
                "id": "ocr_01",
                "text": "ESPN SportsCenter Championship Highlights Live",
                "start_seconds": 2.0,
                "end_seconds": 10.0,
                "confidence": 0.90,
            }
        ],
    )

    # 1. Exact query matches via FTS5
    hits_exact = store.search("Mike Pence", enable_fuzzy_fallback=False)
    assert len(hits_exact) == 1
    assert hits_exact[0].video_id == "vid_pence"

    # 2. Phonetic/spelling error: "Maik Pense" -> exact FTS5 returns 0, but fuzzy fallback rescues
    hits_exact_miss = store.search("Maik Pense", enable_fuzzy_fallback=False)
    assert len(hits_exact_miss) == 0

    hits_fuzzy = store.search("Maik Pense", enable_fuzzy_fallback=True)
    assert len(hits_fuzzy) >= 1
    assert hits_fuzzy[0].video_id == "vid_pence"
    assert "Mike Pence" in hits_fuzzy[0].text

    # 3. Subword / substring fallback: "SportsCenter" -> "sportscenter"
    hits_sub = store.search("SportCenter", enable_fuzzy_fallback=True)
    assert len(hits_sub) >= 1
    assert hits_sub[0].video_id == "vid_espn"

    store.close()



