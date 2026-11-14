"""Unicode normalization and Vietnamese language heuristics for evidence text."""
import re
import unicodedata

_VIETNAMESE_DIACRITICS_PATTERN = re.compile(
    r"[àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ"
    r"ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ]"
)


def normalize_text(text: str | None) -> str:
    """Normalize text to Unicode NFC, collapse whitespace, and strip control characters."""
    if not text:
        return ""
    # 1. NFC normalization
    nfc = unicodedata.normalize("NFC", text)
    # 2. Filter out non-printable control characters (except common space)
    filtered = "".join(ch for ch in nfc if unicodedata.category(ch)[0] != "C" or ch == " ")
    # 3. Collapse multiple whitespaces
    collapsed = re.sub(r"\s+", " ", filtered).strip()
    return collapsed


def contains_text_phrase(text: str | None, phrase: str | None) -> bool:
    """Match a normalized phrase without matching inside a larger word."""
    haystack = normalize_text(text).casefold()
    needle = normalize_text(phrase).casefold()
    if not haystack or not needle:
        return False
    left = r"(?<!\w)" if needle[0].isalnum() or needle[0] == "_" else ""
    right = r"(?!\w)" if needle[-1].isalnum() or needle[-1] == "_" else ""
    return re.search(left + re.escape(needle) + right, haystack) is not None


def has_vietnamese_diacritics(text: str | None) -> bool:
    """Check if the text contains Vietnamese-specific tonal diacritics or modified characters."""
    if not text:
        return False
    norm = unicodedata.normalize("NFC", text)
    return bool(_VIETNAMESE_DIACRITICS_PATTERN.search(norm))


def fold_diacritics(text: str | None) -> str:
    """Fold Vietnamese diacritics and tones to base ASCII characters for unaccented search."""
    if not text:
        return ""
    # Replace special Vietnamese characters that NFD does not decompose into ASCII
    res = text.replace("đ", "d").replace("Đ", "D")
    # Decompose Unicode characters into base + combining diacritics
    nfd = unicodedata.normalize("NFD", res)
    # Remove combining marks
    stripped = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped)
