"""
Indic Text Cleaning, Phonetic De-Obfuscation, and Colloquial Number Normalizer.

Designed for ultra-low latency (<0.5ms per transcript) using pre-compiled regular
expressions and single-pass token collapsing for intentionally spaced/punctuated
attack tokens (e.g., "a.p.k" -> "apk", "b-i-j-l-i" -> "bijli",
"q u i c k s u p p o r t" -> "quicksupport").
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Dict, List, Set, Tuple

# Zero-width and invisible formatting characters often injected by scammers in SMS/WhatsApp
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u00ad]+")

# Devanagari to Hinglish/Roman phonetic transliteration table for common fraud terms
_DEVANAGARI_PHRASE_MAP: Tuple[Tuple[str, str], ...] = (
    ("बिजली कट", "bijli cut"),
    ("बिजली का बिल", "bijli ka bill"),
    ("बिजली", "bijli"),
    ("तुरंत", "turant"),
    ("फटाफट", "fatafat"),
    ("जल्दी करो", "jaldi karo"),
    ("दस मिनट", "10 minute"),
    ("दो घंटे", "2 ghante"),
    ("आज रात नौ बजे", "aaj raat 9 baje"),
    ("आज रात 9 बजे", "aaj raat 9 baje"),
    ("अकाउंट फ्रीज", "account freeze"),
    ("सिम ब्लॉक", "sim block"),
    ("डिजिटल अरेस्ट", "digital arrest"),
    ("पुलिस", "police"),
    ("सीबीआई", "cbi"),
    ("कस्टम्स", "customs"),
    ("कस्टमर सपोर्ट", "customer support"),
    ("स्क्रीन शेयर", "screen share"),
    ("ऐप डाउनलोड करो", "app download karo"),
    ("फॉरवर्ड करो", "forward karo"),
    ("ओटीपी", "otp"),
    ("यूपीआई पिन", "upi pin"),
    ("केवाईसी अपडेट", "kyc update"),
    ("पैन कार्ड", "pan card"),
)

# Dotted, hyphenated, or underscore-separated single-character sequences:
# Matches e.g. "a.p.k", "b-i-j-l-i", "o_t_p", "c.b.i", "u.p.i"
_PUNCTUATED_CHARS_RE = re.compile(
    r"(?<![a-zA-Z0-9])([a-zA-Z])(?:[\.\-_\/\\+*~|]+([a-zA-Z])){1,20}(?![a-zA-Z0-9])"
)
_INNER_PUNCT_RE = re.compile(r"[\.\-_\/\\+*~|]+")

# Currency normalization patterns (e.g., "rs. 5", "inr 5", "rupaye 5" -> "₹5")
_RUPEE_PREFIX_RE = re.compile(r"\b(?:rs\.?|inr|rupaye|rupee|rupees)\s*(\d+)\b", re.IGNORECASE)
_RUPEE_SUFFIX_RE = re.compile(r"\b(\d+)\s*(?:rs\.?|inr|rupaye|rupee|rupees)\b", re.IGNORECASE)

# General punctuation stripping (preserve '.', '₹', and alphanumeric/Devanagari characters initially)
# Note: we preserve '.apk' specifically when followed by apk, otherwise strip punctuation to spaces.
_NON_WORD_PUNCT_RE = re.compile(r"[^\w\s₹\.]", re.UNICODE)
_STANDALONE_DOTS_RE = re.compile(r"\.(?!apk\b)", re.IGNORECASE)
_MULTI_SPACE_RE = re.compile(r"\s+")

# Colloquial Hindi/Bhojpuri number words to digits (sorted longest-first where needed)
_COLLOQUIAL_NUMBER_PHRASES: Tuple[Tuple[re.Pattern[str], str], ...] = (
    # Multi-word colloquial time/quantity expressions first
    (re.compile(r"\b(?:dus|das|ten)\s+(?:minute|min|minat|mint)\b", re.IGNORECASE), "10 minute"),
    (re.compile(r"\b(?:do|two|du|dui)\s+(?:ghante|ghanta|hour|hours|ghnta)\b", re.IGNORECASE), "2 ghante"),
    (re.compile(r"\b(?:ek|one)\s+(?:ghante|ghanta|hour)\b", re.IGNORECASE), "1 ghante"),
    (re.compile(r"\b(?:paanch|panch|five)\s+(?:minute|min|minat|mint)\b", re.IGNORECASE), "5 minute"),
    (re.compile(r"\b(?:pandra|pandrah|fifteen)\s+(?:minute|min|minat)\b", re.IGNORECASE), "15 minute"),
    (re.compile(r"\b(?:bees|bis|twenty)\s+(?:minute|min|minat)\b", re.IGNORECASE), "20 minute"),
    (re.compile(r"\b(?:tees|tis|thirty)\s+(?:minute|min|minat)\b", re.IGNORECASE), "30 minute"),
    (re.compile(r"\baaj\s+raat\s+(?:nau|no|nine)\s+baje\b", re.IGNORECASE), "aaj raat 9 baje"),
    (re.compile(r"\b(?:paanch|panch|five)\s*(?:rupaye|rupaya|rupee|rupees|rs)\s+transfer\b", re.IGNORECASE), "₹5 transfer"),
    (re.compile(r"\b₹\s*5\s+transfer\b", re.IGNORECASE), "₹5 transfer"),
)

# Standalone Hindi/Bhojpuri number tokens when adjacent to time/money units
_UNIT_FOLLOWED_NUMBERS_RE = re.compile(
    r"\b(ek|do|du|dui|teen|char|chaar|paanch|panch|chhah|chhe|saat|aath|nau|dus|das|gyarah|barah|pandra|pandrah|bees|tees)\b"
    r"(?=\s+(?:minute|min|minat|ghante|ghanta|baje|rupaye|rupaya|rs|din|second))",
    re.IGNORECASE,
)

_NUMBER_WORD_TO_DIGIT: Dict[str, str] = {
    "ek": "1",
    "do": "2",
    "du": "2",
    "dui": "2",
    "teen": "3",
    "char": "4",
    "chaar": "4",
    "paanch": "5",
    "panch": "5",
    "chhah": "6",
    "chhe": "6",
    "saat": "7",
    "aath": "8",
    "nau": "9",
    "dus": "10",
    "das": "10",
    "gyarah": "11",
    "barah": "12",
    "pandra": "15",
    "pandrah": "15",
    "bees": "20",
    "tees": "30",
}

# Known attack vocabulary for space-collapsed de-obfuscation (e.g., "q u i c k s u p p o r t" -> "quicksupport")
# Includes all single-word and compound target tokens that scammers commonly space out.
_TARGET_DEOBFUSCATION_VOCAB: Set[str] = {
    "apk",
    "anydesk",
    "teamviewer",
    "quicksupport",
    "rustdesk",
    "bijli",
    "turant",
    "fatafat",
    "jaldi",
    "otp",
    "upipin",
    "upi",
    "pin",
    "cvv",
    "cbi",
    "police",
    "customs",
    "trai",
    "dcp",
    "narcotics",
    "warrant",
    "kyc",
    "pancard",
    "whatsapp",
    "screenshare",
    "download",
    "disconnect",
    "freeze",
    "simblock",
}

# Maps space-collapsed compound words to their canonical spaced pattern if needed
_COMPOUND_RESTORE_MAP: Dict[str, str] = {
    "upipin": "upi pin",
    "screenshare": "screen share",
    "simblock": "sim block",
    "pancard": "pan card",
}

# Common legitimate 1-letter or 2-letter Hindi/English words that should NOT be blindly merged
# unless the combined sequence forms a known attack vocabulary word.
_COMMON_SHORT_STOPWORDS: Set[str] = {
    "a",
    "i",
    "in",
    "on",
    "at",
    "to",
    "of",
    "is",
    "it",
    "or",
    "by",
    "do",
    "go",
    "no",
    "so",
    "up",
    "ka",
    "ki",
    "ke",
    "ko",
    "se",
    "me",
    "ne",
    "pe",
    "ho",
    "hi",
    "ab",
    "ye",
    "wo",
    "na",
    "ha",
    "ji",
    "ek",
}


class IndicTextNormalizer:
    """
    Stateless, thread-safe text normalizer and phonetic de-obfuscator for
    Hindi, Hinglish, Bhojpuri, and Indian English transcripts/SMS payloads.
    """

    def __init__(self) -> None:
        self._vocab = set(_TARGET_DEOBFUSCATION_VOCAB)

    @staticmethod
    def _collapse_punctuated_chars(match: re.Match[str]) -> str:
        """Collapses 'a.p.k' -> 'apk' and 'b-i-j-l-i' -> 'bijli'."""
        raw = match.group(0)
        collapsed = _INNER_PUNCT_RE.sub("", raw).lower()
        return collapsed

    def _collapse_spaced_characters(self, text: str) -> str:
        """
        Reconstructs tokens obfuscated with spaces between single characters,
        such as 'q u i c k s u p p o r t' -> 'quicksupport' or 'a p k' -> 'apk',
        without corrupting normal grammatical words like 'ka ki ke'.
        """
        tokens = text.split(" ")
        if len(tokens) < 2:
            return text

        result: List[str] = []
        n = len(tokens)
        i = 0

        while i < n:
            # Check if we are starting a run of single-character alphabetic tokens
            if len(tokens[i]) == 1 and tokens[i].isalpha():
                j = i
                while j < n and len(tokens[j]) == 1 and tokens[j].isalpha():
                    j += 1

                run_len = j - i
                if run_len >= 2:
                    # Try greedy matching against known attack vocabulary sub-spans first
                    matched_any_vocab = False
                    k = i
                    temp_out: List[str] = []
                    while k < j:
                        best_end = -1
                        best_word = ""
                        # Look ahead up to 15 single-char tokens
                        max_look = min(j, k + 15)
                        for end_idx in range(max_look, k + 1, -1):
                            candidate = "".join(tokens[k:end_idx])
                            if candidate in self._vocab:
                                best_end = end_idx
                                best_word = _COMPOUND_RESTORE_MAP.get(candidate, candidate)
                                break
                        if best_end != -1:
                            temp_out.append(best_word)
                            k = best_end
                            matched_any_vocab = True
                        else:
                            temp_out.append(tokens[k])
                            k += 1

                    if matched_any_vocab:
                        result.extend(temp_out)
                    elif run_len >= 3:
                        # A run of >=3 single letters (e.g. "b i j l i" or arbitrary spelled-out word)
                        combined = "".join(tokens[i:j])
                        result.append(_COMPOUND_RESTORE_MAP.get(combined, combined))
                    else:
                        # Run of 2 single letters not in vocab (e.g. "a b") -> merge if not stopwords
                        combined = "".join(tokens[i:j])
                        if combined not in _COMMON_SHORT_STOPWORDS:
                            result.append(combined)
                        else:
                            result.extend(tokens[i:j])
                    i = j
                    continue

            result.append(tokens[i])
            i += 1

        return " ".join(result)

    def normalize(self, text: str) -> str:
        """
        Executes the full Indic normalization pipeline in <0.5ms:
        1. Unicode NFKC normalization and zero-width character stripping.
        2. Devanagari key phrase transliteration to canonical Roman/Hinglish.
        3. Punctuated character de-obfuscation ("a.p.k" -> "apk", "b-i-j-l-i" -> "bijli").
        4. Punctuation stripping and whitespace normalization.
        5. Spaced-letter de-obfuscation ("q u i c k s u p p o r t" -> "quicksupport").
        6. Colloquial Indic number and time phrase normalization ("dus minute" -> "10 minute").
        """
        if not text:
            return ""

        # Step 1: Strip zero-width characters and normalize Unicode
        cleaned = _ZERO_WIDTH_RE.sub("", text)
        cleaned = unicodedata.normalize("NFKC", cleaned)

        # Step 2: Transliterate common Devanagari fraud phrases if present
        for dev_phrase, roman_phrase in _DEVANAGARI_PHRASE_MAP:
            if dev_phrase in cleaned:
                cleaned = cleaned.replace(dev_phrase, f" {roman_phrase} ")

        cleaned = cleaned.lower()

        # Step 3: Collapse punctuated single-letter obfuscations ("a.p.k" -> "apk", "b-i-j-l-i" -> "bijli")
        cleaned = _PUNCTUATED_CHARS_RE.sub(self._collapse_punctuated_chars, cleaned)

        # Step 4: Normalize currency shorthands ("rs 5 transfer" -> "₹5 transfer")
        cleaned = _RUPEE_PREFIX_RE.sub(r"₹\1", cleaned)
        cleaned = _RUPEE_SUFFIX_RE.sub(r"₹\1", cleaned)

        # Replace underscores and hyphens with spaces so compound tokens separate cleanly
        cleaned = cleaned.replace("_", " ").replace("-", " ")

        # Strip non-word punctuation while keeping '.apk' intact if present
        cleaned = _NON_WORD_PUNCT_RE.sub(" ", cleaned)
        cleaned = _STANDALONE_DOTS_RE.sub(" ", cleaned)
        cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()

        if not cleaned:
            return ""

        # Step 5: Collapse spaced character sequences ("q u i c k s u p p o r t" -> "quicksupport")
        cleaned = self._collapse_spaced_characters(cleaned)

        # Step 6: Normalize colloquial numbers and time expressions
        for pattern, replacement in _COLLOQUIAL_NUMBER_PHRASES:
            cleaned = pattern.sub(replacement, cleaned)

        cleaned = _UNIT_FOLLOWED_NUMBERS_RE.sub(
            lambda m: _NUMBER_WORD_TO_DIGIT.get(m.group(1).lower(), m.group(1)),
            cleaned,
        )

        # Final whitespace cleanup
        cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()
        return cleaned


@lru_cache(maxsize=1)
def get_normalizer() -> IndicTextNormalizer:
    """Returns a cached singleton of `IndicTextNormalizer`."""
    return IndicTextNormalizer()