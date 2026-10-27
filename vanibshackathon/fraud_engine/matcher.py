"""
Ultra-Fast Keyword & Phrase Detection Engine using Aho-Corasick Automaton.

Uses `pyahocorasick` when the C-extension is installed, and automatically falls back
to a pure-Python Aho-Corasick Trie with failure links when `pyahocorasick` is absent.
Scans normalized transcripts in strictly linear time O(n + m) with word-boundary
verification and benign-context suppression.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from .config import (
    BENIGN_CONTEXT_PATTERNS,
    AppSettings,
    get_settings,
)

try:
    import ahocorasick as _pyahocorasick  # type: ignore

    _HAS_PYAHOCORASICK = True
except ImportError:
    _pyahocorasick = None
    _HAS_PYAHOCORASICK = False


@dataclass(frozen=True)
class KeywordMatch:
    """Represents a single verified pattern occurrence in normalized text."""

    category: str
    token: str
    start_char: int
    end_char: int
    weight: float


class _PurePythonAhoCorasickNode:
    """Internal Trie node for the pure-Python Aho-Corasick automaton fallback."""

    __slots__ = ("children", "fail", "output")

    def __init__(self) -> None:
        self.children: Dict[str, _PurePythonAhoCorasickNode] = {}
        self.fail: Optional[_PurePythonAhoCorasickNode] = None
        self.output: List[Tuple[str, str, float]] = []  # (category, pattern, weight)


class _PurePythonAhoCorasickAutomaton:
    """
    Production-grade pure-Python Aho-Corasick automaton with BFS failure links.
    Guarantees O(n) single-pass text scanning when `pyahocorasick` is not installed.
    """

    def __init__(self) -> None:
        self._root = _PurePythonAhoCorasickNode()
        self._finalized = False

    def add_word(self, word: str, value: Tuple[str, str, float]) -> None:
        if not word:
            return
        node = self._root
        for ch in word:
            nxt = node.children.get(ch)
            if nxt is None:
                nxt = _PurePythonAhoCorasickNode()
                node.children[ch] = nxt
            node = nxt
        if value not in node.output:
            node.output.append(value)
        self._finalized = False

    def make_automaton(self) -> None:
        queue: deque[_PurePythonAhoCorasickNode] = deque()
        self._root.fail = self._root

        for child in self._root.children.values():
            child.fail = self._root
            queue.append(child)

        while queue:
            current = queue.popleft()
            for ch, child in current.children.items():
                queue.append(child)
                fail_state = current.fail
                while fail_state is not None and fail_state is not self._root and ch not in fail_state.children:
                    fail_state = fail_state.fail
                if fail_state is not None and ch in fail_state.children:
                    child.fail = fail_state.children[ch]
                else:
                    child.fail = self._root

                if child.fail is not None and child.fail.output:
                    # Merge failure state outputs so overlapping suffix patterns are emitted
                    for item in child.fail.output:
                        if item not in child.output:
                            child.output.append(item)

        self._finalized = True

    def iter(self, text: str) -> List[Tuple[int, List[Tuple[str, str, float]]]]:
        if not self._finalized:
            self.make_automaton()

        results: List[Tuple[int, List[Tuple[str, str, float]]]] = []
        node = self._root
        root = self._root

        for idx, ch in enumerate(text):
            while node is not root and ch not in node.children:
                node = node.fail if node.fail is not None else root
            node = node.children.get(ch, root)
            if node.output:
                results.append((idx, node.output))

        return results


class PatternMatcher:
    """
    Linear-time O(n) multi-vector pattern matcher powered by Aho-Corasick.
    Enforces token boundary rules so short acronyms (e.g., 'otp', 'cbi', 'apk')
    never false-trigger inside unrelated words, and deduplicates overlapping spans
    within the same category.
    """

    def __init__(self, settings: Optional[AppSettings] = None) -> None:
        self._settings = settings or get_settings()
        self._backend_name = (
            "pyahocorasick" if _HAS_PYAHOCORASICK else "pure_python_aho_corasick"
        )
        self._automaton = self._build_automaton()

    @property
    def backend_name(self) -> str:
        """Returns the active automaton backend identifier."""
        return self._backend_name

    def _build_automaton(self) -> object:
        weights = self._settings.CUE_SEVERITY_WEIGHTS

        if _HAS_PYAHOCORASICK:
            automaton = _pyahocorasick.Automaton()
            pattern_map: Dict[str, List[Tuple[str, str, float]]] = {}
            for category, phrases in self._settings.PATTERN_VECTORS.items():
                for raw_phrase in phrases:
                    phrase = raw_phrase.strip().lower()
                    if not phrase:
                        continue
                    w = weights.get(phrase, 0.45)
                    pattern_map.setdefault(phrase, []).append((category, phrase, w))

            for phrase, entries in pattern_map.items():
                automaton.add_word(phrase, entries)
            automaton.make_automaton()
            return automaton

        automaton_py = _PurePythonAhoCorasickAutomaton()
        for category, phrases in self._settings.PATTERN_VECTORS.items():
            for raw_phrase in phrases:
                phrase = raw_phrase.strip().lower()
                if not phrase:
                    continue
                w = weights.get(phrase, 0.45)
                automaton_py.add_word(phrase, (category, phrase, w))
        automaton_py.make_automaton()
        return automaton_py

    @staticmethod
    def _is_valid_word_boundary(text: str, start: int, end: int) -> bool:
        """
        Verifies that the matched span [start, end) sits on clean token boundaries
        so short tokens like 'otp' or 'apk' do not match inside longer words.
        Allows '.apk' when preceded by whitespace or start of string.
        """
        n = len(text)
        if start > 0:
            prev_ch = text[start - 1]
            curr_first = text[start]
            if curr_first == ".":
                # e.g., '.apk' - valid if preceded by space or start
                if prev_ch.isalnum():
                    return False
            else:
                if prev_ch.isalnum():
                    return False
        if end < n:
            next_ch = text[end]
            if next_ch.isalnum():
                return False
        return True

    @staticmethod
    def _has_benign_bill_context(text: str) -> bool:
        """
        Checks if the utterance is a benign past-tense statement (e.g.,
        'Bhaiya bijli ka bill bhar diya maine PhonePe se').
        """
        for benign in BENIGN_CONTEXT_PATTERNS:
            if benign in text:
                return True
        return False

    def scan(self, normalized_text: str) -> List[KeywordMatch]:
        """
        Scans `normalized_text` in O(n) time and returns ordered, boundary-verified
        `KeywordMatch` objects.
        """
        if not normalized_text:
            return []

        raw_matches: List[KeywordMatch] = []

        for end_idx, matched_entries in self._automaton.iter(normalized_text):  # type: ignore[attr-defined]
            for category, pattern, weight in matched_entries:
                start_idx = end_idx - len(pattern) + 1
                end_pos = end_idx + 1
                if start_idx < 0:
                    continue
                if not self._is_valid_word_boundary(normalized_text, start_idx, end_pos):
                    continue
                raw_matches.append(
                    KeywordMatch(
                        category=category,
                        token=pattern,
                        start_char=start_idx,
                        end_char=end_pos,
                        weight=weight,
                    )
                )

        if not raw_matches:
            return []

        # Deduplicate overlapping spans within the same category: prefer longer (more specific) spans
        # e.g., if both "apk" and "apk download karo" match, keep "apk download karo" (or keep distinct non-overlapping).
        raw_matches.sort(key=lambda m: (m.start_char, -(m.end_char - m.start_char), -m.weight))

        filtered_matches: List[KeywordMatch] = []
        last_by_category: Dict[str, KeywordMatch] = {}

        for match in raw_matches:
            prev = last_by_category.get(match.category)
            if prev is not None and match.start_char < prev.end_char:
                # Overlapping span in the same category: skip the shorter/subsumed match
                continue
            filtered_matches.append(match)
            last_by_category[match.category] = match

        # If the text clearly expresses benign past-tense bill payment and there are no
        # remote-access or credential harvesting cues, suppress low-confidence urgency triggers.
        if self._has_benign_bill_context(normalized_text):
            has_high_risk_vector = any(
                m.category in ("REMOTE_ACCESS_APK", "CREDENTIAL_HARVESTING")
                for m in filtered_matches
            )
            if not has_high_risk_vector:
                filtered_matches = [
                    m for m in filtered_matches if m.category != "URGENCY_COERCION"
                ]

        return filtered_matches


@lru_cache(maxsize=1)
def get_matcher() -> PatternMatcher:
    """Returns a cached singleton of `PatternMatcher`."""
    return PatternMatcher()