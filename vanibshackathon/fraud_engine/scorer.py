"""
Stateful Multi-Turn Session Threat Scorer and Vernacular Mitigation Engine.

Maintains a sliding time-window (default 60s) of detected psychological manipulation
and technical fraud cues per `session_id`, computes the synergistic composite threat
score, enforces immediate CRITICAL escalation when `URGENCY_COERCION` and
`REMOTE_ACCESS_APK` co-occur, and generates vernacular voice warning scripts in
Hindi, Bhojpuri, and Indian English.
"""

from __future__ import annotations

import base64
import math
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Deque, Dict, List, Optional, Tuple

from .config import AppSettings, ThreatCategory, get_settings
from .matcher import KeywordMatch, PatternMatcher, get_matcher
from .normalizer import IndicTextNormalizer, get_normalizer
from .schemas import (
    InboundPacket,
    MatchedCue,
    MitigationAction,
    MitigationInstructions,
    OutboundVerdict,
    ThreatLevel,
    VectorScoreBreakdown,
)


@dataclass
class _TimedMatchEvent:
    """Single timestamped cue match stored inside a session's sliding window."""

    timestamp: float
    match: KeywordMatch


@dataclass
class _SessionContext:
    """Sliding-window state for a single active call/SMS session."""

    session_id: str
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    language: str = "hi-IN"
    events: Deque[_TimedMatchEvent] = field(default_factory=deque)


# Pre-synthesized compact 16kHz PCM/WAV acoustic alert header (base64) for WARN/CRITICAL states
@lru_cache(maxsize=2)
def _generate_alert_earcon_b64(is_critical: bool) -> str:
    """
    Synthesizes a tiny, valid in-memory WAV acoustic alert tone (880Hz for CRITICAL,
    587Hz for WARN) encoded as Base64 with zero disk I/O.
    """
    sample_rate = 8000
    duration_sec = 0.04  # 40ms micro-earcon header to keep payload ultra-compact
    num_samples = int(sample_rate * duration_sec)
    freq = 880.0 if is_critical else 587.33
    pcm_bytes = bytearray()
    for i in range(num_samples):
        sample = int(16000 * math.sin(2.0 * math.pi * freq * (i / sample_rate)))
        pcm_bytes.extend(struct.pack("<h", sample))

    data_size = len(pcm_bytes)
    wav_header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM format
        1,  # Mono
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    return base64.b64encode(wav_header + pcm_bytes).decode("ascii")


# Culturally resonant vernacular warning scripts
_VERNACULAR_SCRIPTS: Dict[str, Dict[ThreatLevel, str]] = {
    "hi": {
        ThreatLevel.CRITICAL: (
            "सावधान! यह एक साइबर ठगी कॉल है। बिजली कटने या डिजिटल अरेस्ट के नाम पर कोई भी "
            "रिमोट ऐप (QuickSupport, AnyDesk, APK) डाउनलोड न करें। आपकी सुरक्षा के लिए आपके "
            "UPI पेमेंट्स को अस्थायी रूप से लॉक कर दिया गया है। तुरंत कॉल काटें!"
        ),
        ThreatLevel.SUSPICIOUS: (
            "चेतावनी! यह कॉल संदिग्ध लग रही है। किसी भी अनजान व्यक्ति के साथ अपना OTP, UPI PIN "
            "साझा न करें और कोई भी स्क्रीन शेयरिंग ऐप इंस्टॉल न करें।"
        ),
        ThreatLevel.SAFE: "",
    },
    "bho": {
        ThreatLevel.CRITICAL: (
            "सावधान रहीं! ई एकदम फर्जी आ ठगी के कॉल बा। बिजली कटे भा पुलिस के डर से कवनो "
            "AnyDesk, QuickSupport या APK फाइल मत डाउनलोड करीं। रउआ के बचाव खातिर UPI पेमेंट "
            "लॉक कर दिहल गइल बा। तुरंत फोन काट दीं!"
        ),
        ThreatLevel.SUSPICIOUS: (
            "चेतावनी! ई कॉल संदिग्ध लागत बा। केहू के भी आपन OTP भा UPI PIN मत बताईं आ कवनो "
            "अनजान लिंक पर क्लिक मत करीं।"
        ),
        ThreatLevel.SAFE: "",
    },
    "en": {
        ThreatLevel.CRITICAL: (
            "DANGER! High-risk financial scam detected. Do not install AnyDesk, QuickSupport, "
            "or any APK file, and do not believe electricity disconnection or digital arrest "
            "threats. Outgoing UPI payments have been automatically frozen for your protection. "
            "Hang up immediately!"
        ),
        ThreatLevel.SUSPICIOUS: (
            "WARNING! Suspicious manipulation patterns detected on this call. Never share your "
            "OTP, UPI PIN, or screen with unknown callers."
        ),
        ThreatLevel.SAFE: "",
    },
}


class SessionThreatScorer:
    """
    Thread-safe, sliding-window psychological manipulation scorer.

    Evaluates incoming normalized transcripts in <5ms, tracks multi-turn session
    context across `SESSION_WINDOW_SECONDS` (default 60s), and computes:

        Score = min(
            1.0,
            (Urgency * 0.40)
            + (RemoteAccess * 0.45)
            + (Authority * 0.20)
            + (Credential * 0.35)
            + (Synergy_Bonus * 0.15)
        )

    Enforces the Critical Escalation Rule: if both `URGENCY_COERCION` and
    `REMOTE_ACCESS_APK` are active in the session window, immediately escalates
    `threat_level` to `CRITICAL` and enforces `lock_upi = True`.
    """

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        normalizer: Optional[IndicTextNormalizer] = None,
        matcher: Optional[PatternMatcher] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._normalizer = normalizer or get_normalizer()
        self._matcher = matcher or get_matcher()
        self._sessions: Dict[str, _SessionContext] = {}
        self._lock = threading.RLock()

    @property
    def active_session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _evict_expired_events(self, session: _SessionContext, now: float) -> None:
        cutoff = now - float(self._settings.SESSION_WINDOW_SECONDS)
        while session.events and session.events[0].timestamp < cutoff:
            session.events.popleft()

    def cleanup_stale_sessions(self, now: Optional[float] = None) -> int:
        """
        Purges sessions that have had no activity within `SESSION_WINDOW_SECONDS * 2`
        to guarantee zero memory leaks on long-running servers.
        """
        current_time = now if now is not None else time.time()
        max_idle = float(self._settings.SESSION_WINDOW_SECONDS) * 2.0
        removed = 0
        with self._lock:
            stale_ids = [
                sid
                for sid, ctx in self._sessions.items()
                if (current_time - ctx.last_updated) > max_idle
            ]
            for sid in stale_ids:
                del self._sessions[sid]
                removed += 1
        return removed

    def reset_session(self, session_id: str) -> None:
        """Clears state for a specific session ID."""
        with self._lock:
            self._sessions.pop(session_id, None)

    @staticmethod
    def _resolve_language_bucket(lang_code: str) -> str:
        code = (lang_code or "hi-IN").strip().lower()
        if code.startswith("bho") or "bhojpuri" in code:
            return "bho"
        if code.startswith("en"):
            return "en"
        return "hi"

    @classmethod
    def get_vernacular_script(cls, language: str, threat_level: ThreatLevel) -> str:
        bucket = cls._resolve_language_bucket(language)
        return _VERNACULAR_SCRIPTS.get(bucket, _VERNACULAR_SCRIPTS["hi"]).get(threat_level, "")

    def _compute_vector_scores(
        self, active_matches: List[KeywordMatch]
    ) -> Tuple[VectorScoreBreakdown, List[str], bool]:
        """
        Aggregates matched cues across the 4 primary vectors and calculates synergy.
        Returns `(VectorScoreBreakdown, deduplicated_tokens, critical_combo_triggered)`.
        """
        # Track distinct token weights per category so repeated mentions saturate smoothly to 1.0
        category_token_weights: Dict[str, Dict[str, float]] = {
            ThreatCategory.AUTHORITY_IMPERSONATION.value: {},
            ThreatCategory.URGENCY_COERCION.value: {},
            ThreatCategory.REMOTE_ACCESS_APK.value: {},
            ThreatCategory.CREDENTIAL_HARVESTING.value: {},
        }
        occurrence_counts: Dict[str, int] = {
            ThreatCategory.AUTHORITY_IMPERSONATION.value: 0,
            ThreatCategory.URGENCY_COERCION.value: 0,
            ThreatCategory.REMOTE_ACCESS_APK.value: 0,
            ThreatCategory.CREDENTIAL_HARVESTING.value: 0,
        }

        seen_tokens: List[str] = []
        seen_token_set = set()

        for m in active_matches:
            cat_dict = category_token_weights.get(m.category)
            if cat_dict is not None:
                prev_w = cat_dict.get(m.token, 0.0)
                if m.weight > prev_w:
                    cat_dict[m.token] = m.weight
                occurrence_counts[m.category] += 1
            if m.token not in seen_token_set:
                seen_token_set.add(m.token)
                seen_tokens.append(m.token)

        def _saturate(cat: str) -> float:
            tokens_map = category_token_weights[cat]
            if not tokens_map:
                return 0.0
            base_sum = sum(tokens_map.values())
            extra_repeats = max(0, occurrence_counts[cat] - len(tokens_map))
            # Boost slightly for multiple distinct cues or repeated pressure
            boost = 0.15 * (len(tokens_map) - 1) + 0.08 * extra_repeats
            return min(1.0, round(base_sum + boost, 4))

        authority_score = _saturate(ThreatCategory.AUTHORITY_IMPERSONATION.value)
        urgency_score = _saturate(ThreatCategory.URGENCY_COERCION.value)
        remote_score = _saturate(ThreatCategory.REMOTE_ACCESS_APK.value)
        credential_score = _saturate(ThreatCategory.CREDENTIAL_HARVESTING.value)

        active_vector_count = sum(
            1
            for s in (authority_score, urgency_score, remote_score, credential_score)
            if s > 0.0
        )

        critical_combo = urgency_score > 0.0 and remote_score > 0.0

        if critical_combo:
            synergy_score = 1.0
        elif active_vector_count >= 3:
            synergy_score = 1.0
        elif active_vector_count == 2:
            synergy_score = 0.75
        else:
            synergy_score = 0.0

        breakdown = VectorScoreBreakdown(
            authority_impersonation=authority_score,
            urgency_coercion=urgency_score,
            remote_access_apk=remote_score,
            credential_harvesting=credential_score,
            synergy_bonus=synergy_score,
        )
        return breakdown, seen_tokens, critical_combo

    def evaluate_packet(self, packet: InboundPacket) -> OutboundVerdict:
        """
        Normalizes, matches, and scores an `InboundPacket` within the session's
        sliding time window, returning a complete `OutboundVerdict`.
        """
        start_ns = time.perf_counter_ns()
        now = time.time()

        normalized_text = self._normalizer.normalize(packet.payload)
        frame_matches = self._matcher.scan(normalized_text) if normalized_text else []

        with self._lock:
            session = self._sessions.get(packet.session_id)
            if session is None:
                if len(self._sessions) >= 2048:
                    self.cleanup_stale_sessions(now)
                session = _SessionContext(
                    session_id=packet.session_id,
                    created_at=now,
                    last_updated=now,
                    language=packet.metadata.language,
                )
                self._sessions[packet.session_id] = session

            session.last_updated = now
            if packet.metadata.language:
                session.language = packet.metadata.language

            self._evict_expired_events(session, now)

            for m in frame_matches:
                session.events.append(_TimedMatchEvent(timestamp=now, match=m))

            window_matches = [ev.match for ev in session.events]
            session_lang = session.language

        breakdown, detected_tokens, critical_combo = self._compute_vector_scores(window_matches)

        # Synergistic Threat Formula:
        # Score = min(1.0, (Urgency * 0.40) + (RemoteAccess * 0.45) + (Authority * 0.20) + (Credential * 0.35) + (Synergy_Bonus * 0.15))
        raw_composite = (
            (breakdown.urgency_coercion * self._settings.WEIGHT_URGENCY)
            + (breakdown.remote_access_apk * self._settings.WEIGHT_REMOTE_ACCESS)
            + (breakdown.authority_impersonation * self._settings.WEIGHT_AUTHORITY)
            + (breakdown.credential_harvesting * self._settings.WEIGHT_CREDENTIAL)
            + (breakdown.synergy_bonus * self._settings.WEIGHT_SYNERGY_BONUS)
        )

        # Critical Rule: If both URGENCY_COERCION and REMOTE_ACCESS_APK triggers are detected
        # simultaneously, immediately escalate threat_level to CRITICAL and enforce lock_upi = True.
        if critical_combo:
            raw_composite = max(raw_composite, self._settings.CRITICAL_THRESHOLD, 0.92)

        composite_score = round(min(1.0, max(0.0, raw_composite)), 4)

        if critical_combo or composite_score >= self._settings.CRITICAL_THRESHOLD:
            threat_level = ThreatLevel.CRITICAL
            action = MitigationAction.HALT_AND_FREEZE
            lock_upi = True
            cooling_off = self._settings.CRITICAL_COOLING_OFF_MINUTES
            alert_audio = _generate_alert_earcon_b64(is_critical=True)
        elif composite_score >= self._settings.SUSPICIOUS_THRESHOLD:
            threat_level = ThreatLevel.SUSPICIOUS
            action = MitigationAction.WARN
            lock_upi = False
            cooling_off = self._settings.SUSPICIOUS_COOLING_OFF_MINUTES
            alert_audio = _generate_alert_earcon_b64(is_critical=False)
        else:
            threat_level = ThreatLevel.SAFE
            action = MitigationAction.MONITOR
            lock_upi = False
            cooling_off = 0
            alert_audio = None

        vernacular_script = self.get_vernacular_script(session_lang, threat_level)

        matched_cue_models = [
            MatchedCue(
                category=m.category,
                token=m.token,
                start_char=m.start_char,
                end_char=m.end_char,
                weight=m.weight,
            )
            for m in frame_matches
        ]

        elapsed_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000.0, 3)

        return OutboundVerdict(
            session_id=packet.session_id,
            server_timestamp=now,
            latency_ms=elapsed_ms,
            threat_level=threat_level,
            composite_threat_score=composite_score,
            vector_scores=breakdown,
            detected_cue_tokens=detected_tokens,
            matched_cues=matched_cue_models,
            normalized_transcript=normalized_text,
            mitigation=MitigationInstructions(
                action=action,
                lock_upi=lock_upi,
                cooling_off_minutes=cooling_off,
                alert_audio_stream=alert_audio,
                vernacular_voice_script=vernacular_script,
            ),
        )


@lru_cache(maxsize=1)
def get_scorer() -> SessionThreatScorer:
    """Returns a cached singleton of `SessionThreatScorer`."""
    return SessionThreatScorer()