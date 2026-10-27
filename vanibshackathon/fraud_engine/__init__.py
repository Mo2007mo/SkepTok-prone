"""
Cognitive Threat Brain — Real-Time Digital Financial Fraud Immunity Engine.

Processes live 16 kHz raw PCM call audio streams and SMS/transcript streams over
WebSockets, executes low-latency psychological manipulation scoring across
code-mixed Indic dialects (Hindi, Hinglish, Bhojpuri, Indian English), and emits
structured mitigation payloads in <30ms per text frame (<200ms end-to-end).
"""

from .config import AppSettings, ThreatCategory, get_settings
from .matcher import KeywordMatch, PatternMatcher, get_matcher
from .normalizer import IndicTextNormalizer, get_normalizer
from .schemas import (
    CallerMetadata,
    HealthResponse,
    InboundPacket,
    MatchedCue,
    MitigationAction,
    MitigationInstructions,
    OutboundVerdict,
    StreamType,
    ThreatLevel,
    VectorScoreBreakdown,
)
from .scorer import SessionThreatScorer, get_scorer
from .stt_streamer import (
    BaseSpeechTranscriber,
    CircularPCMBuffer,
    MockStreamingTranscriber,
    SherpaOnnxTranscriber,
    get_transcriber,
)

__version__ = "1.0.0"

__all__ = [
    "AppSettings",
    "ThreatCategory",
    "get_settings",
    "StreamType",
    "ThreatLevel",
    "MitigationAction",
    "CallerMetadata",
    "InboundPacket",
    "MatchedCue",
    "VectorScoreBreakdown",
    "MitigationInstructions",
    "OutboundVerdict",
    "HealthResponse",
    "IndicTextNormalizer",
    "get_normalizer",
    "KeywordMatch",
    "PatternMatcher",
    "get_matcher",
    "SessionThreatScorer",
    "get_scorer",
    "CircularPCMBuffer",
    "BaseSpeechTranscriber",
    "MockStreamingTranscriber",
    "SherpaOnnxTranscriber",
    "get_transcriber",
]