"""
Strict Pydantic Schemas for Inbound Telemetry and Outbound Mitigation Verdicts.

Provides full Pydantic v2 API compatibility (`model_validate`, `model_dump`,
`model_dump_json`, `@field_validator`, `ConfigDict`) while maintaining seamless
compatibility if executed on a Pydantic v1 runtime.
"""

from __future__ import annotations

import base64
import time
from enum import Enum
from typing import Any, Dict, List, Optional

import pydantic
from pydantic import BaseModel, Field

_IS_PYDANTIC_V2 = int(pydantic.__version__.split(".")[0]) >= 2

if _IS_PYDANTIC_V2:
    from pydantic import ConfigDict, field_validator, model_validator
else:
    from pydantic import root_validator, validator


class CompatibilityModel(BaseModel):
    """
    Base schema model providing strict validation and unified Pydantic v2 methods
    (`model_validate`, `model_dump`, `model_dump_json`) across both Pydantic v1 and v2.
    """

    if _IS_PYDANTIC_V2:
        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            populate_by_name=True,
            str_strip_whitespace=False,
        )
    else:

        class Config:
            extra = "forbid"
            validate_assignment = True
            allow_population_by_field_name = True

        @classmethod
        def model_validate(cls, obj: Any) -> "CompatibilityModel":
            return cls.parse_obj(obj)

        @classmethod
        def model_validate_json(cls, json_data: str | bytes) -> "CompatibilityModel":
            return cls.parse_raw(json_data)

        def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
            return self.dict(**kwargs)

        def model_dump_json(self, **kwargs: Any) -> str:
            return self.json(**kwargs)


class StreamType(str, Enum):
    """Supported inbound WebSocket frame payload types."""

    TEXT_TRANSCRIPT = "TEXT_TRANSCRIPT"
    RAW_PCM_AUDIO = "RAW_PCM_AUDIO"
    HEARTBEAT = "HEARTBEAT"


class ThreatLevel(str, Enum):
    """Categorized risk posture for the active session."""

    SAFE = "SAFE"
    SUSPICIOUS = "SUSPICIOUS"
    CRITICAL = "CRITICAL"


class MitigationAction(str, Enum):
    """Client-side defensive directive to execute on the Android device."""

    MONITOR = "MONITOR"
    WARN = "WARN"
    HALT_AND_FREEZE = "HALT_AND_FREEZE"


class CallerMetadata(CompatibilityModel):
    """Contextual metadata regarding the incoming call or SMS message."""

    caller_number: str = Field(
        default="UNKNOWN",
        min_length=1,
        max_length=32,
        description="Originating phone number or sender ID (e.g., '+919876543210' or 'VM-ELECBR').",
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Elapsed call duration in seconds.",
    )
    language: str = Field(
        default="hi-IN",
        min_length=2,
        max_length=16,
        description="BCP-47 or dialect code ('hi-IN', 'bho-IN', 'en-IN', 'hinglish').",
    )
    is_in_contacts: bool = Field(
        default=False,
        description="Whether the caller is saved in the user's local address book.",
    )


class InboundPacket(CompatibilityModel):
    """
    Inbound WebSocket frame sent from the Android client to `/ws/stream`.
    """

    session_id: str = Field(
        ...,
        min_length=3,
        max_length=128,
        description="Unique correlation identifier for the call or SMS evaluation session.",
    )
    client_timestamp: float = Field(
        default_factory=lambda: time.time(),
        gt=0.0,
        description="Client-side Unix epoch timestamp in seconds (or milliseconds).",
    )
    stream_type: StreamType = Field(
        ...,
        description="Type of stream payload: TEXT_TRANSCRIPT, RAW_PCM_AUDIO, or HEARTBEAT.",
    )
    payload: str = Field(
        default="",
        description="Raw UTF-8 transcript text or Base64-encoded 16kHz 16-bit Mono PCM audio chunk.",
    )
    metadata: CallerMetadata = Field(
        default_factory=CallerMetadata,
        description="Call/SMS metadata including caller number, call duration, and dialect.",
    )

    if _IS_PYDANTIC_V2:

        @field_validator("session_id")
        @classmethod
        def _validate_session_id(cls, value: str) -> str:
            cleaned = value.strip()
            if not cleaned:
                raise ValueError("session_id cannot be blank or whitespace.")
            return cleaned

        @model_validator(mode="after")
        def _validate_payload_by_stream_type(self) -> "InboundPacket":
            if self.stream_type == StreamType.RAW_PCM_AUDIO:
                if not self.payload:
                    raise ValueError("RAW_PCM_AUDIO stream_type requires a non-empty base64 payload.")
                try:
                    decoded = base64.b64decode(self.payload, validate=True)
                except Exception as exc:
                    raise ValueError(f"Invalid base64 encoding in RAW_PCM_AUDIO payload: {exc}") from exc
                if len(decoded) == 0:
                    raise ValueError("Decoded RAW_PCM_AUDIO payload cannot be zero bytes.")
            return self

    else:

        @validator("session_id")
        def _validate_session_id(cls, value: str) -> str:
            cleaned = value.strip()
            if not cleaned:
                raise ValueError("session_id cannot be blank or whitespace.")
            return cleaned

        @root_validator
        def _validate_payload_by_stream_type(cls, values: Dict[str, Any]) -> Dict[str, Any]:
            stream_type = values.get("stream_type")
            payload = values.get("payload", "")
            if stream_type == StreamType.RAW_PCM_AUDIO:
                if not payload:
                    raise ValueError("RAW_PCM_AUDIO stream_type requires a non-empty base64 payload.")
                try:
                    decoded = base64.b64decode(payload, validate=True)
                except Exception as exc:
                    raise ValueError(f"Invalid base64 encoding in RAW_PCM_AUDIO payload: {exc}") from exc
                if len(decoded) == 0:
                    raise ValueError("Decoded RAW_PCM_AUDIO payload cannot be zero bytes.")
            return values


class MatchedCue(CompatibilityModel):
    """Detailed representation of a single matched psychological or technical threat pattern."""

    category: str = Field(..., description="Threat vector category name.")
    token: str = Field(..., description="Canonical pattern token matched in normalized text.")
    start_char: int = Field(..., ge=0, description="Start character offset in normalized text.")
    end_char: int = Field(..., ge=0, description="End character offset in normalized text.")
    weight: float = Field(..., ge=0.0, le=1.0, description="Assigned severity weight for this cue.")


class VectorScoreBreakdown(CompatibilityModel):
    """Normalized per-vector score breakdown in [0.0, 1.0]."""

    authority_impersonation: float = Field(default=0.0, ge=0.0, le=1.0)
    urgency_coercion: float = Field(default=0.0, ge=0.0, le=1.0)
    remote_access_apk: float = Field(default=0.0, ge=0.0, le=1.0)
    credential_harvesting: float = Field(default=0.0, ge=0.0, le=1.0)
    synergy_bonus: float = Field(default=0.0, ge=0.0, le=1.0)


class MitigationInstructions(CompatibilityModel):
    """
    Actionable defensive payload instructing the Android client on UPI freezing
    and vernacular acoustic warning playback.
    """

    action: MitigationAction = Field(
        ...,
        description="Immediate mitigation posture (MONITOR, WARN, HALT_AND_FREEZE).",
    )
    lock_upi: bool = Field(
        ...,
        description="If True, Android Accessibility/Overlay service immediately blocks UPI apps & PIN entry.",
    )
    cooling_off_minutes: int = Field(
        default=0,
        ge=0,
        le=1440,
        description="Duration in minutes for which outgoing P2P/merchant UPI transfers remain frozen.",
    )
    alert_audio_stream: Optional[str] = Field(
        default=None,
        description="Base64-encoded WAV/PCM emergency siren header or acoustic earcon when warning/critical.",
    )
    vernacular_voice_script: str = Field(
        default="",
        description="Culturally contextualized vernacular TTS script played loudly over the speaker.",
    )


class OutboundVerdict(CompatibilityModel):
    """
    Real-time threat assessment verdict emitted back to the Android client over WebSocket.
    """

    session_id: str = Field(..., description="Session identifier matching the InboundPacket.")
    server_timestamp: float = Field(
        default_factory=lambda: time.time(),
        description="Server-side Unix epoch timestamp when the verdict was finalized.",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Total server processing latency for this frame in milliseconds.",
    )
    threat_level: ThreatLevel = Field(
        ...,
        description="Evaluated threat severity: SAFE, SUSPICIOUS, or CRITICAL.",
    )
    composite_threat_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregated threat score across sliding session window in [0.0, 1.0].",
    )
    vector_scores: VectorScoreBreakdown = Field(
        default_factory=VectorScoreBreakdown,
        description="Per-vector threat score breakdown.",
    )
    detected_cue_tokens: List[str] = Field(
        default_factory=list,
        description="Deduplicated list of canonical fraud cue tokens triggered in the window.",
    )
    matched_cues: List[MatchedCue] = Field(
        default_factory=list,
        description="Structured match occurrences with character spans and vector categories.",
    )
    normalized_transcript: str = Field(
        default="",
        description="De-obfuscated and normalized transcript evaluated for this frame.",
    )
    mitigation: MitigationInstructions = Field(
        ...,
        description="Nested mitigation directives for the Android client.",
    )


class HealthResponse(CompatibilityModel):
    """Health check response schema for `GET /health`."""

    status: str = Field(default="ok")
    engine_version: str = Field(default="1.0.0")
    automaton_backend: str = Field(..., description="'pyahocorasick' or 'pure_python_aho_corasick'")
    stt_backend: str = Field(..., description="Active Speech-to-Text engine backend.")
    active_ws_connections: int = Field(default=0, ge=0)
    tracked_sessions: int = Field(default=0, ge=0)
    total_frames_processed: int = Field(default=0, ge=0)
    avg_latency_ms: float = Field(default=0.0, ge=0.0)
    p95_latency_ms: float = Field(default=0.0, ge=0.0)