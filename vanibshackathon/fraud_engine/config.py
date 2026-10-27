"""
Configuration and Multi-Dialect Threat Pattern Dictionaries for the Fraud Engine.

Supports both Pydantic v2 (`pydantic-settings`) and Pydantic v1 (`pydantic.BaseSettings`)
seamlessly so it runs across any production or test Python environment.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Dict, List, Tuple

from pydantic import Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore

    _PYDANTIC_V2_SETTINGS = True
except ImportError:
    from pydantic import BaseSettings  # type: ignore

    _PYDANTIC_V2_SETTINGS = False


class ThreatCategory(str, Enum):
    """Core psychological manipulation and technical attack vectors."""

    AUTHORITY_IMPERSONATION = "AUTHORITY_IMPERSONATION"
    URGENCY_COERCION = "URGENCY_COERCION"
    REMOTE_ACCESS_APK = "REMOTE_ACCESS_APK"
    CREDENTIAL_HARVESTING = "CREDENTIAL_HARVESTING"


DEFAULT_PATTERN_VECTORS: Dict[str, List[str]] = {
    ThreatCategory.AUTHORITY_IMPERSONATION.value: [
        "cbi",
        "police",
        "customs",
        "trai",
        "court warrant",
        "dcp",
        "digital arrest",
        "narcotics",
        "cyber crime branch",
        "ed notice",
        "enforcement directorate",
        "supreme court order",
        "arrest warrant",
        "f.i.r",
        "fir darj",
        "money laundering",
        "parcel seized",
        "drugs parcel",
        "mumbai crime branch",
        "rbi officer",
        "income tax raid",
        "thanedar",
        "jail bhej",
        "kanooni karwai",
    ],
    ThreatCategory.URGENCY_COERCION.value: [
        "bijli cut",
        "electricity bill unpaid",
        "power disconnect",
        "turant",
        "fatafat",
        "jaldi karo",
        "10 minute ke andar",
        "10 minute mein",
        "aaj raat 9 baje",
        "account freeze",
        "sim block",
        "sim card block",
        "connection kat jayega",
        "batti kat",
        "line kat jayegi",
        "abhi ke abhi",
        "2 ghante ke andar",
        "2 ghante mein",
        "bank khata band",
        "khatam ho jayega",
        "last warning",
        "immediately update",
        "turanth",
        "turant bhejo",
        "fine lagega",
        "penalty lagegi",
    ],
    ThreatCategory.REMOTE_ACCESS_APK.value: [
        "anydesk",
        "teamviewer",
        "quicksupport",
        "rustdesk",
        ".apk",
        "apk",
        "whatsapp par link",
        "screen share",
        "start now",
        "app download karo",
        "forward karo",
        "apk download karo",
        "customer support app",
        "remote support",
        "9 digit code",
        "access code batao",
        "allow par click",
        "install karo",
        "play store se download",
        "link par click karo",
        "sms forward",
    ],
    ThreatCategory.CREDENTIAL_HARVESTING.value: [
        "otp",
        "upi pin",
        "cvv",
        "₹5 transfer",
        "5 rs transfer",
        "rs 5 transfer",
        "verify pan card",
        "kyc update",
        "kyc expire",
        "kyc suspend",
        "atm pin",
        "mpin",
        "card number batao",
        "expiry date batao",
        "otp share",
        "otp batao",
        "pin darj karo",
        "1 rupaya bhejo",
        "pan card block",
        "aadhaar verify",
    ],
}

# Specific keyword weights within each category (default is 0.45 per matched distinct cue)
DEFAULT_CUE_SEVERITY_WEIGHTS: Dict[str, float] = {
    # Authority Impersonation
    "digital arrest": 0.75,
    "court warrant": 0.65,
    "arrest warrant": 0.65,
    "narcotics": 0.65,
    "cbi": 0.55,
    "trai": 0.50,
    "dcp": 0.50,
    "customs": 0.50,
    "police": 0.45,
    # Urgency & Coercion
    "bijli cut": 0.65,
    "electricity bill unpaid": 0.65,
    "power disconnect": 0.65,
    "account freeze": 0.65,
    "sim block": 0.65,
    "10 minute ke andar": 0.55,
    "10 minute mein": 0.55,
    "aaj raat 9 baje": 0.55,
    "turant": 0.45,
    "fatafat": 0.45,
    "jaldi karo": 0.45,
    # Remote Access & APK
    "quicksupport": 0.80,
    "anydesk": 0.80,
    "teamviewer": 0.80,
    "rustdesk": 0.80,
    ".apk": 0.75,
    "apk": 0.70,
    "apk download karo": 0.85,
    "app download karo": 0.65,
    "screen share": 0.65,
    "whatsapp par link": 0.60,
    "start now": 0.45,
    "forward karo": 0.50,
    # Credential Harvesting
    "upi pin": 0.80,
    "otp": 0.65,
    "cvv": 0.80,
    "₹5 transfer": 0.65,
    "verify pan card": 0.55,
    "kyc update": 0.55,
}

# Benign conversational patterns (e.g. past-tense bill payment confirmation)
# that should not trigger false alerts when no remote-access or credential harvesting is present.
BENIGN_CONTEXT_PATTERNS: Tuple[str, ...] = (
    "bill bhar diya",
    "bhar diya maine",
    "jama kar diya",
    "paid the bill",
    "payment kar diya",
    "recharge kar diya",
    "bill pay kar diya",
)


class AppSettings(BaseSettings):
    """
    Production configuration for the Cognitive Threat Brain.
    Tunable via environment variables prefixed with `FRAUD_ENGINE_`.
    """

    APP_NAME: str = "Cognitive Threat Brain - Financial Fraud Immunity Engine"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "production"
    LOG_LEVEL: str = "INFO"

    # Sensitivity & Scoring Thresholds
    CRITICAL_THRESHOLD: float = Field(default=0.85, ge=0.0, le=1.0)
    SUSPICIOUS_THRESHOLD: float = Field(default=0.50, ge=0.0, le=1.0)
    SESSION_WINDOW_SECONDS: int = Field(default=60, ge=5, le=3600)
    MAX_AUDIO_CHUNK_BYTES: int = Field(default=64000, ge=512, le=1048576)

    # Audio Stream Parameters (16 kHz, 16-bit Mono PCM = 32,000 bytes/sec)
    AUDIO_SAMPLE_RATE_HZ: int = 16000
    AUDIO_SAMPLE_WIDTH_BYTES: int = 2
    AUDIO_CHANNELS: int = 1
    AUDIO_RING_BUFFER_SECONDS: int = 15

    # Synergistic Formula Weights
    WEIGHT_URGENCY: float = 0.40
    WEIGHT_REMOTE_ACCESS: float = 0.45
    WEIGHT_AUTHORITY: float = 0.20
    WEIGHT_CREDENTIAL: float = 0.35
    WEIGHT_SYNERGY_BONUS: float = 0.15

    # Cooling-off durations (minutes)
    CRITICAL_COOLING_OFF_MINUTES: int = 30
    SUSPICIOUS_COOLING_OFF_MINUTES: int = 5

    # Multi-dialect pattern dictionaries
    PATTERN_VECTORS: Dict[str, List[str]] = Field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_PATTERN_VECTORS.items()}
    )
    CUE_SEVERITY_WEIGHTS: Dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_CUE_SEVERITY_WEIGHTS)
    )

    if _PYDANTIC_V2_SETTINGS:
        model_config = SettingsConfigDict(
            env_prefix="FRAUD_ENGINE_",
            case_sensitive=True,
            extra="ignore",
        )
    else:

        class Config:
            env_prefix = "FRAUD_ENGINE_"
            case_sensitive = True
            extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Returns a cached singleton of the application settings."""
    return AppSettings()