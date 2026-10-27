"""
Comprehensive Test Suite for the Cognitive Threat Brain (`fraud_engine`).

Covers all 4 mandatory functional and performance specifications using
`fastapi.testclient.TestClient` and `pytest` (also runnable directly via
`python test_engine.py` or `python -m unittest`):

- Test 1: Real Hindi utility scam transcript ("Bijli cut ho jayegi 10 minute mein,
  QuickSupport apk download karo") triggers `CRITICAL` threat level and
  `lock_upi = True` in <25ms.
- Test 2: Safe daily financial talk ("Bhaiya bijli ka bill bhar diya maine PhonePe se")
  evaluates as `SAFE` without false positives.
- Test 3: Obfuscated inputs ("a.p.k", "q u i c k s u p p o r t", "b-i-j-l-i",
  "dus minute") are resolved by the normalizer and caught by the Aho-Corasick matcher.
- Test 4: End-to-end WebSocket roundtrip test validating JSON structure, schema
  validation, malformed packet recovery, keep-alive PING/PONG, session context
  persistence across reconnects, and 16 kHz PCM audio streaming pipeline.
"""

from __future__ import annotations

import base64
import os
import sys
import time
import unittest

from fastapi.testclient import TestClient

# Allow running directly from inside `fraud_engine/` or from parent workspace directory
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fraud_engine.config import ThreatCategory, get_settings
from fraud_engine.matcher import PatternMatcher
from fraud_engine.normalizer import IndicTextNormalizer
from fraud_engine.schemas import (
    CallerMetadata,
    InboundPacket,
    MitigationAction,
    OutboundVerdict,
    StreamType,
    ThreatLevel,
)
from fraud_engine.scorer import SessionThreatScorer
from fraud_engine.server import create_app
from fraud_engine.stt_streamer import CircularPCMBuffer


def test_1_hindi_utility_scam_triggers_critical_under_25ms() -> None:
    """
    Test 1: Real Hindi utility scam transcript ("Bijli cut ho jayegi 10 minute mein,
    QuickSupport apk download karo") correctly triggers `CRITICAL` threat level and
    `lock_upi = True` in <25ms.
    """
    scorer = SessionThreatScorer()
    packet = InboundPacket(
        session_id="sess-hindi-utility-scam-001",
        client_timestamp=time.time(),
        stream_type=StreamType.TEXT_TRANSCRIPT,
        payload="Bijli cut ho jayegi 10 minute mein, QuickSupport apk download karo",
        metadata=CallerMetadata(
            caller_number="+919988776655",
            duration_seconds=14.2,
            language="hi-IN",
        ),
    )

    t0 = time.perf_counter()
    verdict = scorer.evaluate_packet(packet)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    assert verdict.threat_level == ThreatLevel.CRITICAL, (
        f"Expected CRITICAL threat level, got {verdict.threat_level}"
    )
    assert verdict.mitigation.lock_upi is True, "Expected lock_upi=True for utility APK scam"
    assert verdict.mitigation.action == MitigationAction.HALT_AND_FREEZE
    assert verdict.mitigation.cooling_off_minutes >= 15
    assert verdict.composite_threat_score >= get_settings().CRITICAL_THRESHOLD
    assert verdict.vector_scores.urgency_coercion > 0.0
    assert verdict.vector_scores.remote_access_apk > 0.0
    assert "bijli cut" in verdict.detected_cue_tokens
    assert "quicksupport" in verdict.detected_cue_tokens
    assert verdict.mitigation.vernacular_voice_script != ""
    assert verdict.latency_ms < 25.0, f"Internal latency {verdict.latency_ms}ms exceeded 25ms SLA"
    assert wall_ms < 25.0, f"Wall-clock latency {wall_ms:.3f}ms exceeded 25ms SLA"


def test_2_safe_daily_financial_talk_no_false_positives() -> None:
    """
    Test 2: Safe daily financial talk ("Bhaiya bijli ka bill bhar diya maine PhonePe se")
    correctly evaluates as `SAFE` without false positives.
    """
    scorer = SessionThreatScorer()
    packet = InboundPacket(
        session_id="sess-safe-talk-002",
        client_timestamp=time.time(),
        stream_type=StreamType.TEXT_TRANSCRIPT,
        payload="Bhaiya bijli ka bill bhar diya maine PhonePe se",
        metadata=CallerMetadata(
            caller_number="+919123456789",
            duration_seconds=8.5,
            language="hi-IN",
            is_in_contacts=True,
        ),
    )

    verdict = scorer.evaluate_packet(packet)

    assert verdict.threat_level == ThreatLevel.SAFE, (
        f"Expected SAFE for benign talk, got {verdict.threat_level}"
    )
    assert verdict.composite_threat_score == 0.0
    assert verdict.mitigation.lock_upi is False
    assert verdict.mitigation.action == MitigationAction.MONITOR
    assert verdict.mitigation.cooling_off_minutes == 0
    assert verdict.detected_cue_tokens == []
    assert verdict.latency_ms < 25.0


def test_3_obfuscated_inputs_normalized_and_caught() -> None:
    """
    Test 3: Obfuscated inputs ("a.p.k", "q u i c k s u p p o r t", "b-i-j-l-i",
    "dus minute") are correctly resolved by the normalizer and caught by the matcher.
    """
    normalizer = IndicTextNormalizer()
    matcher = PatternMatcher()
    scorer = SessionThreatScorer(normalizer=normalizer, matcher=matcher)

    raw_obfuscated = (
        "Aapki b-i-j-l-i cut ho jayegi dus minute ke andar, "
        "turant q u i c k s u p p o r t a.p.k download karo!"
    )
    normalized = normalizer.normalize(raw_obfuscated)

    assert "apk" in normalized, f"'a.p.k' was not normalized in: {normalized}"
    assert "quicksupport" in normalized, f"'q u i c k s u p p o r t' was not normalized in: {normalized}"
    assert "bijli cut" in normalized, f"'b-i-j-l-i cut' was not normalized in: {normalized}"
    assert "10 minute ke andar" in normalized, f"'dus minute' was not normalized in: {normalized}"

    matches = matcher.scan(normalized)
    matched_tokens = {m.token for m in matches}
    matched_categories = {m.category for m in matches}

    assert "quicksupport" in matched_tokens
    assert ("apk" in matched_tokens or "apk download karo" in matched_tokens)
    assert "bijli cut" in matched_tokens
    assert "10 minute ke andar" in matched_tokens
    assert ThreatCategory.REMOTE_ACCESS_APK.value in matched_categories
    assert ThreatCategory.URGENCY_COERCION.value in matched_categories

    # Verify full scorer evaluation on obfuscated transcript
    verdict = scorer.evaluate_packet(
        InboundPacket(
            session_id="sess-obfuscated-003",
            stream_type=StreamType.TEXT_TRANSCRIPT,
            payload=raw_obfuscated,
            metadata=CallerMetadata(language="bho-IN"),
        )
    )
    assert verdict.threat_level == ThreatLevel.CRITICAL
    assert verdict.mitigation.lock_upi is True
    assert "ठगी" in verdict.mitigation.vernacular_voice_script


def test_4_end_to_end_websocket_roundtrip_and_audio_streaming() -> None:
    """
    Test 4: End-to-end WebSocket roundtrip test validating correct JSON structure,
    schema validation, malformed packet handling, PING/PONG keep-alives, multi-turn
    session state, and 16 kHz PCM audio streaming pipeline.
    """
    app = create_app()
    client = TestClient(app)

    # 1. Verify /health endpoint before streaming
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_json = health_resp.json()
    assert health_json["status"] == "ok"

    dashboard_resp = client.get("/")
    assert dashboard_resp.status_code == 200
    assert "Sentinel — Fraud intelligence console" in dashboard_resp.text
    assert client.get("/static/app.css").status_code == 200
    frontend_js = client.get("/static/app.js")
    assert frontend_js.status_code == 200
    assert "/ws/stream" in frontend_js.text

    with client.websocket_connect("/ws/stream") as ws:
        # 2. Verify raw PING/PONG keep-alive
        ws.send_text("PING")
        assert ws.receive_text() == "PONG"

        # 3. Verify malformed JSON error handling without dropping connection
        ws.send_text("{bad_json_frame")
        err_msg = ws.receive_json()
        assert err_msg["status"] == "error"
        assert err_msg["error_code"] == "MALFORMED_JSON"

        # 4. Verify schema validation error handling
        ws.send_json({"session_id": "  ", "stream_type": "TEXT_TRANSCRIPT"})
        schema_err = ws.receive_json()
        assert schema_err["status"] == "error"
        assert schema_err["error_code"] == "SCHEMA_VALIDATION_ERROR"

        # 5. Send safe transcript over WebSocket and validate OutboundVerdict schema
        ws.send_json(
            {
                "session_id": "ws-e2e-safe-session",
                "client_timestamp": time.time(),
                "stream_type": "TEXT_TRANSCRIPT",
                "payload": "Bhaiya bijli ka bill bhar diya maine PhonePe se",
                "metadata": {
                    "caller_number": "+919811122233",
                    "duration_seconds": 4.0,
                    "language": "hi-IN",
                },
            }
        )
        safe_data = ws.receive_json()
        safe_verdict = OutboundVerdict.model_validate(safe_data)
        assert safe_verdict.threat_level == ThreatLevel.SAFE
        assert safe_verdict.mitigation.lock_upi is False

        # 6. Send 16 kHz RAW_PCM_AUDIO stream frame and verify STT + CRITICAL verdict
        # Generate 3,200 bytes (100ms of 16kHz 16-bit mono non-silent PCM audio)
        non_silent_pcm = b"\x10\x27" * 1600
        pcm_b64 = base64.b64encode(non_silent_pcm).decode("ascii")

        ws.send_json(
            {
                "session_id": "ws-e2e-audio-session",
                "client_timestamp": time.time(),
                "stream_type": "RAW_PCM_AUDIO",
                "payload": pcm_b64,
                "metadata": {
                    "caller_number": "+918000999888",
                    "duration_seconds": 19.5,
                    "language": "hi-IN",
                },
            }
        )
        audio_verdict_data = ws.receive_json()
        audio_verdict = OutboundVerdict.model_validate(audio_verdict_data)
        assert audio_verdict.session_id == "ws-e2e-audio-session"
        assert audio_verdict.threat_level == ThreatLevel.CRITICAL
        assert audio_verdict.mitigation.lock_upi is True
        assert audio_verdict.mitigation.action == MitigationAction.HALT_AND_FREEZE
        assert audio_verdict.latency_ms < 200.0
        assert audio_verdict.mitigation.alert_audio_stream is not None

        # 7. Verify HEARTBEAT frame
        ws.send_json(
            {
                "session_id": "ws-e2e-audio-session",
                "client_timestamp": time.time(),
                "stream_type": "HEARTBEAT",
                "payload": "",
            }
        )
        hb_data = ws.receive_json()
        hb_verdict = OutboundVerdict.model_validate(hb_data)
        assert hb_verdict.normalized_transcript == "PONG"

    # 8. Verify multi-turn reconnection preserves sliding-window context
    reconnect_sid = f"ws-reconnect-session-{time.time_ns()}"
    with client.websocket_connect("/ws/stream") as ws1:
        ws1.send_json(
            {
                "session_id": reconnect_sid,
                "stream_type": "TEXT_TRANSCRIPT",
                "payload": "Aapki bijli cut ho jayegi 10 minute ke andar",
                "metadata": {"language": "en-IN"},
            }
        )
        turn1 = OutboundVerdict.model_validate(ws1.receive_json())
        assert turn1.vector_scores.urgency_coercion > 0.0
        assert turn1.mitigation.lock_upi is False

    # Reconnect on a brand new WebSocket with the same session_id and send the APK demand
    with client.websocket_connect("/ws/stream") as ws2:
        ws2.send_json(
            {
                "session_id": reconnect_sid,
                "stream_type": "TEXT_TRANSCRIPT",
                "payload": "Please install AnyDesk apk right now",
                "metadata": {"language": "en-IN"},
            }
        )
        turn2 = OutboundVerdict.model_validate(ws2.receive_json())
        # Both URGENCY_COERCION (from Turn 1) and REMOTE_ACCESS_APK (from Turn 2) combine!
        assert turn2.threat_level == ThreatLevel.CRITICAL
        assert turn2.mitigation.lock_upi is True
        assert "bijli cut" in turn2.detected_cue_tokens
        assert "anydesk" in turn2.detected_cue_tokens


def test_5_circular_pcm_buffer_zero_disk_io_and_bounded_memory() -> None:
    """
    Verifies that `CircularPCMBuffer` strictly enforces its byte ceiling and evicts
    oldest samples without leaking memory.
    """
    ring = CircularPCMBuffer(max_bytes=1000)
    chunk_a = b"A" * 600
    chunk_b = b"B" * 600

    assert ring.append(chunk_a) == 600
    assert ring.append(chunk_b) == 1000
    snap = bytes(ring.snapshot())
    assert len(snap) == 1000
    assert snap[:400] == b"A" * 400
    assert snap[400:] == b"B" * 600
    assert ring.total_ingested_bytes == 1200


if __name__ == "__main__":
    # Allow running directly via `python fraud_engine/test_engine.py` without duplicating
    # test collection under `pytest`.
    class _StandaloneRunner(unittest.TestCase):
        def test_1_hindi_utility_scam(self) -> None:
            test_1_hindi_utility_scam_triggers_critical_under_25ms()

        def test_2_safe_daily_financial_talk(self) -> None:
            test_2_safe_daily_financial_talk_no_false_positives()

        def test_3_obfuscated_inputs(self) -> None:
            test_3_obfuscated_inputs_normalized_and_caught()

        def test_4_end_to_end_websocket(self) -> None:
            test_4_end_to_end_websocket_roundtrip_and_audio_streaming()

        def test_5_circular_pcm_buffer(self) -> None:
            test_5_circular_pcm_buffer_zero_disk_io_and_bounded_memory()

    unittest.main(verbosity=2)