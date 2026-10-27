"""
High-Throughput FastAPI Application and WebSocket Streaming Server.

Exposes:
- `GET /`: Interactive WebSocket Threat Simulator & Live Telemetry Dashboard.
- `GET /health`: Real-time engine health, loaded automaton/STT backends, and
  rolling latency telemetry.
- `WebSocket /ws/stream`: Bidirectional low-latency telemetry & mitigation
  endpoint supporting `TEXT_TRANSCRIPT`, `RAW_PCM_AUDIO`, `HEARTBEAT`, and
  raw `PING`/`PONG` keep-alive frames.

Can be launched directly from either the repository root (`uvicorn fraud_engine.server:app --reload`)
or inside `fraud_engine/` (`uvicorn server:app --reload`).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

# Support both package import (`from fraud_engine.server import app`) and direct
# execution from inside the `fraud_engine/` directory (`uvicorn server:app --reload`).
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fraud_engine.config import AppSettings, get_settings
    from fraud_engine.matcher import PatternMatcher, get_matcher
    from fraud_engine.normalizer import IndicTextNormalizer, get_normalizer
    from fraud_engine.schemas import (
        HealthResponse,
        InboundPacket,
        MitigationAction,
        MitigationInstructions,
        OutboundVerdict,
        StreamType,
        ThreatLevel,
        VectorScoreBreakdown,
    )
    from fraud_engine.scorer import SessionThreatScorer
    from fraud_engine.stt_streamer import BaseSpeechTranscriber, MockStreamingTranscriber
else:
    from .config import AppSettings, get_settings
    from .matcher import PatternMatcher, get_matcher
    from .normalizer import IndicTextNormalizer, get_normalizer
    from .schemas import (
        HealthResponse,
        InboundPacket,
        MitigationAction,
        MitigationInstructions,
        OutboundVerdict,
        StreamType,
        ThreatLevel,
        VectorScoreBreakdown,
    )
    from .scorer import SessionThreatScorer
    from .stt_streamer import BaseSpeechTranscriber, MockStreamingTranscriber

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("fraud_engine.server")


class LatencyTelemetry:
    """Bounded rolling-window latency tracker (zero memory leak)."""

    def __init__(self, max_samples: int = 1000) -> None:
        self._samples: Deque[float] = deque(maxlen=max_samples)
        self._total_frames: int = 0
        self._lock = threading.Lock()

    def record(self, latency_ms: float) -> None:
        with self._lock:
            self._samples.append(latency_ms)
            self._total_frames += 1

    def snapshot_metrics(self) -> tuple[int, float, float]:
        with self._lock:
            if not self._samples:
                return self._total_frames, 0.0, 0.0
            ordered = sorted(self._samples)
            avg = round(sum(ordered) / len(ordered), 3)
            p95_idx = min(len(ordered) - 1, int(len(ordered) * 0.95))
            p95 = round(ordered[p95_idx], 3)
            return self._total_frames, avg, p95


class ConnectionManager:
    """
    Manages concurrent WebSocket connections, maps multiple sockets per `session_id`
    to support seamless client reconnection without loss of sliding-window context,
    and ensures clean resource teardown on disconnect.
    """

    def __init__(self) -> None:
        self._active_sockets: Set[WebSocket] = set()
        self._session_to_sockets: Dict[str, Set[WebSocket]] = {}
        self._socket_to_sessions: Dict[WebSocket, Set[str]] = {}
        self._lock = threading.RLock()

    @property
    def active_connection_count(self) -> int:
        with self._lock:
            return len(self._active_sockets)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        with self._lock:
            self._active_sockets.add(websocket)
            self._socket_to_sessions[websocket] = set()
        logger.info(
            "WebSocket connected | active_connections=%d",
            self.active_connection_count,
        )

    def bind_session(self, websocket: WebSocket, session_id: str) -> None:
        with self._lock:
            if websocket in self._socket_to_sessions:
                self._socket_to_sessions[websocket].add(session_id)
            sock_set = self._session_to_sockets.setdefault(session_id, set())
            sock_set.add(websocket)

    def disconnect(
        self,
        websocket: WebSocket,
        transcriber: Optional[BaseSpeechTranscriber] = None,
    ) -> None:
        with self._lock:
            self._active_sockets.discard(websocket)
            bound_sessions = self._socket_to_sessions.pop(websocket, set())
            for sid in bound_sessions:
                sock_set = self._session_to_sockets.get(sid)
                if sock_set is not None:
                    sock_set.discard(websocket)
                    if not sock_set:
                        del self._session_to_sockets[sid]
                        if transcriber is not None:
                            transcriber.release_session(sid)

        logger.info(
            "WebSocket disconnected | active_connections=%d",
            self.active_connection_count,
        )

    async def send_json(self, websocket: WebSocket, payload: Dict[str, Any]) -> None:
        await websocket.send_text(json.dumps(payload, ensure_ascii=False))


def create_app(
    settings: Optional[AppSettings] = None,
    normalizer: Optional[IndicTextNormalizer] = None,
    matcher: Optional[PatternMatcher] = None,
    scorer: Optional[SessionThreatScorer] = None,
    transcriber: Optional[BaseSpeechTranscriber] = None,
) -> FastAPI:
    """
    Factory that builds and wires the production FastAPI Fraud Immunity Engine.
    """
    cfg = settings or get_settings()
    norm = normalizer or get_normalizer()
    mtch = matcher or get_matcher()
    scr = scorer or SessionThreatScorer(settings=cfg, normalizer=norm, matcher=mtch)
    stt = transcriber or MockStreamingTranscriber(settings=cfg)
    manager = ConnectionManager()
    telemetry = LatencyTelemetry(max_samples=2000)

    application = FastAPI(
        title=cfg.APP_NAME,
        version=cfg.APP_VERSION,
        description=(
            "Real-time Cognitive Threat Brain processing 16 kHz PCM audio and "
            "code-mixed Indic transcripts over WebSockets in <30ms."
        ),
    )

    application.state.settings = cfg
    application.state.normalizer = norm
    application.state.matcher = mtch
    application.state.scorer = scr
    application.state.transcriber = stt
    application.state.connection_manager = manager
    application.state.telemetry = telemetry
    application.mount(
        "/static",
        StaticFiles(directory=Path(__file__).parent / "static"),
        name="static",
    )

    @application.get("/", response_class=FileResponse, include_in_schema=False)
    async def root_dashboard() -> FileResponse:
        """Serves the interactive browser WebSocket simulator at http://127.0.0.1:8000/."""
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @application.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        total_frames, avg_ms, p95_ms = telemetry.snapshot_metrics()
        return HealthResponse(
            status="ok",
            engine_version=cfg.APP_VERSION,
            automaton_backend=mtch.backend_name,
            stt_backend=stt.backend_name,
            active_ws_connections=manager.active_connection_count,
            tracked_sessions=scr.active_session_count,
            total_frames_processed=total_frames,
            avg_latency_ms=avg_ms,
            p95_latency_ms=p95_ms,
        )

    @application.websocket("/ws/stream")
    async def websocket_stream_endpoint(websocket: WebSocket) -> None:
        await manager.connect(websocket)
        try:
            while True:
                raw_message = await websocket.receive_text()
                frame_start_ns = time.perf_counter_ns()

                stripped = raw_message.strip()
                if stripped.upper() == "PING":
                    await websocket.send_text("PONG")
                    continue

                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    await manager.send_json(
                        websocket,
                        {
                            "status": "error",
                            "error_code": "MALFORMED_JSON",
                            "detail": f"Invalid JSON payload: {exc.msg}",
                            "server_timestamp": time.time(),
                        },
                    )
                    continue

                if not isinstance(data, dict):
                    await manager.send_json(
                        websocket,
                        {
                            "status": "error",
                            "error_code": "INVALID_PACKET_FORMAT",
                            "detail": "Inbound packet must be a JSON object.",
                            "server_timestamp": time.time(),
                        },
                    )
                    continue

                if str(data.get("type", "")).upper() == "PING":
                    await manager.send_json(
                        websocket,
                        {
                            "type": "PONG",
                            "session_id": data.get("session_id", "keepalive"),
                            "server_timestamp": time.time(),
                        },
                    )
                    continue

                try:
                    packet = InboundPacket.model_validate(data)
                except ValidationError as exc:
                    await manager.send_json(
                        websocket,
                        {
                            "status": "error",
                            "error_code": "SCHEMA_VALIDATION_ERROR",
                            "detail": exc.errors(),
                            "server_timestamp": time.time(),
                        },
                    )
                    continue

                manager.bind_session(websocket, packet.session_id)

                if packet.stream_type == StreamType.HEARTBEAT:
                    elapsed_ms = round(
                        (time.perf_counter_ns() - frame_start_ns) / 1_000_000.0, 3
                    )
                    heartbeat_verdict = OutboundVerdict(
                        session_id=packet.session_id,
                        server_timestamp=time.time(),
                        latency_ms=elapsed_ms,
                        threat_level=ThreatLevel.SAFE,
                        composite_threat_score=0.0,
                        vector_scores=VectorScoreBreakdown(),
                        detected_cue_tokens=[],
                        matched_cues=[],
                        normalized_transcript="PONG",
                        mitigation=MitigationInstructions(
                            action=MitigationAction.MONITOR,
                            lock_upi=False,
                            cooling_off_minutes=0,
                            alert_audio_stream=None,
                            vernacular_voice_script="",
                        ),
                    )
                    telemetry.record(elapsed_ms)
                    await manager.send_json(websocket, heartbeat_verdict.model_dump())
                    continue

                effective_packet = packet
                if packet.stream_type == StreamType.RAW_PCM_AUDIO:
                    pcm_bytes = base64.b64decode(packet.payload)
                    if len(pcm_bytes) > cfg.MAX_AUDIO_CHUNK_BYTES:
                        await manager.send_json(
                            websocket,
                            {
                                "status": "error",
                                "error_code": "AUDIO_CHUNK_TOO_LARGE",
                                "detail": (
                                    f"Decoded audio chunk ({len(pcm_bytes)} bytes) exceeds "
                                    f"MAX_AUDIO_CHUNK_BYTES ({cfg.MAX_AUDIO_CHUNK_BYTES})."
                                ),
                                "server_timestamp": time.time(),
                            },
                        )
                        continue

                    transcript_text = await stt.transcribe_chunk(
                        session_id=packet.session_id,
                        pcm_bytes=pcm_bytes,
                        language=packet.metadata.language,
                    )
                    effective_packet = InboundPacket(
                        session_id=packet.session_id,
                        client_timestamp=packet.client_timestamp,
                        stream_type=StreamType.TEXT_TRANSCRIPT,
                        payload=transcript_text,
                        metadata=packet.metadata,
                    )

                verdict = scr.evaluate_packet(effective_packet)
                total_elapsed_ms = round(
                    (time.perf_counter_ns() - frame_start_ns) / 1_000_000.0, 3
                )
                verdict.latency_ms = total_elapsed_ms
                telemetry.record(total_elapsed_ms)

                logger.info(
                    "Session=%s | Stream=%s | Threat=%s | Score=%.4f | LockUPI=%s | Cues=%s | Latency=%.3fms",
                    verdict.session_id,
                    packet.stream_type.value,
                    verdict.threat_level.value,
                    verdict.composite_threat_score,
                    verdict.mitigation.lock_upi,
                    verdict.detected_cue_tokens,
                    verdict.latency_ms,
                )

                await manager.send_json(websocket, verdict.model_dump())

        except WebSocketDisconnect:
            manager.disconnect(websocket, transcriber=stt)
        except Exception as exc:
            logger.exception("Unhandled error on WebSocket stream: %s", exc)
            manager.disconnect(websocket, transcriber=stt)

    return application


app = create_app()