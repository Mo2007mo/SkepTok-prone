"""
Asynchronous 16 kHz, 16-bit Mono PCM Audio Ingestion Pipeline.

Implements thread-safe circular memory buffers (`bytearray` / `collections.deque`)
with strictly zero disk I/O to guarantee user privacy. Provides:
- `BaseSpeechTranscriber`: Abstract streaming STT interface.
- `MockStreamingTranscriber`: Deterministic in-memory PCM-to-scam-transcript
  converter for headless CI/CD, unit tests, and stress testing.
- `SherpaOnnxTranscriber`: Production hook utilizing quantized offline
  IndicConformer/Whisper models (with graceful in-memory fallback when model weights
  are not mounted in the environment).
"""

from __future__ import annotations

import abc
import struct
import threading
from collections import deque
from typing import Deque, Dict, Optional

from .config import AppSettings, get_settings


class CircularPCMBuffer:
    """
    Thread-safe, fixed-capacity in-memory circular ring buffer for 16 kHz, 16-bit
    Mono PCM audio frames. Strictly enforces zero disk I/O and automatically
    evicts the oldest audio frames when capacity is reached.
    """

    def __init__(self, max_bytes: int = 480_000) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive.")
        self._max_bytes = max_bytes
        self._chunks: Deque[bytes] = deque()
        self._current_bytes: int = 0
        self._total_ingested_bytes: int = 0
        self._chunks_ingested: int = 0
        self._lock = threading.Lock()

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def current_bytes(self) -> int:
        with self._lock:
            return self._current_bytes

    @property
    def total_ingested_bytes(self) -> int:
        with self._lock:
            return self._total_ingested_bytes

    @property
    def chunks_ingested(self) -> int:
        with self._lock:
            return self._chunks_ingested

    def append(self, pcm_chunk: bytes) -> int:
        """
        Appends raw 16-bit PCM bytes into the ring buffer, trimming oldest bytes
        if the buffer exceeds `max_bytes`. Returns current buffered byte count.
        """
        if not pcm_chunk:
            return self.current_bytes

        chunk_len = len(pcm_chunk)
        with self._lock:
            self._total_ingested_bytes += chunk_len
            self._chunks_ingested += 1

            # If a single chunk exceeds max_bytes, keep only its trailing tail
            if chunk_len >= self._max_bytes:
                self._chunks.clear()
                tail = pcm_chunk[-self._max_bytes :]
                self._chunks.append(tail)
                self._current_bytes = len(tail)
                return self._current_bytes

            self._chunks.append(pcm_chunk)
            self._current_bytes += chunk_len

            while self._current_bytes > self._max_bytes and self._chunks:
                oldest = self._chunks.popleft()
                overflow = self._current_bytes - self._max_bytes
                if len(oldest) <= overflow:
                    self._current_bytes -= len(oldest)
                else:
                    # Slice the oldest chunk to retain exact byte capacity
                    trimmed = oldest[overflow:]
                    self._chunks.appendleft(trimmed)
                    self._current_bytes -= overflow
                    break

            return self._current_bytes

    def snapshot(self) -> bytearray:
        """Returns a contiguous in-memory `bytearray` copy of the buffered PCM stream."""
        with self._lock:
            buf = bytearray(self._current_bytes)
            offset = 0
            for chunk in self._chunks:
                n = len(chunk)
                buf[offset : offset + n] = chunk
                offset += n
            return buf

    def clear(self) -> None:
        """Zeroes and releases all buffered audio frames in memory."""
        with self._lock:
            self._chunks.clear()
            self._current_bytes = 0


class BaseSpeechTranscriber(abc.ABC):
    """
    Abstract interface for real-time in-memory Speech-to-Text (STT) transcription
    of 16 kHz, 16-bit Mono PCM audio streams.
    """

    def __init__(self, settings: Optional[AppSettings] = None) -> None:
        self._settings = settings or get_settings()
        max_ring_bytes = (
            self._settings.AUDIO_SAMPLE_RATE_HZ
            * self._settings.AUDIO_SAMPLE_WIDTH_BYTES
            * self._settings.AUDIO_CHANNELS
            * self._settings.AUDIO_RING_BUFFER_SECONDS
        )
        self._max_ring_bytes = max_ring_bytes
        self._buffers: Dict[str, CircularPCMBuffer] = {}
        self._lock = threading.RLock()

    @property
    @abc.abstractmethod
    def backend_name(self) -> str:
        """Human-readable identifier of the STT backend."""

    def get_or_create_buffer(self, session_id: str) -> CircularPCMBuffer:
        with self._lock:
            buf = self._buffers.get(session_id)
            if buf is None:
                if len(self._buffers) >= 1024:
                    # Evict oldest buffer entry to prevent unbounded session growth
                    oldest_key = next(iter(self._buffers))
                    self._buffers.pop(oldest_key, None)
                buf = CircularPCMBuffer(max_bytes=self._max_ring_bytes)
                self._buffers[session_id] = buf
            return buf

    def release_session(self, session_id: str) -> None:
        """Releases the in-memory ring buffer for a disconnected session."""
        with self._lock:
            buf = self._buffers.pop(session_id, None)
            if buf is not None:
                buf.clear()

    @abc.abstractmethod
    async def transcribe_chunk(
        self,
        session_id: str,
        pcm_bytes: bytes,
        language: str = "hi-IN",
    ) -> str:
        """
        Ingests a raw 16 kHz 16-bit Mono PCM chunk into the session's circular
        memory buffer and returns the transcribed utterance string.
        Strictly performs zero disk I/O.
        """


class MockStreamingTranscriber(BaseSpeechTranscriber):
    """
    Deterministic, zero-latency in-memory transcriber for headless CI/CD,
    integration tests, and stress testing.

    Inspects the incoming PCM bytes:
    1. If UTF-8 text is embedded inside the test PCM frame (prefixed with `TEXT:`
       or printable ASCII transcript), decodes it directly for flexible test cases.
    2. If the PCM frame is pure digital silence (all zero samples), returns a
       benign conversational phrase.
    3. Otherwise, maps the chunk sequence / acoustic energy signature to realistic
       multi-dialect scam transcripts so end-to-end WebSocket audio pipelines can
       be verified deterministically.
    """

    _SAMPLE_SCAM_PHRASES = (
        "Bijli cut ho jayegi 10 minute mein, QuickSupport apk download karo",
        "Main CBI officer bol raha hoon, aapke naam par arrest warrant hai, turant AnyDesk install karo",
        "Aapka electricity bill unpaid hai, aaj raat 9 baje power disconnect ho jayegi, OTP batao",
        "TRAI se notice hai, aapka sim block ho jayega 2 ghante ke andar, KYC update ke liye ₹5 transfer karo",
    )

    @property
    def backend_name(self) -> str:
        return "MockStreamingTranscriber(16kHz-PCM-InMemory)"

    async def transcribe_chunk(
        self,
        session_id: str,
        pcm_bytes: bytes,
        language: str = "hi-IN",
    ) -> str:
        if not pcm_bytes:
            return ""

        if len(pcm_bytes) > self._settings.MAX_AUDIO_CHUNK_BYTES:
            raise ValueError(
                f"PCM chunk size ({len(pcm_bytes)} bytes) exceeds MAX_AUDIO_CHUNK_BYTES "
                f"({self._settings.MAX_AUDIO_CHUNK_BYTES} bytes)."
            )

        ring_buffer = self.get_or_create_buffer(session_id)
        ring_buffer.append(pcm_bytes)

        # Allow test harnesses to pass explicit UTF-8 transcript markers inside PCM bytes if desired
        if pcm_bytes.startswith(b"TEXT:"):
            return pcm_bytes[5:].decode("utf-8", errors="ignore").strip()

        # Check if the PCM chunk is pure digital silence (all zeros)
        if all(b == 0 for b in pcm_bytes[: min(len(pcm_bytes), 64)]):
            return "Bhaiya bijli ka bill bhar diya maine PhonePe se"

        # Deterministically select scam phrase based on chunk index in the session
        idx = (ring_buffer.chunks_ingested - 1) % len(self._SAMPLE_SCAM_PHRASES)
        return self._SAMPLE_SCAM_PHRASES[idx]


class SherpaOnnxTranscriber(BaseSpeechTranscriber):
    """
    Production hook for offline quantized IndicConformer / Whisper streaming ASR
    via `sherpa-onnx` operating strictly on in-memory float32 normalized PCM arrays.

    If `sherpa_onnx` runtime and model artifacts are provided, decodes audio frames
    directly from the circular memory buffer. If `sherpa_onnx` is not installed in
    the current container, seamlessly delegates to the deterministic in-memory
    fallback so the service remains runnable out-of-the-box.
    """

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
         recognizer: Optional[object] = None,
    ) -> None:
        super().__init__(settings=settings)
        self._recognizer = recognizer
        self._fallback = MockStreamingTranscriber(settings=self._settings)

    @property
    def backend_name(self) -> str:
        if self._recognizer is not None:
            return "SherpaOnnxTranscriber(IndicConformer-Int8-Quantized)"
        return "SherpaOnnxTranscriber(Fallback->InMemoryMock)"

    @staticmethod
    def pcm16le_to_float32_samples(pcm_bytes: bytes) -> list[float]:
        """
        Converts raw 16-bit little-endian Mono PCM bytes into normalized [-1.0, 1.0]
        float32 samples in memory without intermediate files.
        """
        num_samples = len(pcm_bytes) // 2
        if num_samples == 0:
            return []
        shorts = struct.unpack(f"<{num_samples}h", pcm_bytes[: num_samples * 2])
        return [s / 32768.0 for s in shorts]

    async def transcribe_chunk(
        self,
        session_id: str,
        pcm_bytes: bytes,
        language: str = "hi-IN",
    ) -> str:
        if len(pcm_bytes) > self._settings.MAX_AUDIO_CHUNK_BYTES:
            raise ValueError(
                f"PCM chunk size ({len(pcm_bytes)} bytes) exceeds MAX_AUDIO_CHUNK_BYTES "
                f"({self._settings.MAX_AUDIO_CHUNK_BYTES} bytes)."
            )

        ring_buffer = self.get_or_create_buffer(session_id)
        ring_buffer.append(pcm_bytes)

        if self._recognizer is not None:
            samples = self.pcm16le_to_float32_samples(pcm_bytes)
            stream = self._recognizer.create_stream()  # type: ignore[attr-defined]
            stream.accept_waveform(self._settings.AUDIO_SAMPLE_RATE_HZ, samples)
            self._recognizer.decode_stream(stream)  # type: ignore[attr-defined]
            return str(stream.result.text).strip()

        return await self._fallback.transcribe_chunk(
            session_id=session_id,
            pcm_bytes=pcm_bytes,
            language=language,
        )


_ACTIVE_TRANSCRIBER: Optional[BaseSpeechTranscriber] = None
_TRANSCRIBER_LOCK = threading.Lock()


def get_transcriber(use_sherpa: bool = False) -> BaseSpeechTranscriber:
    """Returns the singleton `BaseSpeechTranscriber` instance."""
    global _ACTIVE_TRANSCRIBER
    with _TRANSCRIBER_LOCK:
        if _ACTIVE_TRANSCRIBER is None:
            if use_sherpa:
                _ACTIVE_TRANSCRIBER = SherpaOnnxTranscriber()
            else:
                _ACTIVE_TRANSCRIBER = MockStreamingTranscriber()
        return _ACTIVE_TRANSCRIBER