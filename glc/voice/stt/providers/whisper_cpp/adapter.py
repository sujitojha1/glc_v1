"""whisper.cpp (local, offline) STT provider.

Subclasses `STTProvider` and returns the canonical `TranscribeResult`
shape so callers can't tell which provider ran. Silent input is
VAD-detected and short-circuited before any upstream/subprocess work.
With no mock configured, the real whisper-cli subprocess path runs.
Error propagation hardening lands in a later slice.
"""

from __future__ import annotations

import array
import asyncio
import subprocess

from glc.voice.stt.base import STTError, STTProvider, TranscribeResult

# Audio is assumed 16 kHz mono 16-bit PCM (whisper.cpp's native format).
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
# Max int16 magnitude at/below which a clip counts as silence.
SILENCE_MAX_AMPLITUDE = 32
# Inputs longer than this are VAD-trimmed (slot README: ">~30s").
VAD_LENGTH_THRESHOLD_S = 30.0


def _max_amplitude(audio: bytes) -> int:
    """Peak int16 sample magnitude (little-endian); 0 for empty/odd-only."""
    n = len(audio) - (len(audio) % BYTES_PER_SAMPLE)
    if n == 0:
        return 0
    samples = array.array("h")
    samples.frombytes(audio[:n])
    return max((abs(s) for s in samples), default=0)


def _is_silent(audio: bytes) -> bool:
    """True for empty or near-silent audio (peak amplitude under the floor).

    Shelling out to whisper-cli on silence wastes hundreds of ms of
    subprocess startup for an empty transcript, so the adapter
    short-circuits these before touching upstream. Native `--vad` does
    NOT cover this: it runs inside the subprocess, after startup is paid.
    """
    return not audio or _max_amplitude(audio) <= SILENCE_MAX_AMPLITUDE


def _duration_s(audio: bytes) -> float:
    """Approximate clip length in seconds (16 kHz mono 16-bit assumption)."""
    return len(audio) / (SAMPLE_RATE * BYTES_PER_SAMPLE)


def _should_use_vad(audio: bytes) -> bool:
    """VAD-trim only long inputs, where internal silence inflates latency."""
    return _duration_s(audio) > VAD_LENGTH_THRESHOLD_S


class Provider(STTProvider):
    name = "whisper_cpp"

    def _empty_result(self) -> TranscribeResult:
        return TranscribeResult(
            text="",
            language="en",
            duration_ms=0,
            provider=self.name,
            cost_usd=0.0,
        )

    async def transcribe(self, audio: bytes, mime: str) -> TranscribeResult:
        # VAD short-circuit: skip the upstream dispatch below entirely on
        # silent/empty input so we never pay subprocess startup for an
        # empty transcript.
        if _is_silent(audio):
            return self._empty_result()

        mock = self.config.get("mock")
        if mock is not None:
            r = await mock.transcribe(audio, mime)
            return TranscribeResult(
                text=r.text,
                language=r.language,
                duration_ms=r.duration_ms,
                provider=self.name,
                cost_usd=0.0,
            )

        # Production path: lazily import the subprocess wrapper so module
        # import stays cheap and free of subprocess/binary assumptions
        # (NFR-5). Run the blocking subprocess off the event loop.
        from .wrapper import run_whisper_cpp

        use_vad = _should_use_vad(audio)
        try:
            text, language, duration_ms = await asyncio.to_thread(run_whisper_cpp, audio, mime, use_vad)
        except STTError:
            # Already in the canonical shape (e.g. raised upstream) — let it pass.
            raise
        except subprocess.CalledProcessError as e:
            # Non-zero exit from whisper-cli: surface its status + stderr.
            detail = (e.stderr or "").strip() or str(e)
            raise STTError(f"whisper-cli failed: {detail}", status=e.returncode) from e
        except Exception as e:
            # Binary/model missing, decode failure, etc. — wrap as STTError so
            # callers see one error type regardless of provider (IF-3).
            raise STTError(f"whisper_cpp transcription failed: {e}") from e
        return TranscribeResult(
            text=text,
            language=language,
            duration_ms=duration_ms,
            provider=self.name,
            cost_usd=0.0,
        )
