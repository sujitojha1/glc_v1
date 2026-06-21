"""whisper.cpp (local, offline) STT provider.

Subclasses `STTProvider` and returns the canonical `TranscribeResult`
shape so callers can't tell which provider ran. Mock-upstream
delegation, VAD silence short-circuit, error propagation, and the
production subprocess path land in later slices.
"""

from __future__ import annotations

from glc.voice.stt.base import STTProvider, TranscribeResult


class Provider(STTProvider):
    name = "whisper_cpp"

    async def transcribe(self, audio: bytes, mime: str) -> TranscribeResult:
        return TranscribeResult(
            text="",
            language="en",
            duration_ms=0,
            provider=self.name,
            cost_usd=0.0,
        )
