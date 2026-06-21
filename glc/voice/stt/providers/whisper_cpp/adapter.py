"""whisper.cpp (local, offline) STT provider.

Subclasses `STTProvider` and returns the canonical `TranscribeResult`
shape so callers can't tell which provider ran. Silent input is
VAD-detected and short-circuited before any upstream/subprocess work.
Mock-upstream delegation, error propagation, and the production
subprocess path land in later slices.
"""

from __future__ import annotations

from glc.voice.stt.base import STTProvider, TranscribeResult


def _is_silent(audio: bytes) -> bool:
    """True for empty or zero-amplitude (pure-silence) audio.

    Shelling out to whisper-cli on silence wastes hundreds of ms of
    subprocess startup for an empty transcript, so the adapter
    short-circuits these before touching upstream.
    """
    return not audio or not any(audio)


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

        # Production subprocess path lands in a later slice (#9).
        raise NotImplementedError(
            "whisper_cpp production subprocess path not yet wired (see #9)."
        )
