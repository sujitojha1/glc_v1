"""Production-path error propagation for the whisper.cpp slot (#8).

The mock raises STTError directly (covered by the canonical suite); these
assert the *production* path wraps raw subprocess/runtime failures into
STTError, and that an STTError raised downstream bubbles up unchanged.
"""

from __future__ import annotations

import subprocess

import pytest

from glc.voice.stt.base import STTError
from glc.voice.stt.providers.whisper_cpp.adapter import Provider

# Non-silent, short audio so we reach the production path (no mock configured).
LOUD = b"\x00\x40" * 16000  # 1s, peak 16384


@pytest.mark.asyncio
async def test_called_process_error_wrapped_with_status(monkeypatch):
    def boom(audio, mime, vad=False):
        raise subprocess.CalledProcessError(
            returncode=2, cmd=["whisper-cli"], stderr="bad model"
        )

    monkeypatch.setattr("glc.voice.stt.providers.whisper_cpp.wrapper.run_whisper_cpp", boom)
    with pytest.raises(STTError) as ei:
        await Provider().transcribe(LOUD, "audio/wav")
    assert ei.value.status == 2
    assert "bad model" in str(ei.value)


@pytest.mark.asyncio
async def test_runtime_error_wrapped(monkeypatch):
    def boom(audio, mime, vad=False):
        raise RuntimeError("whisper-cli binary not found on PATH")

    monkeypatch.setattr("glc.voice.stt.providers.whisper_cpp.wrapper.run_whisper_cpp", boom)
    with pytest.raises(STTError) as ei:
        await Provider().transcribe(LOUD, "audio/wav")
    assert "not found" in str(ei.value)


@pytest.mark.asyncio
async def test_production_success_maps_to_transcribe_result(monkeypatch):
    monkeypatch.setattr(
        "glc.voice.stt.providers.whisper_cpp.wrapper.run_whisper_cpp",
        lambda audio, mime, vad=False: ("hi there", "en", 321),
    )
    r = await Provider().transcribe(LOUD, "audio/wav")
    assert (r.text, r.language, r.duration_ms) == ("hi there", "en", 321)
    assert r.provider == "whisper_cpp"
    assert r.cost_usd == 0.0


def test_schemas_module_imports():
    import glc.voice.stt.providers.whisper_cpp.schemas as s

    assert s is not None


@pytest.mark.asyncio
async def test_stterror_bubbles_unchanged(monkeypatch):
    original = STTError("upstream is down", status=503)

    def boom(audio, mime, vad=False):
        raise original

    monkeypatch.setattr("glc.voice.stt.providers.whisper_cpp.wrapper.run_whisper_cpp", boom)
    with pytest.raises(STTError) as ei:
        await Provider().transcribe(LOUD, "audio/wav")
    assert ei.value is original
    assert ei.value.status == 503
