"""VAD gating tests for the whisper.cpp slot (#18).

The mock does not model whisper-cli's native VAD, so these exercise the
pure decision/argv helpers directly: silence short-circuit, the ~30s
length gate, and argv construction incl. the missing-model fallback.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from glc.voice.stt.providers.whisper_cpp import wrapper
from glc.voice.stt.providers.whisper_cpp.adapter import (
    SAMPLE_RATE,
    Provider,
    _is_silent,
    _max_amplitude,
    _should_use_vad,
)
from tests.voice.stt.mocks.whisper_cpp_mock import WhisperCppMock

BYTES_PER_S = SAMPLE_RATE * 2  # 16 kHz mono 16-bit


def _loud(seconds: float) -> bytes:
    # 0x4000 = 16384 little-endian int16, well above the silence floor.
    return b"\x00\x40" * int(SAMPLE_RATE * seconds)


# ── silence detection ──────────────────────────────────────────────


def test_is_silent_empty():
    assert _is_silent(b"")


def test_is_silent_zero_amplitude():
    assert _is_silent(b"\x00" * BYTES_PER_S)


def test_max_amplitude_loud():
    assert _max_amplitude(_loud(0.1)) == 16384


def test_max_amplitude_single_odd_byte_is_zero():
    # < one full 16-bit sample → no samples to measure → treated as silent.
    assert _max_amplitude(b"\x05") == 0
    assert _is_silent(b"\x05")


def test_not_silent_for_loud_audio():
    assert not _is_silent(_loud(0.1))


# ── length gate ────────────────────────────────────────────────────


def test_should_use_vad_false_for_short():
    assert _should_use_vad(_loud(10)) is False


def test_should_use_vad_true_for_long():
    assert _should_use_vad(_loud(35)) is True


# ── argv construction ──────────────────────────────────────────────


def test_build_argv_no_vad():
    argv = wrapper._build_argv("whisper-cli", Path("/m.bin"), Path("/a.wav"), vad=False)
    assert "--vad" not in argv
    assert argv[-1] == "-oj"


def test_build_argv_vad_with_model(tmp_path, monkeypatch):
    vm = tmp_path / "ggml-silero.bin"
    vm.write_bytes(b"x")
    monkeypatch.setenv("GLC_WHISPER_VAD_MODEL", str(vm))
    argv = wrapper._build_argv("whisper-cli", Path("/m.bin"), Path("/a.wav"), vad=True)
    assert "--vad" in argv
    assert argv[argv.index("-vm") + 1] == str(vm)


def test_build_argv_vad_missing_model_falls_back(tmp_path, monkeypatch):
    missing = tmp_path / "nope.bin"
    monkeypatch.setenv("GLC_WHISPER_VAD_MODEL", str(missing))
    with pytest.warns(UserWarning, match="VAD requested"):
        argv = wrapper._build_argv("whisper-cli", Path("/m.bin"), Path("/a.wav"), vad=True)
    assert "--vad" not in argv


# ── silent input never reaches the subprocess ──────────────────────


@pytest.mark.asyncio
async def test_silent_input_never_invokes_upstream():
    mock = WhisperCppMock()
    adapter = Provider(config={"mock": mock})
    r = await adapter.transcribe(b"\x00" * BYTES_PER_S, "audio/wav")
    assert r.text == ""
    assert mock.subprocess_call_count == 0
    assert not mock.received_calls
