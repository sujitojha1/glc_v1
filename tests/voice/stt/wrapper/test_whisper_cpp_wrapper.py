"""run_whisper_cpp subprocess-body tests for the whisper.cpp slot (#10).

The adapter tests monkeypatch run_whisper_cpp, so the wrapper's own
argv-build → subprocess → JSON-parse flow is exercised here with a faked
binary + filesystem (no real whisper-cli needed).
"""

from __future__ import annotations

import json
import subprocess
import types
from pathlib import Path

import pytest

from glc.voice.stt.providers.whisper_cpp import wrapper


def _fake_which(found="/fake/whisper-cli"):
    return lambda name: found


@pytest.fixture
def model(tmp_path, monkeypatch):
    m = tmp_path / "ggml-base.bin"
    m.write_bytes(b"model")
    monkeypatch.setattr(wrapper, "MODEL_FILE", m)
    return m


def test_run_parses_json_output(model, monkeypatch):
    monkeypatch.setattr(wrapper.shutil, "which", _fake_which())

    def fake_run(argv, **kw):
        # whisper-cli writes <input>.json next to the -f input file.
        audio = argv[argv.index("-f") + 1]
        Path(audio + ".json").write_text(
            json.dumps(
                {
                    "transcription": [
                        {"text": " hello", "offsets": {"to": 1234}},
                    ],
                    "language": "en",
                }
            )
        )
        return types.SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)
    text, language, duration_ms = wrapper.run_whisper_cpp(b"AUDIO", "audio/wav")
    assert text == "hello"
    assert language == "en"
    assert duration_ms == 1234


def test_run_falls_back_to_stdout_without_json(model, monkeypatch):
    monkeypatch.setattr(wrapper.shutil, "which", _fake_which())
    monkeypatch.setattr(
        wrapper.subprocess,
        "run",
        lambda argv, **kw: types.SimpleNamespace(stdout="  plain text  ", returncode=0),
    )
    text, language, duration_ms = wrapper.run_whisper_cpp(b"AUDIO", "audio/ogg")
    assert text == "plain text"
    assert language == "en"
    assert duration_ms == 0


def test_run_raises_when_binary_missing(model, monkeypatch):
    monkeypatch.setattr(wrapper.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="whisper-cli binary not found"):
        wrapper.run_whisper_cpp(b"AUDIO", "audio/wav")


def test_run_raises_when_model_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(wrapper.shutil, "which", _fake_which())
    monkeypatch.setattr(wrapper, "MODEL_FILE", tmp_path / "absent.bin")
    with pytest.raises(RuntimeError, match="model not found"):
        wrapper.run_whisper_cpp(b"AUDIO", "audio/wav")


def test_run_propagates_called_process_error(model, monkeypatch):
    monkeypatch.setattr(wrapper.shutil, "which", _fake_which())

    def boom(argv, **kw):
        raise subprocess.CalledProcessError(returncode=1, cmd=argv, stderr="boom")

    monkeypatch.setattr(wrapper.subprocess, "run", boom)
    with pytest.raises(subprocess.CalledProcessError):
        wrapper.run_whisper_cpp(b"AUDIO", "audio/wav")
