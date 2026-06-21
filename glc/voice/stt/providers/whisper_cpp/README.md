# whisper.cpp STT provider (local, offline)

A speech-to-text provider that transcribes audio on your own machine using the
[whisper.cpp](https://github.com/ggerganov/whisper.cpp) `whisper-cli` binary —
no network, no API key, no per-request cost. It plugs into the GLC voice layer
as the `local` STT option and returns the same `TranscribeResult` shape as the
cloud providers, so callers can't tell which engine ran.

- **Provider name:** `whisper_cpp`
- **Selected via:** `prefer="local"` on `POST /v1/transcribe`
- **Cost:** `0.0` USD per call (runs locally)
- **Owned files:** `adapter.py`, `wrapper.py`, `schemas.py`

---

## Prerequisites

You need two things on the machine that runs the gateway:

1. **A `whisper-cli` binary on your `PATH`.** Build it from source:
   ```bash
   git clone https://github.com/ggerganov/whisper.cpp
   cd whisper.cpp && cmake -B build && cmake --build build --config Release
   # copy/symlink the resulting binary onto PATH, e.g.:
   sudo cp build/bin/whisper-cli /usr/local/bin/whisper-cli
   ```
   Verify: `whisper-cli --help` should print usage.

2. **A base GGML model** at `~/.glc/models/whisper-base/ggml-base.bin`:
   ```bash
   ./daemon/install.sh --models     # creates ~/.glc/models and prints instructions
   ```
   > Note: `install.sh --models` creates the directory and tells you what to
   > fetch — it does **not** auto-download the whisper model. Download the base
   > model yourself and drop it in place:
   ```bash
   mkdir -p ~/.glc/models/whisper-base
   # from your whisper.cpp checkout:
   bash ./models/download-ggml-model.sh base
   cp models/ggml-base.bin ~/.glc/models/whisper-base/ggml-base.bin
   ```

That's the whole setup. The model is ~150 MB and runs in roughly real time on
Apple Silicon.

### Configuration (optional)

| Env var | Default | Purpose |
|---------|---------|---------|
| `GLC_WHISPER_MODEL_DIR` | `~/.glc/models/whisper-base` | Where `ggml-base.bin` lives |
| `GLC_WHISPER_VAD_MODEL` | `<model dir>/ggml-silero-v6.2.0.bin` | Optional Silero VAD model (see below) |

---

## Usage

Audio must be **16 kHz mono 16-bit PCM** (whisper.cpp's native format); WAV is
the expected container. Send it base64-encoded:

```bash
curl -s localhost:8000/v1/transcribe \
  -H 'content-type: application/json' \
  -d "{\"audio_b64\": \"$(base64 -i sample.wav)\", \"mime\": \"audio/wav\", \"prefer\": \"local\"}"
```

Response:

```json
{
  "text": "hello world",
  "language": "en",
  "duration_ms": 1200,
  "provider": "whisper_cpp",
  "cost_usd": 0.0
}
```

Or call the provider directly in Python:

```python
from glc.voice.stt.providers.whisper_cpp.adapter import Provider

provider = Provider()
result = await provider.transcribe(audio_bytes, "audio/wav")
print(result.text)
```

---

## How it works

1. **Silence short-circuit (VAD pre-filter).** Empty or near-silent input (peak
   amplitude ≤ 32) returns an empty `TranscribeResult` immediately, without
   shelling out. This saves hundreds of ms of subprocess startup for clips that
   would transcribe to nothing. (This is *not* whisper.cpp's `--vad`, which runs
   *inside* the subprocess after startup is already paid.)
2. **Subprocess transcription.** For real audio, the adapter writes the bytes to
   a temp file and runs `whisper-cli -m <model> -f <audio> -oj`, then parses the
   JSON whisper.cpp emits next to the input file.
3. **Native VAD for long inputs.** Clips longer than ~30 s are run with
   whisper.cpp's `--vad` flag to trim internal silence. This needs a Silero VAD
   model; if it isn't present the provider **warns and falls back** to a plain
   run rather than failing.
4. **Uniform results & errors.** Every success returns a `TranscribeResult`
   (`language` defaults to `"en"`, `cost_usd` is always `0.0`). Any failure —
   missing binary, missing model, non-zero exit — is wrapped as a single
   `STTError`, so callers see one error type regardless of provider.

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `STTError: whisper-cli binary not found on PATH` | Build whisper.cpp and put `whisper-cli` on `PATH` (step 1). |
| `STTError: whisper base model not found at …` | Place `ggml-base.bin` under `~/.glc/models/whisper-base/` (step 2), or set `GLC_WHISPER_MODEL_DIR`. |
| `STTError: whisper-cli failed: …` | Non-zero exit — usually a malformed/non-16 kHz WAV. Re-encode to 16 kHz mono 16-bit. |
| `VAD requested but Silero model not found …` warning | Harmless — long-input VAD degraded to a plain run. Provide the Silero model at `GLC_WHISPER_VAD_MODEL` to enable it. |
| Empty `text` on real audio | Input was below the silence floor, or whisper produced no segments. Check the clip is audible and 16 kHz. |

---

## Limitations

- Expects 16 kHz mono 16-bit PCM; other sample rates may transcribe poorly or
  fail. Re-encode upstream.
- The base model trades accuracy for speed and degrades on heavily accented
  speech. Swap in a larger GGML model via `GLC_WHISPER_MODEL_DIR` if needed.
- `prefer="streaming"` is **not** served here — streaming STT goes through the
  Gemini Live WebSocket route, not this synchronous endpoint.
- First run after a cold start pays model-load time; subsequent calls are faster.

---

## Tests

```bash
uv run pytest tests/voice/stt/test_whisper_cpp.py
```

Seven tests cover the result contract and the silence short-circuit. They run
against an injected mock (`WhisperCppMock`), so they need **no** binary or model
— the real subprocess path is exercised by the manual demo above.
