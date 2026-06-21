# whisper-cli reference (whisper.cpp)

Reference for the `whisper-cli` binary the `whisper_cpp` STT slot shells out
to via [`wrapper.py`](../glc/voice/stt/providers/whisper_cpp/wrapper.py). Captured against a real build so the
adapter authors don't have to reverse-engineer flags from the source.

- **Binary version:** `whisper.cpp version: 1.9.1`
- **Upstream:** https://github.com/ggerganov/whisper.cpp
- **Supported audio formats:** `flac, mp3, ogg, wav` (decoded via miniaudio)
- **Invocation shape used by the slot:**
  `whisper-cli -m <model> -f <audio> -oj` → writes `<audio>.json` next to the input.

## How the slot invokes it

`wrapper.run_whisper_cpp()` runs:

```sh
whisper-cli -m ~/.glc/models/whisper-base/ggml-base.bin -f <tmp>.wav -oj
```

then parses the sibling `<tmp>.wav.json`, joining `transcription[].text` into one
string and reading `language` and the last segment's end offset as `duration_ms`.

## Options (grouped)

### Input / output
| Flag | Default | Meaning |
|------|---------|---------|
| `-f, --file FNAME` | — | input audio file path |
| `-m, --model FNAME` | `models/ggml-base.en.bin` | model path |
| `-of, --output-file FNAME` | — | output path (no extension) |
| `-oj, --output-json` | false | write result as JSON (what the slot parses) |
| `-ojf, --output-json-full` | false | richer JSON (per-token data) |
| `-otxt / -ovtt / -osrt / -olrc / -ocsv` | false | other output formats |
| `-np, --no-prints` | false | print only results (quiet stderr) |
| `-nt, --no-timestamps` | false | omit timestamps |

### Decoding / quality
| Flag | Default | Meaning |
|------|---------|---------|
| `-t, --threads N` | 4 | CPU threads |
| `-p, --processors N` | 1 | parallel processors |
| `-d, --duration N` | 0 | process only first N ms |
| `-ot, --offset-t N` | 0 | start offset (ms) |
| `-bo, --best-of N` | 5 | candidates kept |
| `-bs, --beam-size N` | 5 | beam search width |
| `-ac, --audio-ctx N` | 0 (all) | audio context size |
| `-ml, --max-len N` | 0 | max segment length (chars) |
| `-sow, --split-on-word` | false | split on word not token |
| `-wt, --word-thold N` | 0.01 | word timestamp prob threshold |
| `-et, --entropy-thold N` | 2.40 | decoder-fail entropy threshold |
| `-lpt, --logprob-thold N` | -1.00 | decoder-fail logprob threshold |
| `-nth, --no-speech-thold N` | 0.60 | no-speech threshold |
| `-tp, --temperature N` | 0.00 | sampling temperature |
| `-tpi, --temperature-inc N` | 0.20 | temperature fallback increment |
| `-nf, --no-fallback` | false | disable temperature fallback |
| `-sns, --suppress-nst` | false | suppress non-speech tokens |

### Language / mode
| Flag | Default | Meaning |
|------|---------|---------|
| `-l, --language LANG` | en | spoken language (`auto` to detect) |
| `-dl, --detect-language` | false | detect language then exit |
| `-tr, --translate` | false | translate to English |
| `-di, --diarize` | false | stereo diarization |
| `-tdrz, --tinydiarize` | false | tinydiarize (needs tdrz model) |
| `--prompt PROMPT` | — | initial prompt |

### Hardware
| Flag | Default | Meaning |
|------|---------|---------|
| `-ng, --no-gpu` | false | disable GPU |
| `-fa / -nfa, --flash-attn` | true | flash attention on/off |
| `-dev, --device N` | 0 | GPU device id |

(Full `--help` also lists karaoke output, grammar-guided decoding, OpenVINO,
and DTW token-level timestamps — not used by this slot.)

## Voice Activity Detection (VAD)

**This is native to whisper.cpp — you do not implement VAD in Python.** The
slot README's "VAD-trim long audio before transcription" guidance maps directly
onto whisper-cli's `--vad` subsystem.

### How it works
With `--vad`, audio is first run through a **separate VAD model** that detects
speech segments. Only those segments are extracted and passed to whisper, so
long internal silences are never decoded. This reduces the audio whisper
processes and **significantly speeds up transcription** on inputs with silence.

This is distinct from the adapter's `_is_silent()` short-circuit:

| | `_is_silent()` (adapter, issue #6) | `--vad` (whisper-cli) |
|---|---|---|
| Scope | whole clip is zero-amplitude | clip has speech + internal silence |
| Action | skip the subprocess entirely | trim silent spans, transcribe the rest |
| Where | Python, before spawning | inside whisper-cli |
| Goal | avoid startup for empty transcript | cut decode latency, same transcript |

### Requirements
A VAD model is **required** in addition to `--vad`. Supported: **Silero-VAD**.

```sh
# download (Linux/macOS) — ~864 KB
./models/download-vad-model.sh silero-v6.2.0
# → models/ggml-silero-v6.2.0.bin
```

### Invocation
```sh
whisper-cli \
  -m ~/.glc/models/whisper-base/ggml-base.bin \
  -vm <path>/ggml-silero-v6.2.0.bin \
  --vad \
  -f audio.wav -oj
```

### VAD options
| Flag | Default | Meaning |
|------|---------|---------|
| `--vad` | false | enable VAD |
| `-vm, --vad-model FNAME` | — | VAD model path (**required** with `--vad`) |
| `-vt, --vad-threshold N` | 0.50 | speech-probability threshold; frames above are speech |
| `-vspd, --vad-min-speech-duration-ms N` | 250 | discard speech segments shorter than this (filters brief noise) |
| `-vsd, --vad-min-silence-duration-ms N` | 100 | silence must last this long to end a segment; shorter silence stays inside speech |
| `-vmsd, --vad-max-speech-duration-s N` | FLT_MAX | auto-split segments longer than this at silence points >98ms |
| `-vp, --vad-speech-pad-ms N` | 30 | padding added before/after each segment so edges aren't clipped |
| `-vo, --vad-samples-overlap N` | 0.10 | seconds of overlap carried between concatenated segments |

### Per-option exploration (this install, v1.9.1)

Each option was run individually and inspected at the **VAD-segment level** (not
just the transcript). whisper-cli logs the detected segments to stderr — enable
by *not* passing `-np` and reading lines like:

```
whisper_vad_segments_from_probs: Final speech segments after filtering: 8
whisper_vad_segments_from_probs: VAD segment 0: start = 1.79, end = 3.77 (duration: 1.98)
```

Swept on a 27s synthetic clip — `1.5s silence + jfk(11s) + 2s silence + jfk(11s)
+ 1.5s silence` — with `silero-v6.2.0` + base model. **Baseline `--vad` default
→ 8 segments**, first at `1.79s` (leading silence dropped). Per-option result:

| Option | Setting | Segment-level result vs. default (8 segs) |
|--------|---------|-------------------------------------------|
| `-vt, --vad-threshold` | `0.1` | **2 segments**; seg0 = `1.54 → 12.64` (11.1s) — loose threshold admits more, bridges gaps into long segments |
| | `0.9` | **10 segments**; seg0 = `1.79 → 3.74` (1.95s) — strict threshold fragments speech |
| `-vsd, --vad-min-silence-duration-ms` | `800` | **6 segments** — sub-800ms gaps no longer split (seg merged to `6.88 → 12.54`, 5.66s) |
| `-vspd, --vad-min-speech-duration-ms` | `2000` | **4 segments** — ⚠️ every segment <2s **discarded**, incl. the opening; transcript starts mid-sentence. Dangerous knob. |
| `-vp, --vad-speech-pad-ms` | `0` | seg0 = `1.82 → 3.74` (tight, 1.92s) |
| | `400` | seg0 = `1.42 → 4.14` (2.72s) — pads ~0.4s each edge, prevents clipping |
| `-vmsd, --vad-max-speech-duration-s` | `1`–`5` | no visible split on this clip — segments are already ≤2.5s and max-speech only splits *at internal silence points >98ms*, which dense speech lacks. Mechanism confirmed via source/docs, not reproducible here without a longer dense segment. |
| `-vo, --vad-samples-overlap` | `0.5`–`1.0` | segment list **unchanged** — overlap affects audio carried across segment *joins* (concatenation), not boundary detection |

Takeaways:
- **Defaults are safe.** Start there.
- `-vt` is the master recall/precision dial: **low → fewer, longer** segments
  (may swallow noise); **high → more, fragmented** segments (may clip onsets).
- `-vspd` (min-speech) is the **dangerous** knob — too high *silently drops real
  speech*; verified it deleted the opening phrase at 2000ms.
- `-vsd` (min-silence) controls how long a pause must be to break a segment.
- `-vp` (pad) widens segment edges; `-vo` (overlap) only affects joins.
- `-vmsd` (max-speech) only acts on over-long *dense* segments — niche.

### Verified behaviour
On `samples/jfk.wav` (base model), VAD trims the leading/trailing silence:

```
# without --vad
[00:00:00.000 --> 00:00:10.500]  And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.

# with --vad (silero-v6.2.0)
[00:00:00.320 --> 00:00:08.170]  And so, my fellow Americans, ask not what your country can do for you,
[00:00:08.170 --> 00:00:10.470]  ask what you can do for your country.
```

Same transcript content; decoded audio window starts at 0.320s instead of
0.000s. The benefit scales with how much silence the input contains — hence the
README's ">~30s" guidance, where long silent stretches otherwise inflate latency.

### Implications for the slot
- Prefer the **native `--vad`** path over a hand-rolled Python trimmer: it's
  maintained upstream, segment-aware, and avoids re-encoding audio.
- It adds a dependency: the Silero VAD model must be provisioned alongside the
  base model (extends the runtime setup tracked in #17).
- The mock (`WhisperCppMock`) does not model VAD segmentation, so this is a
  **production-path** concern (relates to #9) and currently has no test — any
  implementation should ship a regression test.
