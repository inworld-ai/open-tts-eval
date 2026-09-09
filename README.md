# Inworld TTS Open Evaluation Toolkit

[![CI](https://github.com/inworld-ai/open-tts-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/inworld-ai/open-tts-eval/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Typer 0.12+](https://img.shields.io/badge/typer-0.12%2B-009688.svg)](https://libraries.io/pypi/typer)
[![JiWER 3.0+](https://img.shields.io/badge/jiwer-3.0%2B-white.svg)](https://libraries.io/pypi/jiwer)
[![faster-whisper 1.0+](https://img.shields.io/badge/faster--whisper-1.0%2B-orange.svg)](https://libraries.io/pypi/faster-whisper)
[![Pydantic 2.0+](https://img.shields.io/badge/pydantic-2.0%2B-e92063.svg)](https://libraries.io/pypi/pydantic)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## TL;DR

> Two TTS evaluations can use the same data and both report WER, yet produce different numbers.
> The result depends on ASR settings, text normalization, the voices chosen, and synthesis parameters. 
> Naturalness and expressiveness are harder to compare because their definitions and scoring methods vary.
>
> We are open-sourcing **Inworld TTS Open Evaluation Toolkit** to support fair, independent
> evaluation across the speech industry. It is a simple, transparent framework for reproducible
> TTS testing.
>
> This is not a perfect or definitive way to evaluate speech models. It is a shared foundation:
> each run includes per-sample measurements and offline reports, with the full configuration saved
> in `summary.json`. Scores can be traced to the samples and settings that produced them. New
> metrics and datasets can be added, and the toolkit can evaluate audio from other systems.
>
> The repository includes baseline metrics and a
> [100-utterance dialogue stress set](data/inworld.tts.open_benchmak.en.json) that concentrates
> difficult cases into a small test run.
>
> To try it, ask your coding agent to clone this repository and evaluate a model. The repository
> includes agent instructions in [`AGENTS.md`](AGENTS.md) and [`CLAUDE.md`](CLAUDE.md).

The evaluation core runs offline and does not depend on internal infrastructure or datasets. The
optional [sampling subsystem](#sampling-generating-audio-to-evaluate) is the only component that
calls public provider APIs (Inworld, ElevenLabs, and Hume).

## What It Does

- Reads a JSONL or CSV manifest of `text` plus `audio_path`.
- Runs local ASR with `faster-whisper` or a mock backend for tests.
- Normalizes expected text and ASR transcript.
- Computes WER/CER, insertion/deletion/substitution rates, hallucination heuristics, duration,
  speech-rate proxy, silence, clipping, loudness proxy, and tail-click signals.
- Optionally runs heavier public metrics: NISQAv2, ECAPA speaker similarity, vowel/prolongation
  heuristic, voice-lens-style expressiveness proxy, and open-aligner availability checks.
- Applies configurable warn/fail thresholds and grades each model good/warn/fail per metric.
- Writes `results.jsonl`, `summary.json`, optional `results.csv`, `report_data.json`, and a
  minimal black-and-white `report.html` — a Metric Comparison table, a Model Health grid, and a
  Threshold Violations chapter — that renders fully offline (no CDN assets).
- Merges several runs (different providers, models, or voices) into a single comparative report
  with grouped metric tables, best/worst highlighting, and statistical-significance markers from
  a paired, per-text bootstrap.
- Records provenance: which evaluator code, ASR model, and normalizer produced every score, and
  which model and settings the provider actually served during sampling. Comparisons refuse runs
  scored by different evaluators.

## Install

Core install:

```bash
pip install -e .
```

Install with local ASR:

```bash
pip install -e ".[asr]"
```

Install all optional metric stacks:

```bash
pip install -e ".[all]"
```

## Manifest Format

JSONL is recommended:

```json
{"id":"sample_001","text":"Hello world.","audio_path":"audio/sample_001.wav"}
{"id":"sample_002","text":"The quick brown fox jumps over the lazy dog.","audio_path":"audio/sample_002.wav","speaker_id":"spk1","reference_audio_path":"voices/spk1_ref.wav"}
```

Required fields:

- `id`
- `text`
- `audio_path`

Optional fields:

- `speaker_id`
- `reference_audio_path`
- `language`
- `metadata`

Speaker similarity runs only when `reference_audio_path` exists and the similarity extra is installed.

## Sampling (generating audio to evaluate)

The `tts_assess.sampling` subsystem synthesizes a text dataset with prebuilt provider voices and
writes manifests this toolkit can evaluate directly. It ships with **Inworld**, **ElevenLabs**, and
**Hume** backends and is provider-pluggable (add a `TTSProvider` subclass under
`sampling/providers/`).

Each provider reads its API key from an environment variable (or a file via `--api-key-file`).
The default env var is `INWORLD_API_KEY`; override per provider with `--api-key-env`:

```bash
export INWORLD_API_KEY=...          # base64 key from the Inworld portal
tts-assess voices --provider inworld

export ELEVENLABS_API_KEY=...
tts-assess voices --provider elevenlabs --api-key-env ELEVENLABS_API_KEY
# ...or read the key from a file instead of the environment:
tts-assess voices --provider hume --api-key-file ~/hume_key.txt
```

Sample a dataset with chosen voices and models:

```bash
tts-assess sample data/inworld.tts.open_benchmak.en.json \
  --provider inworld \
  --voice Ashley --voice Sarah \
  --model inworld-tts-2 --model inworld-tts-1.5-max \
  --language en-US --format WAV --sample-rate 24000 \
  --output-dir out/samples
```

`data/inworld.tts.open_benchmak.en.json` is the bundled English benchmark: 100 short,
dialogue-style utterances with a realistic mix of lengths, punctuation, names, numbers, and
stage-direction/OCR noise, for stress-testing TTS on messy input.

Instead of naming voices you can take the first N from the catalog with `--num-voices 5`, or
`--num-voices 10 --shuffle-voices` for a seeded random-but-diverse selection.

The dataset is `.txt` (one utterance per line), `.json` / `.jsonl` (objects or strings with a
`text` field, optional `id` / `language`), or `.csv` with a `text` column. Output layout — **one
run directory per model**, named `<provider>-<model>`, each a ready `tts-assess run` input with
voices recorded as `speaker_id`:

```
out/samples/
  sampling_summary.json
  inworld-inworld-tts-2/
    manifest.jsonl        # id, text, audio_path, speaker_id (=voice), language, metadata
    sampling_meta.json    # per-run provider/model/voice/error detail
    audio/<voice>__<text-id>.wav
  inworld-inworld-tts-1.5-max/
    ...
```

Each manifest row's `metadata` records what was actually produced, not only what was asked for:
`api_model` (requested) and `returned_model` (what the provider reports having served),
`sent_request` (the request body minus the text), the probed `sample_rate_hz` / `channels` /
`duration_sec` of the returned audio alongside `requested_sample_rate_hz`, and a
`sampling_fingerprint` of the full request. A clip served by a different model than requested is
logged as an error and skipped, so a run labelled with one model never contains another's audio;
pass `--allow-model-mismatch` to keep such clips (flagged via `returned_model`). Text ids that
sanitize to the same file name (`a/b` vs `a:b`) get a short hash suffix instead of overwriting each
other.

Providers differ in which synthesis controls they accept, and the adapters reject a control they
cannot transmit rather than recording a setting the audio was not produced with:

| provider | `--speaking-rate` | `--temperature` | `--language` |
|---|---|---|---|
| `inworld` | sent | sent | sent |
| `elevenlabs` | sent as `voice_settings.speed` | rejected | sent as `language_code` only for models that enforce it (`eleven_turbo_v2_5`, `eleven_flash_v2_5`) |
| `hume` | sent as utterance `speed` | rejected | not supported by the API |

Hume requests always pin `version` (`octave-1` → `1`, `octave-2` → `2`); an omitted version would
let the API choose the model.

This maps cleanly onto the rest of the toolkit: evaluate each model's manifest, then compare.
Evaluate a run **into its own directory** so audio, manifest, results, and report end up together
with portable relative audio paths (the report's `<audio>` players then work when the folder is
opened or moved):

```bash
tts-assess run out/samples/inworld-inworld-tts-2/manifest.jsonl \
  -o out/samples/inworld-inworld-tts-2
tts-assess run out/samples/inworld-inworld-tts-1.5-max/manifest.jsonl \
  -o out/samples/inworld-inworld-tts-1.5-max
tts-assess compare out/samples/inworld-inworld-tts-2 out/samples/inworld-inworld-tts-1.5-max \
  --label "TTS 2" --label "TTS 1.5 Max" -o out/comparison
```

WAV output is the default because the evaluator decodes audio with `soundfile`; other encodings
(`MP3`, `FLAC`, `OGG_OPUS`, …) are available but may not be readable by every audio metric.

## Run

Create a default config:

```bash
tts-assess init-config tts-assess.yml
```

Assess any manifest of `text` + `audio_path` rows (the sampler writes these for you, or supply
your own):

```bash
tts-assess run out/samples/inworld-inworld-tts-1.5-max/manifest.jsonl \
  --config tts-assess.yml --output-dir out/eval
```

Use the mock ASR backend for smoke tests or demos without downloading Whisper:

```yaml
asr:
  backend: mock
```

Preview the HTML report with live reload:

```bash
tts-assess preview out/eval/report.html
```

To score a single clip in memory (a service, a notebook), call the per-pair core the pipeline
itself uses; the returned row has the same shape as a `results.jsonl` line:

```python
from tts_assess.config import load_config
from tts_assess.evaluate import evaluate_pair

row = evaluate_pair(wav_bytes, "Hello world.", load_config(Path("tts-assess.yml")))
row["status"], row["wer"], row["evaluator"]["code_hash"]
```

## Compare Runs

Assess each provider, model, or voice into its own output directory, then merge them into one
comparative report:

```bash
tts-assess run manifests/provider_a.jsonl -o out/provider_a
tts-assess run manifests/provider_b.jsonl -o out/provider_b
tts-assess compare out/provider_a out/provider_b \
  --label "Provider A" --label "Provider B" \
  --title "Provider A vs B" -o out/comparison
```

Each positional argument is a run's output directory (or a `results.jsonl` path). Labels default
to the directory name; pass one `--label` per run to override, in order. The comparison:

- Applies one shared threshold set to every run, so the tables are comparable. Clips whose
  measurement failed (decode, ASR, or a model error) stay failed; re-applying thresholds never
  turns an unmeasured clip into a pass.
- Rejects runs whose text/language cohort or per-speaker sampling profile differs; provider-specific
  sample and voice IDs may differ. Also rejects runs scored by different evaluators (ASR backend or
  model, normalizer, or measurement code), since an ASR change would otherwise read as a TTS
  difference. Runs scored before provenance was recorded produce a warning instead.
- **Metric Comparison** — each metric's mean and CI per run, highlighting the best run and marking
  with `*` the runs whose paired per-text difference to the best run excludes zero.
- **Model Health** — grades every run good/warn/fail per metric on its pass-rate, counting failed
  measurements as not passing.

It writes `comparison.html` and `comparison.json` into the output directory.

## Reports

The per-run report (`report.html`) and the cross-run comparison (`comparison.html`) share one
minimal, black-and-white layout, built from the same tables. Both render fully offline (no CDN
assets); the companion JSON (`report_data.json` / `comparison.json`) stays canonical.

- **Metric Comparison** — metrics grouped into **Accuracy** (WER + insertion / deletion /
  substitution rates, CER), **NISQAv2** (MOS + noisiness / discontinuity / coloration / loudness),
  **Subjective** (chars/sec, arousal, expressiveness), and **Silence** (silence ratio, lead/tail
  silence). Each cell is the mean with its 95% bootstrap CI beneath. Because every text is
  synthesized by several voices, clips are not independent draws: the CI is a **cluster bootstrap
  over texts** (clips sharing a text are resampled together), which is 1.5–2.5× wider than a naive
  per-clip bootstrap on the bundled benchmark. The best run per metric is highlighted green and the
  worst red, with `*` when the **paired per-text difference** to the best run has a bootstrap CI
  that excludes zero; the Silence group is shown without colouring. A cell that covers fewer clips
  than the run has shows `n=measured/total` and the number of failed measurements, so a high mean
  over a handful of clips cannot pass for a healthy run. (A per-run report has a single column, so
  no best/worst.)
- **Model Health** — for each threshold-backed metric, the share of clips that pass that metric's
  per-sample threshold (shown as a `pass if …` rule), graded **good (≥99%)**, **warn (≥95%)**, or
  **fail (<95%)** — bands configurable. A clip whose measurement failed counts as not passing, and
  the cell shows its measured/failed counts. WER/CER/insertions are omitted here (they live in
  Metric Comparison); the hallucination heuristics (`empty_transcript`, `repeated_span`,
  `tail_hallucination`, `tail_click_detected`) appear as `not flagged` rows.
- **Threshold Violations** *(per-run report only)* — warn/fail counts per metric and per failed
  measurement stage (audio decode, ASR, NISQA, …), plus worst-case examples that show the
  normalized `expected` vs `heard` text (so you can tell a real TTS error from a normalization
  mismatch or an **ASR mishearing** — Whisper often trips on accents and names, inflating WER) and
  an inline `<audio>` player when the clip exists on disk. Each sample is shown once, under its
  most-violated metric; failed measurements show the error instead of a transcript.

NISQA MOS (and other `optional_metrics`) appear only when enabled in config and their extra is
installed (`pip install -e ".[quality]"` for NISQA).

Relevant `reporting` config knobs:

```yaml
reporting:
  confidence_level: 0.95     # bootstrap CI level for metric means
  bootstrap_resamples: 2000  # bootstrap resamples per metric
  health_good_rate: 0.99     # Model Health: pass-rate >= this is "good"
  health_warn_rate: 0.95     # >= this is "warn"; below is "fail"
  embed_audio: true          # inline <audio> players in violation examples
  max_violation_examples: 5  # worst examples shown per violated metric
  max_worst_samples: 25      # maximum violation examples across the whole report
  question: "Is this run acceptable?" # question shown below the report title
  group_by_voice: true       # also compute per-voice stats into summary.json
```

Unknown config fields are rejected, so misspelled options cannot silently fall back to defaults.

## LibriTTS-R Example Workflow

LibriTTS-R is the recommended public English example source because it is TTS-oriented, includes
text, has improved audio quality, and is CC BY 4.0.

Suggested workflow:

1. Download a small LibriTTS-R split from OpenSLR.
2. Pick a tiny subset for customer-facing examples.
3. Create a manifest where each row maps the transcript to the corresponding WAV.
4. Run `tts-assess` and inspect `report.html`.

Keep bundled sample assets tiny. For larger examples, document the download step rather than
checking full datasets into the repository.

## Thresholds

Thresholds are starter bands, not universal pass/fail truth. Calibrate them on a known-good
baseline for each customer, domain, voice family, and use case.

Example (a subset of the defaults):

```yaml
thresholds:
  wer:
    warn: 0.05
    fail: 0.10
  tail_click_detected:
    fail_if_true: true            # boolean flag: any True is a fail
  repeated_span:
    fail_if_true: true            # hallucination heuristics are hard fails and show in reports
  vowel_prolongation_score:
    fail: 0.8                     # values at or above 0.8s fail
  speaker_similarity:
    warn_below: 0.65              # lower-is-worse metric
    fail_below: 0.55
```

A per-sample threshold decides pass/warn/fail for each clip; the **Model Health** table then grades
the whole model by what share of clips pass, using the `health_good_rate` / `health_warn_rate`
bands (default good ≥99%, warn ≥95%, fail <95%). A NaN or infinite metric value never passes: it
is reported as `invalid:<metric>` and fails the clip, and audio containing non-finite samples is
rejected at decode time.

How the audio heuristics behind these thresholds are defined:

- `silence_ratio` / lead & tail silence — a frame is silent below an adaptive threshold
  (`max(min(1.5 × p20, 0.1 × p95), −80 dBFS)` of frame RMS) estimated from the span between the
  clip's first and last non-silent frame, so padding a clip with silence can only add silence.
- `tail_click_score` — largest sample-to-sample jump in the last 30 ms over the RMS of the
  preceding 200 ms, computed after resampling that window to 24 kHz. The raw jump scales with the
  sample rate, so a fixed analysis rate keeps the `≥ 4.0` flag comparable between 24 kHz and
  44.1/48 kHz providers.
- `vowel_prolongation_score` — longest run of loud, low-centroid frames. The loudness cut is
  derived from non-silent frames and capped at half the speech level, so a long sustained vowel
  cannot raise the cut above itself and vanish; a longer sound always scores at least as long.
- NISQA is fed the clip at its **native** sample rate (the model accepts any rate). Downsampling to
  16 kHz first would discard everything above 8 kHz, exactly where hiss and codec artifacts live.

## Normalization

Three normalizer modes are supported:

- `english-basic`: transparent built-in English cleanup and number expansion. Numbers are parsed
  whole before punctuation is stripped: `1,000` → `one thousand`, `3.14` → `three point one
  four`, `12,345,678` → `twelve million …`, so a correct reading scores 0 and a wrong one does not.
- `nemo`: optional NeMo text normalization when installed. Recommended for numerically heavy text
  (dates, currency, ordinals), which `english-basic` does not attempt.
- `plugin`: custom `module:function` normalizer supplied by the user.

ASR and normalization choices affect WER/CER. Always expose normalized text and normalized
transcript when debugging outliers.

## Provenance and the Measurement Cache

Every `results.jsonl` row and `summary.json` carries an `evaluator` block: the package version, a
`code_hash` of the measurement modules (ASR backend, audio features, metrics, normalization), a
`plugin_hash` of a custom normalizer's module source, and the ASR and normalizer names. The
measurement cache key includes the same hashes together with the decoded audio, text, and
ASR/normalization/optional-metric config, so editing metric or normalizer code (or a plugin under an
unchanged name) forces recomputation instead of replaying stale scores. Thresholds and reporting are
never cached. Failed measurements (a NISQA exception, say) are not cached either, so a repaired
metric is retried on the next run.

`tts-assess compare` requires all runs to share the ASR backend and model, the normalizer, and the
measurement code hash. Results from an older evaluator should be re-evaluated (the cache makes this
cheap when only thresholds changed, and forces recomputation when measurement code changed).

The example runs under `out/eval_multi/` were produced by an earlier evaluator (a transcript-only
`repeated_span` rule and the previous silence, prolongation, and NISQA definitions). They document
the report layout; regenerate them with `scripts/eval_multi.py` before quoting their numbers.

## Report Design

The HTML reports are deliberately plain: black-and-white tables, system fonts, and best/worst cell
highlighting only. There are **no CDN assets or scripts**, so a report renders identically offline
and when the folder is copied elsewhere. See [Reports](#reports) for the sections.

The JSON artifacts (`report_data.json`, `comparison.json`) remain the canonical, machine-readable
form of every report.

## Limitations

- **WER/CER also capture ASR errors, not only TTS errors.** They score the `faster-whisper`
  transcript against the text, so the ASR's own mistakes — mishearing accents, proper names, and
  expressive delivery — inflate WER/CER even when the synthesis is correct. Read them as a joint
  TTS+ASR signal: check the `expected` vs `heard` columns in Threshold Violations, cross-reference
  NISQA (which needs no transcript), and confirm outliers by listening. A larger or multilingual
  Whisper model reduces, but does not remove, this bias.
- Automated metrics narrow review; they do not replace human listening tests.
- Whisper timestamps are coarse diagnostics, not forced alignment.
- Speaker similarity requires a reference voice and is backend-dependent.
- NISQAv2 and similar MOS predictors are useful proxies, not subjective MOS.
- Subjective dimensions such as preference, rhythm, expressiveness, intonation, and brand fit still
  need curated listening subsets.

### Related evaluation tools

Several projects found in our landscape review overlap with part of this workflow, but focus on a
different evaluation layer:

| Tool | Primary focus | How this toolkit differs |
|---|---|---|
| [`tts-bench`](https://github.com/5uck1ess/tts-bench) | Local TTS model speed, memory, samples, objective scores, and a listening arena | Accepts audio from any system through a simple manifest, adds provider sampling, configurable thresholds, per-sample diagnostics, and portable offline reports; it does not benchmark local inference speed. |
| [Promptfoo voice evaluation](https://www.promptfoo.dev/docs/guides/evaluate-elevenlabs/) | General application evaluation, including TTS latency, cost, round-trip transcription, and voice-agent tests | Specializes in clip-level TTS quality and intelligibility, with speech metrics, audio-health checks, a measurement cache, and a bundled stress dataset. |
| [ServiceNow EVA](https://github.com/ServiceNow/eva) | End-to-end, multi-turn voice-agent behavior and user experience | Isolates synthesized speech for reference-text and audio analysis rather than scoring task completion or the whole conversational agent. |
| [Pipecat STT Benchmark](https://github.com/pipecat-ai/stt-benchmark) | Streaming speech-to-text latency and semantic WER | Evaluates TTS output; ASR is used only as one instrument for measuring intelligibility. |
| [TTS Arena V2](https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2) and [Artificial Analysis](https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice) | Hosted human-preference comparisons and live leaderboards | Runs locally with fixed inputs and configuration, automated metrics, and inspectable per-sample output; it complements rather than replaces listening tests. |

## Possible Improvements

- Custom metrics
- Better voice control
- Multi-ASR backend
- Multilingual datasets and normalization
