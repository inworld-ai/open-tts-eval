# Agent guide — sample, evaluate, and report TTS with `tts-assess`

This is the operating manual for AI coding agents (Codex, Claude Code, Cursor, …) working in this
repo. It covers the three workflows the toolkit exists for: **sample** audio from TTS providers,
**evaluate** it against the reference text, and build **reports** (single-run and cross-run).

Golden rules:
- **Never commit API keys or audio.** Keys come from env vars / key files; audio (`*.wav`,
  `*.flac`, `*.mp3`, …) and `tmp/`, virtualenvs, and `.measure_cache/` are git-ignored — keep it
  that way.
- The **evaluation core is offline**; only the *sampling* subsystem calls external provider APIs.
- After code changes, run `ruff check .` and `pytest -q` (both must pass).
- `results.jsonl` is the canonical per-sample record; the HTML reports are derived from it.

---

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[asr,quality]"     # asr = faster-whisper; quality = torch + NISQA (torchmetrics)
```

Install extras by need: core (`pip install -e .`) is enough for **sampling** and **mock-ASR**
tests; `[asr]` adds real Whisper; `[quality]` adds NISQAv2; `[similarity]` adds ECAPA speaker
similarity; `[all]` = everything.

The CLI entry point is `tts-assess` (module `tts_assess.cli`). Commands: `sample`, `run`,
`compare`, `voices`, `init-config`, `preview`.

---

## Providers & API keys

| provider | key env (default) | default model | notes |
|---|---|---|---|
| `inworld` | `INWORLD_API_KEY` | `inworld-tts-1.5-max` | also `inworld-tts-2`, `inworld-tts-1-max`, `inworld-tts-1` |
| `elevenlabs` | `ELEVENLABS_API_KEY` | `eleven_multilingual_v2` | `eleven_v3`, `eleven_turbo_v2_5` |
| `hume` | `HUME_API_KEY` | `octave-2` | `octave-1`; sits behind Cloudflare (a UA is set for you) |

Provide the key with `--api-key-env NAME` (reads that env var) or `--api-key-file PATH`. List a
provider's prebuilt voices:

```bash
tts-assess voices --provider inworld --api-key-env INWORLD_API_KEY
```

---

## Workflow 1 — Sample

Synthesize a text dataset with chosen voices/models. One run directory per model, named
`<provider>-<model>`, each holding `audio/`, `manifest.jsonl`, `sampling_meta.json`.

```bash
tts-assess sample data/inworld.tts.open_benchmak.en.json \
  --provider inworld --api-key-file ~/inworld.key \
  --model inworld-tts-2 --model inworld-tts-1.5-max \
  --voice Ashley --voice Sarah --voice Oliver \
  --language en-US --format WAV --sample-rate 24000 \
  --limit 20 -o out/samples
```

Key flags: `--provider/-p`, `--model` (repeatable, latest first), `--voice/-v` (repeatable) **or**
`--num-voices N` (+ `--shuffle-voices --seed S` for a seeded diverse pick), `--format`
(WAV/LINEAR16/MP3/FLAC/OGG_OPUS/…; **WAV is the safe default** — `soundfile` must decode it),
`--sample-rate`, `--limit N` (first N texts, for smoke runs), `--concurrency`, `--overwrite`,
`--language`, `--speaking-rate`, `--temperature`, `--allow-model-mismatch` (keep clips the provider
reports as served by another model; by default they are errors).

- Controls are only accepted where the adapter transmits them: Inworld takes speaking rate and
  temperature; ElevenLabs takes speaking rate (`voice_settings.speed`) and `language_code` for
  language-enforcing models only; Hume takes speaking rate (utterance `speed`) and always pins
  `version`. Anything else raises `ProviderError` instead of being silently dropped.
- Each manifest row's `metadata` records `api_model` / `returned_model`, `sent_request` (body minus
  text), probed `sample_rate_hz` / `channels` / `duration_sec` plus `requested_sample_rate_hz`, and a
  `sampling_fingerprint`. Ids that sanitize to the same file name get a hash suffix (no overwrites).

- Datasets: `.txt` (one utterance/line), `.json`/`.jsonl` (objects or bare strings with a `text`
  field; optional `id`, `language`), or `.csv` with a `text` column. Bundled benchmark:
  `data/inworld.tts.open_benchmak.en.json` (100 messy dialogue utterances).
- Audio is **cached by existence** — re-running resumes; dead keys / quota caps are logged per
  sample and skipped (the run continues). Here "cached by existence" also requires a non-empty
  file and a matching full-request fingerprint; older unfingerprinted rows are refreshed once.
- Voices become `speaker_id` in the manifest.

For a reproducible example of a multi-provider run, use `scripts/sample_multi.py` (pinned voices/models; keys
from env or file; presents `inworld-tts-2` as `inworld-tts-2-preview` via a model alias).
Build your own sampling script to support the new providers, or just provide already sampled audios.


## Workflow 2 — Evaluate

Assess a manifest of `{id, text, audio_path}` (the sampler writes these). Runs ASR → normalize →
WER/CER + audio-health + optional NISQA/speaker/prosody → thresholds.

```bash
tts-assess run out/samples/inworld-inworld-tts-2/manifest.jsonl \
  --config eval.yml -o out/samples/inworld-inworld-tts-2
```

- **Evaluate a run into its own dir** (as above): audio, manifest, results, and report end up
  together, and `audio_path` is stored **relative** → the report's `<audio>` players work when the
  folder is opened or moved. Evaluating into a *different* dir keeps absolute paths.
- **Measurement cache** (`.measure_cache/`, keyed by decoded audio + sample rate/channels + text +
  reference-audio content when used + ASR/normalization/metric config): re-runs are instant
  (`--no-cache` / `--cache-dir` to control). Only thresholds and the report are re-applied each run,
  so tuning bands or the report needs **no** recompute.
- Use `asr.backend: mock` (config) for plumbing tests without downloading Whisper.
- Outputs per run: `results.jsonl`, `summary.json`, `results.csv`, `report_data.json`,
  `report.html`.

## Workflow 3 — Reports & Compare

Every `run` already writes a single-run `report.html`. To compare runs:

```bash
tts-assess compare out/samples/inworld-inworld-tts-2 out/samples/inworld-inworld-tts-1.5-max \
  --label "TTS 2" --label "TTS 1.5 Max" --config eval.yml -o out/comparison
```

Positional args are run dirs (or `results.jsonl` paths); `--label` (repeatable, one per run) sets
column names. Runs must have matching text/language cohorts and equivalent per-speaker sampling
profiles, although provider-specific sample and voice IDs may differ. Runs must also have been
scored by the same evaluator (`asr_backend`/`asr_model` on rows, and the `evaluator.code_hash` /
`plugin_hash` / `normalization` stamp); runs without the stamp only warn. Rows whose measurement
failed stay failed when thresholds are re-applied. Writes `comparison.html` + `comparison.json`.

Both reports share one **minimal black-and-white** layout (no CDN/scripts, fully offline):
- **Metric Comparison** — metrics grouped **Accuracy** (WER, insertion/deletion/substitution, CER),
  **NISQAv2** (MOS + noisiness/discontinuity/coloration/loudness), **Subjective** (chars/sec,
  arousal, expressiveness), **Silence** (silence ratio, lead/tail silence). Each cell = mean with a
  95% **cluster bootstrap over texts** CI beneath (clips sharing a text are resampled together);
  best run per metric green, worst red, `*` when the paired per-text difference to the best run
  excludes zero; Silence is uncolored. `n=measured/total · k failed` appears when a cell covers
  fewer clips than the run has.
- **Model Health** — per threshold-backed metric, the share of clips passing its per-sample
  threshold (a `pass if …` rule), graded good ≥99% / warn ≥95% / fail <95% (configurable). Failed
  measurements count as not passing. WER/CER/insertions are omitted here; the hallucination flags
  (`empty_transcript`, `repeated_span`, `tail_hallucination`, `tail_click_detected`) are rows.
- **Threshold Violations** (single-run report only) — warn/fail counts per metric and per failed
  measurement stage (audio/ASR/NISQA/…), worst examples showing normalized `expected` vs `heard`
  (or the error text), an `<audio>` player if the clip exists, each sample once.

---

## Config

`tts-assess init-config eval.yml` writes an editable default. Shape:

```yaml
asr:
  backend: faster-whisper   # or "mock"
  model: base.en            # small, medium, large-v3, …
  language: en
normalization:
  backend: english-basic    # or "nemo", or "plugin" (module:function)
  expand_numbers: true
optional_metrics:
  nisqa_v2: false           # needs [quality]
  speaker_similarity: false # needs [similarity] + reference_audio_path
  vowel_prolongation: true
  voice_lens: true          # expressiveness/arousal proxies
thresholds:                 # per-sample pass/warn/fail bands (see below)
  wer: {warn: 0.05, fail: 0.10}
  vowel_prolongation_score: {fail: 0.8}
  nisqa_mos: {warn_below: 3.3, fail_below: 2.8}
  repeated_span: {fail_if_true: true}   # also empty_transcript, tail_hallucination, tail_click_detected
reporting:
  confidence_level: 0.95
  bootstrap_resamples: 2000
  health_good_rate: 0.99    # Model Health bands
  health_warn_rate: 0.95
  embed_audio: true
  max_violation_examples: 5
  max_worst_samples: 25
  question: "Are these TTS audios acceptable?"
```

Unknown config fields are rejected instead of being silently ignored.

`BoundThreshold` fields: `warn`/`fail` (upper-bound: value ≥ band → warn/fail), `warn_below`/
`fail_below` (lower-bound, e.g. similarity/MOS), `fail_if_true` (booleans like `tail_click_detected`).

## Metric glossary (direction = better)

- **Accuracy (lower better):** `wer`, `cer`, `insertion_rate`, `deletion_rate`, `substitution_rate`.
- **NISQAv2 (higher better, 1–5):** `nisqa_mos` + `nisqa_noisiness/discontinuity/coloration/loudness`.
- **Audio health (lower better):** `clipping_ratio`, `silence_ratio`, `leading/trailing_silence_sec`
  (frame RMS below `max(min(1.5·p20, 0.1·p95), −80 dBFS)`, percentiles over the span between the
  first and last non-silent frame, so padding only adds silence), `tail_click_score` (largest
  sample jump in the last 30 ms / RMS of the preceding 200 ms, **computed at 24 kHz** whatever the
  file's rate; boolean `tail_click_detected` at score ≥ 4.0).
- **Pacing/subjective (neutral):** `duration_sec`, `chars_per_second`, `arousal_proxy`;
  `expressiveness_proxy`; `vowel_prolongation_score` (seconds of the longest loud low-centroid run;
  the loudness cut comes from non-silent frames and is capped at half the speech level so a longer
  vowel never scores shorter; the default threshold fails values at or above 0.8s).
- **Speaker (higher better):** `speaker_similarity` (needs `reference_audio_path`).
- **NISQA** runs on the clip at its native rate (not downsampled to 16 kHz); the model instance is
  cached per rate.
- **Hallucination flags (boolean, fail if true):** `empty_transcript`, `repeated_span` (an adjacent
  repeat in the transcript that the reference does not contain), `tail_hallucination`.
- A NaN/Inf value fails its threshold as `invalid:<metric>`; a NaN sample fails the clip at decode.

## Repo map

```
src/tts_assess/
  cli.py                       # Typer CLI: run / sample / compare / voices / init-config / preview
  config.py                    # pydantic config + default thresholds
  pipeline.py                  # run_assessment over a manifest: cache, thresholds, reports
  evaluate.py                  # measure_pair / evaluate_pair: per-pair core (bytes or arrays), shared by pipeline + services
  provenance.py                # evaluator fingerprint (version, measurement code hash, plugin hash)
  asr/backends.py              # faster-whisper + mock (model cached via lru_cache)
  audio/features.py            # decode once; rms/peak/clipping/silence/tail-click
  normalization/               # english-basic (+ nemo, plugin)
  metrics/                     # text (jiwer WER/CER + hallucination heuristics), audio, optional (NISQA/ECAPA/prosody)
  reporting/
    aggregate.py               # summarize: means, cluster-bootstrap CI, coverage, violations, by-voice
    stats.py                   # iid / cluster bootstrap CI + paired per-text difference CI
    metrics_meta.py            # metric direction + display names
    thresholds.py              # classify_row / evaluate_thresholds
    compare.py                 # build_comparison + build_run_report (+ Model Health, violations)
    comparison_html.py         # the black-and-white renderer (both reports)
  sampling/
    datasets.py                # .txt/.json/.jsonl/.csv loaders
    sampler.py                 # run_sampling: texts × voices × models → manifests (+ model aliases)
    providers/                 # base.py + inworld.py / elevenlabs.py / hume.py (+ _http.py)
scripts/sample_multi.py        # reproducible multi-provider sampling
scripts/eval_multi.py          # evaluate all runs in place + build compare_all
data/…open_benchmak.en.json    # bundled benchmark dataset
out/eval_multi/                # example evaluated runs + compare_all report (audio git-ignored);
                               # stamped with their evaluator — regenerate when measurement code changes
```

## Gotchas

- **WER/CER also capture ASR errors, not just TTS errors.** Whisper mishears accents, proper names,
  and expressive delivery, inflating WER even when synthesis is fine. Cross-check NISQA (needs no
  transcript), read the `expected` vs `heard` columns, and confirm by listening. Bigger/multilingual
  Whisper reduces it.
- Committed reports reference audio by **relative** path; audio isn't shipped, so per-run `<audio>`
  players are inert in a fresh checkout. The **comparison report has no audio** and always works.
- `tail_click_score` is an unbounded ratio (heavy-tailed): its mean is outlier-dominated — prefer
  the boolean `tail_click_detected` or a clipped/percentile view.
- Competitor keys may be dead or quota-limited; the sampler records per-sample errors and continues.
  Use `--limit` for cheap smoke runs.
- The measurement cache key includes a hash of the measurement modules and of any normalizer
  plugin's source (`evaluator.code_hash` / `plugin_hash`), plus `_MEASUREMENT_CACHE_VERSION` in
  `pipeline.py`. Editing metric or normalizer code therefore recomputes everything (ASR included);
  editing reporting code does not. Failed optional metrics are never cached.
- `compare` refuses runs scored by different ASR/normalizer/measurement code. Old runs without an
  `evaluator` stamp produce a warning; re-evaluate them rather than reasoning about mixed scores.
- **Whisper on CPU is not bit-reproducible.** The decoder is seeded per clip (removes the
  temperature-fallback sampling randomness), but float reduction order can still flip a near-tie
  beam decision: expect a few tail hallucinations ("The The The…", a repeated sentence) to appear or
  vanish per thousand clips between identical runs, each moving one clip's pass/fail. Quote pass
  counts with their CIs, and rely on the measurement cache (not re-transcription) for stability.
- The checked-in `out/eval_multi/` results are only valid for the code that produced them: compare
  `summary.json["evaluator"]["code_hash"]` with `tts_assess.provenance.measurement_code_hash()`
  before citing them, and regenerate with `scripts/eval_multi.py` (needs the audio and the
  `[asr,quality]` extras) after any change to measurement code.

## Extending — add a provider

1. Subclass `TTSProvider` in `src/tts_assess/sampling/providers/<name>.py`, implementing
   `synthesize(SynthesisRequest) -> SynthesisResult` and `list_voices() -> list[Voice]` (use the
   injectable `transport` and the `providers/_http` helpers; set `forced_encoding` if the API only
   emits one decodable format).
2. Register it in `providers/__init__.py` (`_PROVIDERS`).
3. Add tests with a fake transport (see `tests/test_sampling_competitors.py`).

## Dev conventions

- `ruff check .` and `pytest -q` must pass; line length 100.
- Prefer the measurement cache over re-synthesizing/re-transcribing.
- Match existing style; keep audio and secrets out of git.
