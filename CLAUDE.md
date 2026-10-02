# CLAUDE.md

Full agent guide: **[AGENTS.md](AGENTS.md)** — read it for the sample / evaluate / report workflows,
config, metric glossary, repo map, and gotchas. Quick reference below.

## The three workflows
```bash
# 1. Sample audio from a provider (inworld | elevenlabs | hume | gradium)
tts-assess sample data/inworld.tts.open_benchmark.en.json -p inworld \
  --api-key-file ~/inworld.key --model inworld-tts-2 --voice Ashley --voice Sarah -o out/samples

# 2. Evaluate a manifest INTO ITS OWN DIR (keeps audio+report together, relative paths)
tts-assess run out/samples/inworld-inworld-tts-2/manifest.jsonl -c eval.yml -o out/samples/inworld-inworld-tts-2

# 3. Compare runs → black-and-white comparison.html
tts-assess compare out/samples/<runA> out/samples/<runB> --label A --label B -o out/comparison
```

## Rules
- **Never commit API keys or audio.** Audio (`*.wav`…), `tmp/`, venvs, `.measure_cache/` are
  git-ignored — keep it so. Keys via `--api-key-env`/`--api-key-file` only.
- Run `ruff check .` and `pytest -q` after changes (both must pass; line length 100).
- The measurement cache makes re-runs instant — don't re-synthesize/re-transcribe unnecessarily.
- `asr.backend: mock` for tests without Whisper. `[asr]` = Whisper, `[quality]` = NISQA.
- **WER/CER include ASR (Whisper) errors**, not only TTS errors — cross-check NISQA and listen.
- Reports are minimal B&W, offline (no CDN); `results.jsonl` is canonical. Details in AGENTS.md.
- Every row carries an `evaluator` stamp (code hash, ASR, normalizer); `compare` refuses runs
  scored by different evaluators. Re-evaluate old runs instead of mixing them. Unmeasured clips
  (decode/ASR/model errors) count as failures and show as `n=…` coverage in reports.
- `out/eval_multi/` results carry an `evaluator` stamp; if its `code_hash` differs from the current
  code, regenerate with `scripts/eval_multi.py` before quoting numbers.
