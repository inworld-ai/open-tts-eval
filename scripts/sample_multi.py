"""Self-sufficient driver to sample all 6 benchmark models.

Reproduces exactly the runs under ``out/multi/`` for the bundled dataset
``data/inworld.tts.open_benchmark.en.json``:

    Inworld     : inworld-tts-2, inworld-tts-1.5-max, inworld-tts-1-max
    ElevenLabs  : eleven_v3, eleven_multilingual_v2
    Hume        : octave-2

Voices are pinned (not re-sampled from the live catalog) so runs are
deterministic and hit the audio cache even as provider catalogs drift. Audio is
cached, so re-running resumes; a provider with no key or an exhausted quota is
logged and skipped.

Keys are read from an environment variable, falling back to a key file. Set the
env vars (recommended) or point the file paths at your keys:

    INWORLD_API_KEY      or ~/alt.inworld.key
    ELEVENLABS_API_KEY   or ~/11labs_key.txt
    HUME_API_KEY         or ~/hume_key.txt

Run from the repo root:  python scripts/sample_multi.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from tts_assess.sampling.providers import build_provider
from tts_assess.sampling.sampler import SamplingConfig, run_sampling

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "data" / "inworld.tts.open_benchmark.en.json"
# Sample straight into the eval tree so each run dir ends up self-contained
# (audio + manifest + results + report together, with relative audio paths).
OUT = REPO / "out" / "eval_multi"

# 10 diverse English voices per provider, pinned from the benchmark runs.
INWORLD_VOICES = [
    "Oliver", "Rosalind", "Avery", "Jarrah", "Victor",
    "Timothy", "Reed", "Lauren", "Tessa", "Miranda",
]
# ElevenLabs prebuilt English voice IDs (names vary; IDs are stable).
ELEVENLABS_VOICES = [
    "lripbtloD4ja6bwAMrej", "9LQRvqwzjElMJrr5cC7j", "XrExE9yKIg1WjnnlVkGX",
    "ENvAQOIl5783LQCEY3Xy", "MCs02AmHQeN2qsH0EqSW", "CDKaxr6sKMXHhQc4FPlW",
    "2GErPEnQqbyeqhZQPM6r", "fnX380ktr2DhFHZjwtCX", "kVp3G6YINMUwOL7ROfIF",
    "r5iFzIytiA1rzjhWFCjW",
]
HUME_VOICES = [
    "Casual Podcast Host", "Anna", "English Casual Conversationalist",
    "Ghost With Unfinished Business", "Turtle Guru", "Welsh Folk Storyteller",
    "Friendly Kiwi Girl", "Demure Conversationalist", "Live Comedian",
    "Friendly Kiwi Guy",
]


@dataclass
class Plan:
    provider: str
    models: list                # latest first; str id or (api_id, label) alias
    voices: list[str]
    env: str                    # env var holding the API key
    key_file: Path              # fallback file if env var is unset
    concurrency: int = 6
    provider_kwargs: dict = field(default_factory=dict)


PLANS = [
    Plan(
        # inworld-tts-2 is the API id; it is presented as "inworld-tts-2-preview".
        provider="inworld",
        models=[("inworld-tts-2", "inworld-tts-2-preview"), "inworld-tts-1.5-max", "inworld-tts-1-max"],
        voices=INWORLD_VOICES,
        env="INWORLD_API_KEY",
        key_file=Path.home() / "alt.inworld.key",
        concurrency=8,
    ),
    Plan(
        provider="elevenlabs",
        models=["eleven_v3", "eleven_multilingual_v2"],
        voices=ELEVENLABS_VOICES,
        env="ELEVENLABS_API_KEY",
        key_file=Path.home() / "11labs_key.txt",
        concurrency=6,
        provider_kwargs={"max_retries": 1},   # fail fast on a character-quota cap
    ),
    Plan(
        provider="hume",
        models=["octave-2"],
        voices=HUME_VOICES,
        env="HUME_API_KEY",
        key_file=Path.home() / "hume_key.txt",
        concurrency=4,
        provider_kwargs={"max_retries": 1},   # fail fast on an exhausted quota
    ),
]


def resolve_key(plan: Plan) -> str | None:
    key = os.environ.get(plan.env)
    if key:
        return key.strip()
    if plan.key_file.exists():
        return plan.key_file.read_text(encoding="utf-8").strip()
    return None


def main() -> None:
    if not DATASET.exists():
        sys.exit(f"dataset not found: {DATASET}")
    for plan in PLANS:
        print(f"\n===== {plan.provider} ({', '.join(plan.models)}) =====", flush=True)
        key = resolve_key(plan)
        if not key:
            print(
                f"  SKIPPED {plan.provider}: no key (set ${plan.env} or {plan.key_file})",
                file=sys.stderr,
                flush=True,
            )
            continue
        try:
            provider = build_provider(plan.provider, key, **plan.provider_kwargs)
            config = SamplingConfig(
                language="en-US",
                audio_encoding="WAV",
                sample_rate_hz=24000,
                concurrency=plan.concurrency,
            )
            summary = run_sampling(
                DATASET, OUT, provider, plan.models, voices=plan.voices, config=config
            )
            for run in summary["runs"]:
                print(
                    f"  {run['model']}: {run['samples']} ok, {run['errors']} errors",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 - keep going to the next provider
            print(f"  SKIPPED {plan.provider}: {str(exc)[:200]}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
