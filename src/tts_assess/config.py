from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Status = Literal["pass", "warn", "fail", "skip"]


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BoundThreshold(StrictConfigModel):
    warn: float | None = None
    fail: float | None = None
    warn_below: float | None = None
    fail_below: float | None = None
    fail_if_true: bool | None = None


class ASRConfig(StrictConfigModel):
    backend: Literal["faster-whisper", "mock"] = "faster-whisper"
    model: str = "small"
    device: str = "auto"
    compute_type: str = "default"
    language: str | None = "en"


class NormalizationConfig(StrictConfigModel):
    backend: Literal["english-basic", "nemo", "plugin"] = "english-basic"
    plugin: str | None = None
    expand_numbers: bool = True


class OptionalMetricsConfig(StrictConfigModel):
    nisqa_v2: bool = False
    speaker_similarity: bool = False
    vowel_prolongation: bool = False
    voice_lens: bool = False
    open_aligner: bool = False


class ReportingConfig(StrictConfigModel):
    html: bool = True
    csv: bool = True
    max_worst_samples: int = 25
    max_violation_examples: int = 5
    confidence_level: float = 0.95
    bootstrap_resamples: int = 2000
    group_by_voice: bool = True
    embed_audio: bool = True
    # Comparison "model health": a model is good/warn/fail on a metric when its
    # per-sample pass-rate for that metric clears these bands.
    health_good_rate: float = 0.99
    health_warn_rate: float = 0.95
    title: str = "TTS Assessment Report"
    question: str = "Are these TTS audios acceptable for the configured quality thresholds?"


class AssessmentConfig(StrictConfigModel):
    asr: ASRConfig = Field(default_factory=ASRConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    optional_metrics: OptionalMetricsConfig = Field(default_factory=OptionalMetricsConfig)
    thresholds: dict[str, BoundThreshold] = Field(default_factory=dict)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    @classmethod
    def default(cls) -> AssessmentConfig:
        return cls(
            thresholds={
                "wer": BoundThreshold(warn=0.05, fail=0.10),
                "cer": BoundThreshold(warn=0.03, fail=0.08),
                "insertion_rate": BoundThreshold(warn=0.03, fail=0.08),
                # Heuristic flags are hard failures; listing them here makes them
                # visible in the health table and violations chapter like any metric.
                "empty_transcript": BoundThreshold(fail_if_true=True),
                "tail_hallucination": BoundThreshold(fail_if_true=True),
                "repeated_span": BoundThreshold(fail_if_true=True),
                "tail_click_detected": BoundThreshold(fail_if_true=True),
                "clipping_ratio": BoundThreshold(warn=0.001, fail=0.01),
                "silence_ratio": BoundThreshold(warn=0.45, fail=0.65),
                "vowel_prolongation_score": BoundThreshold(fail=0.8),
                "speaker_similarity": BoundThreshold(warn_below=0.65, fail_below=0.55),
                "nisqa_mos": BoundThreshold(warn_below=3.3, fail_below=2.8),
            }
        )


def load_config(path: Path | None) -> AssessmentConfig:
    base = AssessmentConfig.default()
    if path is None:
        return base
    raw = yaml.safe_load(path.read_text()) or {}
    merged = base.model_dump()
    _deep_update(merged, raw)
    return AssessmentConfig.model_validate(merged)


def write_default_config(path: Path) -> None:
    path.write_text(yaml.safe_dump(AssessmentConfig.default().model_dump(), sort_keys=False))


def _deep_update(target: dict, updates: dict) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
