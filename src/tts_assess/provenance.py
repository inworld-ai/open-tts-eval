"""Evaluator identity: which code and settings produced a measurement.

Scores are only comparable when the same evaluator produced them. The
fingerprint below is stamped on every results row and folded into the
measurement cache key, so changed metric or normalizer code cannot silently
reuse stale scores, and a comparison can refuse runs scored differently.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
from functools import lru_cache
from pathlib import Path
from typing import Any

from tts_assess import __version__
from tts_assess.config import AssessmentConfig

# Modules whose source defines what a measurement means. Orchestration and
# reporting code is deliberately excluded: changing a table layout must not
# invalidate cached transcripts.
MEASUREMENT_MODULES: tuple[str, ...] = (
    "asr/backends.py",
    "audio/features.py",
    "evaluate.py",
    "metrics/audio.py",
    "metrics/optional.py",
    "metrics/text.py",
    "normalization/backends.py",
    "normalization/english.py",
)

_HASH_LENGTH = 16


def evaluator_fingerprint(config: AssessmentConfig) -> dict[str, Any]:
    """Identity of the evaluator that ``config`` selects, as stored on each row."""
    return {
        "version": __version__,
        "code_hash": measurement_code_hash(),
        "plugin_hash": plugin_hash(config),
        "asr": f"{config.asr.backend}:{config.asr.model}",
        "normalization": describe_normalization(config),
    }


@lru_cache(maxsize=1)
def measurement_code_hash() -> str:
    """Hash of the measurement modules' source as installed."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for relative in MEASUREMENT_MODULES:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.exists() else b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()[:_HASH_LENGTH]


def plugin_hash(config: AssessmentConfig) -> str | None:
    """Hash of a normalization plugin's module source; None when no plugin is used.

    The whole module is hashed (not just the entry function) because helpers it
    calls change its behaviour too. ``unavailable`` means the source could not be
    read, which the comparison treats as a value to match, not as "unknown".
    """
    if config.normalization.backend != "plugin" or not config.normalization.plugin:
        return None
    module_name, _, attr = config.normalization.plugin.partition(":")
    source: bytes | None = None
    try:
        module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file:
            source = Path(module_file).read_bytes()
        else:
            source = inspect.getsource(getattr(module, attr)).encode("utf-8")
    except Exception:
        source = None
    if source is None:
        return "unavailable"
    return hashlib.sha256(source).hexdigest()[:_HASH_LENGTH]


def describe_normalization(config: AssessmentConfig) -> str:
    normalization = config.normalization
    if normalization.backend == "plugin":
        return f"plugin:{normalization.plugin}"
    if normalization.backend == "english-basic" and not normalization.expand_numbers:
        return "english-basic(no-number-expansion)"
    return normalization.backend
