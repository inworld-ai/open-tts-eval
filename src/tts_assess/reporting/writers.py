from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    # allow_nan=False: NaN/Infinity are not JSON and would silently corrupt the
    # canonical record. Upstream guards keep them out; this makes a slip loud.
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str, ensure_ascii=False, allow_nan=False) + "\n")


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(summary, indent=2, default=str, ensure_ascii=False, allow_nan=False)
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, list | dict):
        return json.dumps(value, ensure_ascii=False)
    return value
