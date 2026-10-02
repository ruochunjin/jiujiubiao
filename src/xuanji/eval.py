"""Evaluation metrics and dataset helpers (WTQ / TabFact style).

Dataset format (JSON array)::

    [{"id": "...", "table_caption": "...", "table_text": [[header], [row]...],
      "question": "...", "label": "..."}]

TabFact-style files use ``statement`` instead of ``question`` and a boolean
label.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from xuanji.normalize import normalize


def denotation_accuracy(predictions: Sequence[str], labels: Sequence[Any], task: str = "tqa") -> float:
    """Fraction of predictions whose normalized form matches the gold label."""

    if not predictions:
        return 0.0
    hits = 0
    for pred, gold in zip(predictions, labels):
        gold_norms = gold if isinstance(gold, list) else [gold]
        norm = normalize(pred, "tfv" if task == "tfv" else "tqa")
        if any(norm == normalize(g, "tfv" if task == "tfv" else "tqa") for g in gold_norms):
            hits += 1
    return hits / len(predictions)


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load a WTQ/TabFact-style JSON dataset."""

    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("dataset must be a JSON array of examples")
    return data


__all__ = ["denotation_accuracy", "load_dataset"]
