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

from jiujiubiao.normalize import normalize


def _norm_parts(value: Any, task: str) -> frozenset:
    """Normalized answer as a SET of parts (multi-answer " | " splits)."""

    norm = normalize(value, "tfv" if task == "tfv" else "tqa")
    return frozenset(part for part in norm.split(" | ") if part)


def denotation_accuracy(predictions: Sequence[str], labels: Sequence[Any], task: str = "tqa") -> float:
    """Fraction of predictions matching the gold label after normalization.

    Multi-answer values compare as SETS (WTQ official semantics): part order
    and case are irrelevant, but extra/missing parts miss
    (WTQ-50 nu-3074: "surround stakes | warwick stakes" must match the
    capitalized gold).
    """

    if not predictions:
        return 0.0
    hits = 0
    for pred, gold in zip(predictions, labels):
        gold_norms = gold if isinstance(gold, list) else [gold]
        pred_parts = _norm_parts(pred, task)
        # The gold label's entries TOGETHER form the reference set (WTQ
        # multi-part answers arrive as several entries); a single-part
        # prediction may match one entry of an alternatives set.
        gold_parts = frozenset().union(*(_norm_parts(g, task) for g in gold_norms))
        if pred_parts == gold_parts or (len(pred_parts) == 1 and pred_parts <= gold_parts):
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
