"""Benchmark loaders with a local table cache.

WTQ comes from the TableSenseAI parquet mirror (the Stanford original needs a
legacy dataset script); table CSVs are fetched per-row from the same repo and
cached under ``benchmarks/data/cache/``.  Set ``HF_ENDPOINT`` for mirrors.
"""

from __future__ import annotations

import csv
import io
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "data" / "cache"

_WTQ_DATASET = "TableSenseAI/WikiTableQuestions"
_WTQ_RESOLVE = f"https://hf-mirror.com/datasets/{_WTQ_DATASET}/resolve/main"


@dataclass(frozen=True)
class BenchmarkItem:
    """One benchmark question over one table."""

    id: str
    question: str
    answers: tuple[str, ...]
    table_path: str  # repo-relative or cached local path
    caption: str = ""


@dataclass(frozen=True)
class BenchmarkTable:
    """A loaded table: header + string rows (WTQ CSV convention)."""

    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    def to_rows(self) -> list[list[str]]:
        return [list(self.header)] + [list(r) for r in self.rows]


def _fetch(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        dest.write_bytes(response.read())
    return dest


def _fetch_hf_file(repo_path: str, dest: Path) -> Path:
    """Download one file from the WTQ mirror repo via huggingface_hub.

    Plain urllib hits 403 on the mirror's CDN redirect; hf_hub_download
    handles the auth/redirect dance correctly (and caches locally itself,
    but we also copy into our own cache so paths stay explicit).
    """

    if dest.exists():
        return dest
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(_WTQ_DATASET, repo_path, repo_type="dataset")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(Path(downloaded).read_bytes())
    return dest


def load_wtq_table(context_csv: str) -> BenchmarkTable:
    """Load one WTQ table by its repo-relative csv path (cached on disk)."""

    local = _fetch_hf_file(context_csv, CACHE_DIR / "wtq" / context_csv.replace("/", "_"))
    # WTQ CSVs are mixed-encoding (cp1252 en-dashes etc.); utf-8-replace
    # mojibake corrupts answers (nu-3088: "8–14" became "8C14").
    raw = local.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = [tuple(cell.strip() for cell in row) for row in reader if row]
    if not rows:
        raise ValueError(f"empty table: {context_csv}")
    return BenchmarkTable(tuple(rows[0]), tuple(rows[1:]))


def load_wtq(
    limit: int = 100, *, seed: int = 20261009, split: str = "test"
) -> list[BenchmarkItem]:
    """WTQ items, stratified by table so no table dominates the sample."""

    from datasets import load_dataset

    ds = load_dataset(_WTQ_DATASET, split=split)
    rng = random.Random(seed)
    by_table: dict[str, list[int]] = {}
    for index, row in enumerate(ds):
        by_table.setdefault(row["context"]["csv"], []).append(index)
    tables = sorted(by_table)
    rng.shuffle(tables)
    picked: list[int] = []
    per_table = 1  # strict stratification first, then top-up
    for table in tables:
        if len(picked) >= limit:
            break
        indices = by_table[table]
        rng.shuffle(indices)
        picked.extend(indices[:per_table])
    for table in tables:  # top-up for small limits absorbed already
        if len(picked) >= limit:
            break
        indices = [i for i in by_table[table] if i not in picked]
        rng.shuffle(indices)
        picked.extend(indices)
    picked = picked[:limit]

    items: list[BenchmarkItem] = []
    for index in picked:
        row = ds[index]
        answers = tuple(
            part.strip()
            for part in str(row["target_value"]).split("|")
            if part.strip()
        )
        items.append(
            BenchmarkItem(
                id=row["id"],
                question=str(row["utterance"]),
                answers=answers,
                table_path=row["context"]["csv"],
            )
        )
    return items


def load_tabfact(
    limit: int = 100, *, seed: int = 20261009, split: str = "validation"
) -> tuple[list[BenchmarkItem], dict[str, BenchmarkTable]]:
    """TabFact via the target-benchmark mirrors (queries + inline corpus).

    Returns (items, table_map): queries reference tables by ``table_id``;
    the corpus is loaded once and kept as a map so the runner can reuse it.
    """

    from datasets import load_dataset

    queries = load_dataset("target-benchmark/tabfact-queries", split=split)
    table_map: dict[str, BenchmarkTable] = {}
    for corpus_split in ("train", "validation", "test"):
        corpus = load_dataset("target-benchmark/tabfact-corpus", split=corpus_split)
        for row in corpus:
            table_id = row["table_id"]
            if table_id in table_map:
                continue
            rows = [tuple(str(c).strip() for c in r) for r in row["table"] if r]
            if rows:
                table_map[table_id] = BenchmarkTable(tuple(rows[0]), tuple(rows[1:]))

    rng = random.Random(seed)
    indices = list(range(len(queries)))
    rng.shuffle(indices)
    items: list[BenchmarkItem] = []
    for index in indices:
        row = queries[index]
        if row["table_id"] not in table_map:
            continue
        answer = row["answer"]
        # The mirror stores the label as a STRING "True"/"False" — a bare
        # truthiness check marks everything True (TabFact-100 artifact).
        answer_bool = answer if isinstance(answer, bool) else str(answer) == "True"
        items.append(
            BenchmarkItem(
                id=str(row["query_id"]),
                question=str(row["query"]),
                answers=("True" if answer_bool else "False",),
                table_path=row["table_id"],
            )
        )
        if len(items) >= limit:
            break
    return items, table_map


__all__ = [
    "BenchmarkItem",
    "BenchmarkTable",
    "load_tabfact",
    "load_wtq",
    "load_wtq_table",
]
