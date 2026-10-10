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
    """Load one WTQ table by its repo-relative csv path (cached on disk).

    Prefers the ``.tsv`` variant: the mirror's CSVs have mojibake baked in
    upstream ("8–14" → "8C14"), while the TSVs are clean UTF-8.
    """

    tsv_path = context_csv.rsplit(".", 1)[0] + ".tsv"
    local = _fetch_hf_file(tsv_path, CACHE_DIR / "wtq" / tsv_path.replace("/", "_"))
    text = local.read_text(encoding="utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    rows = [tuple(cell.strip() for cell in row) for row in reader if row]
    if not rows:
        raise ValueError(f"empty table: {tsv_path}")
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
    "SqlItem",
    "bird_db_path",
    "clambsql_db_path",
    "load_bird",
    "load_clambsql",
    "load_tabfact",
    "load_wtq",
    "load_wtq_table",
]


_CLAMBSQL_DIR = Path(__file__).parent / "data" / "clambsql"


@dataclass(frozen=True)
class SqlItem:
    """One Text-to-SQL benchmark question over a sqlite database."""

    id: str
    question: str
    db_file: str
    gold_sql: str


def load_clambsql(limit: int = 60, *, seed: int = 20261009) -> list[SqlItem]:
    """CLAMBSQL (CLEAR benchmark): 7 ambiguity types x 200, vendored locally."""

    import json

    data = json.loads((_CLAMBSQL_DIR / "clambsql.json").read_text(encoding="utf-8"))
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for row in data:
        by_type.setdefault(row["ambig_type"], []).append(row)
    items: list[SqlItem] = []
    per_type = max(1, limit // len(by_type))
    for ambig_type, rows in sorted(by_type.items()):
        rng.shuffle(rows)
        for row in rows[:per_type]:
            items.append(
                SqlItem(
                    id=f"{row['ambig_type']}-{row['index']}",
                    question=row["question"],
                    db_file=row["db_file"],
                    gold_sql=row["gold_query"],
                )
            )
    return items[:limit]


def clambsql_db_path(db_file: str) -> str:
    return str(_CLAMBSQL_DIR / "database" / db_file)


_BIRD_DIR = Path(__file__).parent / "data" / "bird"
_BIRD_DATASET = "prem-research/birdbench"


_BIRD_MAX_DB_BYTES = 300 * 1024 * 1024  # skip multi-GB databases in benchmarks
_BIRD_SIZE_CACHE = _BIRD_DIR / "db_sizes.json"


def _bird_db_sizes() -> dict[str, int]:
    """Database size map from the repo tree API (cached): multi-GB databases
    (bike_share_1 is 5.2GB) would stall every benchmark run on download."""

    import json
    import urllib.request

    if _BIRD_SIZE_CACHE.exists():
        return json.loads(_BIRD_SIZE_CACHE.read_text(encoding="utf-8"))
    sizes: dict[str, int] = {}

    def _tree(subpath: str):
        url = (
            "https://hf-mirror.com/api/datasets/"
            f"{_BIRD_DATASET}/tree/main/{subpath}?recursive=false"
        )
        # the mirror rejects the bare python-urllib User-Agent with 403
        request = urllib.request.Request(url, headers={"User-Agent": "xuanji-benchmarks"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)

    for entry in _tree("train/train_databases"):
        if entry["type"] != "directory":
            continue
        db_id = entry["path"].rsplit("/", 1)[-1]
        best = 0
        for f in _tree(entry["path"]):
            if f["path"].endswith(".sqlite"):
                best = max(best, f.get("size") or f.get("lfs", {}).get("size") or 0)
        sizes[db_id] = best
    _BIRD_SIZE_CACHE.write_text(json.dumps(sizes), encoding="utf-8")
    return sizes


def load_bird(limit: int = 60, *, seed: int = 20261009, max_db_bytes: int = _BIRD_MAX_DB_BYTES) -> list[SqlItem]:
    """BIRD (public Text-to-SQL bench): train split, stratified by database.

    Databases over ``max_db_bytes`` are excluded from the sample.
    """

    import json

    data = json.loads((_BIRD_DIR / "train.json").read_text(encoding="utf-8"))
    sizes = _bird_db_sizes()
    rng = random.Random(seed)
    by_db: dict[str, list[dict]] = {}
    for row in data:
        if sizes.get(row["db_id"], 0) <= max_db_bytes:
            by_db.setdefault(row["db_id"], []).append(row)
    dbs = sorted(by_db)
    rng.shuffle(dbs)
    items: list[SqlItem] = []
    serial = 0
    for db_id in dbs:
        rows = by_db[db_id]
        rng.shuffle(rows)
        take = min(3, len(rows), max(1, limit // len(dbs) + 1))
        for row in rows[:take]:
            serial += 1
            items.append(
                SqlItem(
                    id=f"bird-{serial:03d}",
                    question=row["question"] + (f" Hint: {row['evidence']}" if row.get("evidence") else ""),
                    db_file=row["db_id"],
                    gold_sql=row["SQL"],
                )
            )
        if len(items) >= limit:
            break
    return items[:limit]


def bird_db_path(db_id: str) -> str:
    """BIRD sqlite path, downloaded on demand from the mirror repo."""

    local = _BIRD_DIR / "databases" / db_id / f"{db_id}.sqlite"
    if not local.exists():
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            _BIRD_DATASET,
            f"train/train_databases/{db_id}/{db_id}.sqlite",
            repo_type="dataset",
        )
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(Path(downloaded).read_bytes())
    return str(local)
