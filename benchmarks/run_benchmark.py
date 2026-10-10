"""Generic benchmark runner for XuanJi (test-driven development harness).

Usage (from the repo root, e2e environment):

    XUANJI_BACKEND=anthropic .venv/Scripts/python.exe benchmarks/run_benchmark.py \
        --benchmark wtq --n 100 [--sc 3] [--mode auto] [--out report.json]

The runner is deliberately benchmark-agnostic: loaders yield BenchmarkItem
rows, the engine answers, denotation accuracy is scored with the shared
normalizer, and a per-case JSON report is written for failure analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loader import (  # noqa: E402
    bird_db_path,
    clambsql_db_path,
    load_bird,
    load_clambsql,
    load_tabfact,
    load_wtq,
    load_wtq_table,
)


def _norm_value(text: str) -> str:
    """Loose value normalization for gold-coverage scoring."""

    import re as _re

    s = str(text).strip().casefold()
    s = _re.sub(r"[\s_]+", " ", s)
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else f"{f:.4g}"
    except ValueError:
        return s


def gold_coverage(answer: str, gold_result: str) -> float:
    """Fraction of gold result cell values present in the answer text.

    Database-benchmark scoring: the engine answers in prose while gold is
    SQL-execution rows; a fair check is whether every gold value appears in
    the answer (header lines excluded), number/case normalized.
    """

    lines = [line for line in str(gold_result).splitlines() if line.strip()]
    if not lines:
        return 0.0
    # First line is the SQL result header (column names) — never a value.
    lines = lines[1:] if len(lines) > 1 else lines
    cells: set[str] = set()
    for line in lines:
        for cell in line.split(" | "):
            cell = cell.strip()
            if cell and cell.casefold() not in {"none", "null", "nan"}:
                cells.add(_norm_value(cell))
    if not cells:
        return 0.0
    # Header values equal to row values collapse into the cell set anyway.
    answer_norm = _norm_value(answer)
    hits = sum(1 for value in cells if value in answer_norm)
    return hits / len(cells)


def run(benchmark: str, n: int, sc: int, mode: str, out: str, seed: int) -> dict:
    import xuanji
    from xuanji.eval import denotation_accuracy
    from xuanji.table import Table

    table_map: dict = {}
    db_cache: dict = {}
    if benchmark == "wtq":
        items = load_wtq(n, seed=seed)
        task = "tqa"
    elif benchmark == "tabfact":
        items, table_map = load_tabfact(n, seed=seed)
        task = "tfv"
    elif benchmark == "clambsql":
        items = load_clambsql(n, seed=seed)
        task = "tqa"
    elif benchmark == "bird":
        items = load_bird(n, seed=seed)
        task = "tqa"
    else:
        raise ValueError(f"unknown benchmark: {benchmark!r}")

    engine = xuanji.XuanJi(mode=mode, task=task, solver_kwargs={"sc": sc})
    llm = engine.llm

    report: list[dict] = []
    predictions: list[str] = []
    labels: list[list[str]] = []
    total_seconds = 0.0
    for index, item in enumerate(items, 1):
        start = time.perf_counter()
        error = None
        try:
            if benchmark in {"clambsql", "bird"}:
                from xuanji.tools.db import Database

                if item.db_file not in db_cache:
                    path_fn = clambsql_db_path if benchmark == "clambsql" else bird_db_path
                    db_cache[item.db_file] = Database.from_sqlite(
                        path_fn(item.db_file)
                    )
                db = db_cache[item.db_file]
                gold_answer = db.execute(item.gold_sql)
                gold = [gold_answer] if gold_answer else [""]
                result = engine.ask(item.question, db=db, task=task)
            else:
                if benchmark == "wtq":
                    rows = load_wtq_table(item.table_path).to_rows()
                else:
                    rows = table_map[item.table_path].to_rows()
                result = engine.ask(
                    item.question,
                    table=Table.from_obj(rows),
                    caption=item.caption,
                    task=task,
                )
            answer, norm, status, steps = (
                result.answer, result.norm, result.status, result.steps,
            )
        except Exception as exc:  # noqa: BLE001 - benchmark records, not raises
            answer, norm, status, steps = "", "", f"{type(exc).__name__}: {exc}", 0
            error = status
        elapsed = time.perf_counter() - start
        total_seconds += elapsed
        predictions.append(norm or answer)
        if benchmark in {"clambsql", "bird"}:
            labels.append([part.strip() for part in str(gold[0]).splitlines() if part.strip()])
        else:
            labels.append(list(item.answers))
        report.append(
            {
                "id": item.id,
                "question": item.question,
                "gold": labels[-1],
                "answer": answer,
                "norm": norm,
                "status": status,
                "steps": steps,
                "seconds": round(elapsed, 1),
                "error": error,
            }
        )
        print(
            f"[{index}/{len(items)}] {item.id} {status} "
            f"answer={norm or answer!r} gold={labels[-1]!r} ({elapsed:.0f}s)",
            flush=True,
        )

    if benchmark in {"clambsql", "bird"}:
        # value-coverage scoring for prose-vs-SQL-result comparison
        accuracy = sum(
            gold_coverage(pred, " | ".join(gold_vals))
            for pred, gold_vals in zip(predictions, labels)
        ) / max(len(predictions), 1)
    else:
        accuracy = denotation_accuracy(predictions, labels)
    summary = {
        "benchmark": benchmark,
        "n": len(items),
        "accuracy": accuracy,
        "seconds": round(total_seconds, 1),
        "llm_calls": getattr(llm, "n_calls", None),
        "llm_tokens": getattr(llm, "n_tokens", None),
        "cases": report,
    }
    Path(out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"\n{benchmark}: accuracy {accuracy:.3f} on {len(items)} items "
        f"({total_seconds:.0f}s; llm calls={summary['llm_calls']}, "
        f"tokens={summary['llm_tokens']})"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="wtq", choices=["wtq", "tabfact", "clambsql", "bird"])
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--sc", type=int, default=3)
    parser.add_argument("--mode", default="auto", choices=["auto", "solve", "refine"])
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--out", default="benchmarks/wtq_report.json")
    args = parser.parse_args()
    run(args.benchmark, args.n, args.sc, args.mode, args.out, args.seed)


if __name__ == "__main__":
    main()
