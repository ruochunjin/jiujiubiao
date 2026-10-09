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
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loader import load_tabfact, load_wtq, load_wtq_table  # noqa: E402


def run(benchmark: str, n: int, sc: int, mode: str, out: str, seed: int) -> dict:
    import xuanji
    from xuanji.eval import denotation_accuracy
    from xuanji.table import Table

    table_map: dict = {}
    if benchmark == "wtq":
        items = load_wtq(n, seed=seed)
        task = "tqa"
    elif benchmark == "tabfact":
        items, table_map = load_tabfact(n, seed=seed)
        task = "tfv"
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
        labels.append(list(item.answers))
        report.append(
            {
                "id": item.id,
                "question": item.question,
                "gold": list(item.answers),
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
            f"answer={norm or answer!r} gold={item.answers!r} ({elapsed:.0f}s)",
            flush=True,
        )

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
    parser.add_argument("--benchmark", default="wtq", choices=["wtq", "tabfact"])
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--sc", type=int, default=3)
    parser.add_argument("--mode", default="auto", choices=["auto", "solve", "refine"])
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--out", default="benchmarks/wtq_report.json")
    args = parser.parse_args()
    run(args.benchmark, args.n, args.sc, args.mode, args.out, args.seed)


if __name__ == "__main__":
    main()
