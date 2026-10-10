"""Re-run WTQ missed cases with full trajectory capture for failure analysis.

Usage:
    XUANJI_BACKEND=anthropic .venv/Scripts/python.exe benchmarks/debug_misses.py [wtq_report.json]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loader import load_wtq, load_wtq_table  # noqa: E402


def main() -> None:
    import xuanjitu
    from xuanjitu.eval import _norm_parts
    from xuanjitu.table import Table

    report_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmarks/wtq_100.json")
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
    missed_ids = []
    for c in rep["cases"]:
        pred = c["norm"] or c["answer"]
        pp = _norm_parts(pred, "tqa")
        gp = frozenset().union(*(_norm_parts(g, "tqa") for g in c["gold"]))
        if pp != gp and not (len(pp) == 1 and pp <= gp):
            if only is None or c["id"] in only:
                missed_ids.append(c["id"])
    print(f"re-running {len(missed_ids)} missed cases with trajectories")

    items = load_wtq(100)  # same seed -> same sample
    engine = xuanjitu.XuanJi(mode="auto", solver_kwargs={"sc": 3})
    out = []
    for item in items:
        if item.id not in missed_ids:
            continue
        start = time.perf_counter()
        result = engine.ask(item.question, table=Table.from_obj(load_wtq_table(item.table_path).to_rows()))
        elapsed = time.perf_counter() - start
        trajectory = ""
        if result.history and isinstance(result.history[0], dict):
            trajectory = result.history[0].get("trajectory", "")[:4000]
            if not trajectory:
                # refiner: per-stage answers/votes/reasoning snippets
                trajectory = json.dumps(
                    [
                        {k: str(v)[:600] for k, v in stage.items() if k in ("stage", "answer", "votes", "reasoning", "winner_tool", "per_tool")}
                        for stage in result.history
                    ],
                    ensure_ascii=False, indent=1,
                )[:4000]
        out.append(
            {
                "id": item.id,
                "question": item.question,
                "gold": list(item.answers),
                "answer": result.answer,
                "norm": result.norm,
                "mode": result.mode,
                "seconds": round(elapsed, 1),
                "trajectory_tail": trajectory[-3000:],
            }
        )
        print(f"{item.id}: {result.norm or result.answer!r} vs gold {item.answers!r} ({result.mode}, {elapsed:.0f}s)", flush=True)
    Path("benchmarks/wtq_misses_debug.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("written to benchmarks/wtq_misses_debug.json")


if __name__ == "__main__":
    main()
