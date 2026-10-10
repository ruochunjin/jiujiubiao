"""CLAMBSQL disambiguation A/B: weak types (table/vague) with and without
the disambiguator + benchmark clarification context.

Usage:
    XUANJI_BACKEND=anthropic .venv/Scripts/python.exe benchmarks/clambsql_disambig.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_benchmark import gold_coverage  # noqa: E402


def main() -> None:
    import xuanjitu
    from xuanjitu.tools.db import Database

    data = json.loads(Path("benchmarks/data/clambsql/clambsql.json").read_text(encoding="utf-8"))
    # same stratified sample as the clambsql_60 run: take the weak types'
    # first-9 shuffled items (same seed logic => overlapping subset)
    import random

    rng = random.Random(20261009)
    rows = [r for r in data if r["ambig_type"] in {"table", "vague"}]
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["ambig_type"], []).append(r)
    picked = []
    for t in sorted(by_type):
        rs = by_type[t]
        rng.shuffle(rs)
        picked.extend(rs[:8])
    print(f"A/B on {len(picked)} weak-type items")

    db_cache: dict[str, Database] = {}
    results = {"direct": [], "disambiguated": []}
    for row in picked:
        db_file = row["db_file"]
        if db_file not in db_cache:
            db_cache[db_file] = Database.from_sqlite(f"benchmarks/data/clambsql/database/{db_file}")
        db = db_cache[db_file]
        gold = db.execute(row["gold_query"])

        direct = xuanjitu.XuanJi()
        t0 = time.perf_counter()
        r1 = direct.ask(row["question"], db=db)
        cov1 = gold_coverage(r1.norm or r1.answer, gold)

        clarify_answer = row.get("clarification_context") or ""
        engine2 = xuanjitu.XuanJi(disambiguate=True, clarify=lambda _q, a=clarify_answer: a)
        r2 = engine2.ask(row["question"], db=db)
        cov2 = gold_coverage(r2.norm or r2.answer, gold)
        _elapsed = time.perf_counter() - t0  # noqa: F841 - timing kept for debug

        results["direct"].append(cov1)
        results["disambiguated"].append(cov2)
        print(
            f"{row['ambig_type']}-{row['index']}: direct {cov1:.2f} vs disambig {cov2:.2f}"
            + (f" (rewrite: {r2.disambiguation.question[:60]})" if r2.disambiguation and r2.disambiguation.question != row["question"] else ""),
            flush=True,
        )

    for name, covs in results.items():
        print(f"{name}: mean coverage {sum(covs) / len(covs):.3f} on {len(covs)}")
    Path("benchmarks/clambsql_disambig_ab.json").write_text(
        json.dumps(
            {
                "direct": results["direct"],
                "disambiguated": results["disambiguated"],
                "direct_mean": sum(results["direct"]) / len(results["direct"]),
                "disambiguated_mean": sum(results["disambiguated"]) / len(results["disambiguated"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
